// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/InsertSoftSync.cpp
 * \brief Insert synchronization for soft-pipeline cross-stage communication.
 *
 * This pass analyzes soft-pipeline IR and automatically inserts
 * sync_block_set/wait operations for cross-stage communication buffers.
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRINSERTSOFTSYNC
#include "tilelangir/Transforms/Passes.h.inc"

namespace {
#define DEBUG_TYPE "tilelangir-insert-soft-sync"

static constexpr size_t SYNC_FLAGS_LIMIT = 16;

struct StageInfo {
  scf::IfOp stageIf;
  hivm::TCoreType coreType;
  int stageOrder;
  Value slotIndex;
};

struct BufferSyncInfo {
  Value buffer;
  size_t multiBufferDepth;
  size_t flagBase;
  StageInfo producerStage;
  StageInfo consumerStage;
  bool hasProducer = false;
  bool hasConsumer = false;
};

static Value convertToI64(OpBuilder &builder, Location loc, Value val) {
  auto type = val.getType();
  if (type.isInteger(64))
    return val;
  if (type.isIndex())
    return builder.create<arith::IndexCastOp>(loc, builder.getI64Type(), val);
  if (type.isInteger())
    return builder.create<arith::ExtSIOp>(loc, builder.getI64Type(), val);
  llvm_unreachable("Unsupported type for conversion to i64");
}

static Value getRootBuffer(Value val) {
  while (auto subviewOp = val.getDefiningOp<memref::SubViewOp>()) {
    val = subviewOp.getSource();
  }
  while (auto collapseOp = val.getDefiningOp<memref::CollapseShapeOp>()) {
    val = collapseOp.getViewSource();
  }
  while (auto reinterpretOp = val.getDefiningOp<memref::ReinterpretCastOp>()) {
    val = reinterpretOp.getViewSource();
  }
  return val;
}

static bool isCommunicationBuffer(Value val) {
  Value rootVal = getRootBuffer(val);
  auto memrefType = rootVal.getType().dyn_cast<MemRefType>();
  if (!memrefType)
    return false;

  if (memrefType.getShape().size() >= 1 && memrefType.getShape()[0] >= 2) {
    auto addrSpace = hivm::getOptionalHIVMAddressSpace(memrefType);
    if (addrSpace.has_value()) {
      if (*addrSpace == hivm::AddressSpace::GM ||
          *addrSpace == hivm::AddressSpace::UB) {
        return true;
      }
    }
  }

  return false;
}

static size_t getMultiBufferDepth(Value buffer) {
  auto memrefType = buffer.getType().dyn_cast<MemRefType>();
  if (!memrefType || memrefType.getShape().empty())
    return 1;

  if (memrefType.getShape()[0] >= 2) {
    return memrefType.getShape()[0];
  }
  return 1;
}

static Value extractSlotIndexFromSubview(Value val) {
  if (auto subviewOp = val.getDefiningOp<memref::SubViewOp>()) {
    if (subviewOp.getMixedOffsets().size() > 0) {
      auto firstOffset = subviewOp.getMixedOffsets()[0];
      if (auto offsetVal = firstOffset.dyn_cast<Value>()) {
        return offsetVal;
      }
    }
  }
  return Value();
}

static void buildSyncBlockSet(OpBuilder &builder, Location loc,
                              hivm::TCoreType coreType, Value flagId) {
  auto coreTypeAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  auto syncMode = hivm::SyncBlockInstrModeAttr::get(
      builder.getContext(),
      hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr = hivm::PipeAttr::get(
      builder.getContext(),
      coreType == hivm::TCoreType::VECTOR ? hivm::PIPE::PIPE_MTE3
                                          : hivm::PIPE::PIPE_FIX);

  builder.create<hivm::SyncBlockSetOp>(loc, coreTypeAttr, pipTypeAttr,
                                       tPipTypeAttr, flagId, Value(),
                                       syncMode);
}

static void buildSyncBlockWait(OpBuilder &builder, Location loc,
                               hivm::TCoreType coreType, Value flagId) {
  auto coreTypeAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), coreType);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE2);

  builder.create<hivm::SyncBlockWaitOp>(loc, coreTypeAttr, tPipTypeAttr,
                                        pipTypeAttr, flagId);
}

static SmallVector<scf::IfOp> collectStageIfOps(scf::ForOp pipelineFor) {
  SmallVector<scf::IfOp> stageIfOps;
  for (Operation &op : pipelineFor.getBody()->getOperations()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      if (ifOp->getAttr(hivm::TCoreTypeAttr::name)) {
        stageIfOps.push_back(ifOp);
      }
    }
  }
  return stageIfOps;
}

struct TileLangIRInsertSoftSync
    : impl::TileLangIRInsertSoftSyncBase<TileLangIRInsertSoftSync> {
  void runOnOperation() override;

private:
  size_t nextFlagBase = 0;

  SmallVector<scf::ForOp> findPipelineLoops(ModuleOp module);
  SmallVector<BufferSyncInfo> analyzePipelineBuffers(scf::ForOp pipelineFor);
  void insertStageSync(BufferSyncInfo &bsi, scf::ForOp pipelineFor);
  void insertInitAndCleanup(SmallVector<BufferSyncInfo> &buffers,
                            scf::ForOp pipelineFor);
};

void TileLangIRInsertSoftSync::runOnOperation() {
  ModuleOp module = getOperation();

  auto pipelineLoops = findPipelineLoops(module);
  if (pipelineLoops.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No pipeline loops found\n");
    return;
  }

  for (scf::ForOp pipelineFor : pipelineLoops) {
    nextFlagBase = 0;

    auto buffers = analyzePipelineBuffers(pipelineFor);

    for (auto &buffer : buffers) {
      insertStageSync(buffer, pipelineFor);
    }

    insertInitAndCleanup(buffers, pipelineFor);
  }
}

SmallVector<scf::ForOp>
TileLangIRInsertSoftSync::findPipelineLoops(ModuleOp module) {
  SmallVector<scf::ForOp> pipelineLoops;

  module.walk([&](scf::ForOp forOp) {
    if (forOp->getAttr("tilelangir.num_stages")) {
      pipelineLoops.push_back(forOp);
    }
  });

  return pipelineLoops;
}

SmallVector<BufferSyncInfo>
TileLangIRInsertSoftSync::analyzePipelineBuffers(scf::ForOp pipelineFor) {
  SmallVector<BufferSyncInfo> bufferInfos;
  DenseMap<Value, BufferSyncInfo> bufferMap;

  auto stageIfOps = collectStageIfOps(pipelineFor);

  int stageOrder = 0;
  for (scf::IfOp stageIf : stageIfOps) {
    auto coreTypeAttr = stageIf->getAttrOfType<hivm::TCoreTypeAttr>(
        hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr)
      continue;

    auto coreType = coreTypeAttr.getTcoretype();

    bool isWriteToBuffer = false;
    bool isReadFromBuffer = false;
    Value buffer;
    Value slotIndex;

    stageIf.thenBlock()->walk([&](Operation *op) {
      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        Value srcRoot = getRootBuffer(copyOp.getSource());
        Value dstRoot = getRootBuffer(copyOp.getTarget());

        if (isCommunicationBuffer(dstRoot)) {
          isWriteToBuffer = true;
          buffer = dstRoot;
          slotIndex = extractSlotIndexFromSubview(copyOp.getTarget());
        }
        if (isCommunicationBuffer(srcRoot)) {
          isReadFromBuffer = true;
          buffer = srcRoot;
          slotIndex = extractSlotIndexFromSubview(copyOp.getSource());
        }
      }

      if (op->getName().getStringRef().contains("fixpipe")) {
        for (Value operand : op->getOperands()) {
          Value root = getRootBuffer(operand);
          if (isCommunicationBuffer(root)) {
            isWriteToBuffer = true;
            buffer = root;
            slotIndex = extractSlotIndexFromSubview(operand);
            break;
          }
        }
      }

      if (op->getName().getStringRef().contains("nd2nz")) {
        for (Value result : op->getResults()) {
          for (auto &use : result.getUses()) {
            if (auto subviewOp = dyn_cast<memref::SubViewOp>(use.getOwner())) {
              for (Value operand : subviewOp->getOperands()) {
                if (operand == result) {
                  Value root = getRootBuffer(subviewOp.getResult());
                  if (isCommunicationBuffer(root)) {
                    isReadFromBuffer = true;
                    buffer = root;
                    slotIndex = extractSlotIndexFromSubview(subviewOp.getResult());
                  }
                }
              }
            }
          }
        }
        for (Value operand : op->getOperands()) {
          Value root = getRootBuffer(operand);
          if (isCommunicationBuffer(root)) {
            isReadFromBuffer = true;
            buffer = root;
            slotIndex = extractSlotIndexFromSubview(operand);
            break;
          }
        }
      }
    });

    if (buffer) {
      if (!bufferMap.count(buffer)) {
        BufferSyncInfo bsi;
        bsi.buffer = buffer;
        bsi.multiBufferDepth = getMultiBufferDepth(buffer);
        bsi.flagBase = nextFlagBase;
        nextFlagBase += bsi.multiBufferDepth * 2;
        bufferMap[buffer] = bsi;
      }

      BufferSyncInfo &bsi = bufferMap[buffer];

      if (isWriteToBuffer) {
        StageInfo si;
        si.stageIf = stageIf;
        si.coreType = coreType;
        si.stageOrder = stageOrder;
        si.slotIndex = slotIndex;
        bsi.producerStage = si;
        bsi.hasProducer = true;
      }

      if (isReadFromBuffer && !isWriteToBuffer) {
        StageInfo si;
        si.stageIf = stageIf;
        si.coreType = coreType;
        si.stageOrder = stageOrder;
        si.slotIndex = slotIndex;
        bsi.consumerStage = si;
        bsi.hasConsumer = true;
      }
    }

    stageOrder++;
  }

  for (auto &[val, bsi] : bufferMap) {
    if (bsi.hasProducer && bsi.hasConsumer &&
        bsi.producerStage.coreType != bsi.consumerStage.coreType) {
      bufferInfos.push_back(bsi);
    }
  }

  return bufferInfos;
}

void TileLangIRInsertSoftSync::insertStageSync(BufferSyncInfo &bsi,
                                                scf::ForOp pipelineFor) {
  if (!bsi.hasProducer || !bsi.hasConsumer)
    return;

  OpBuilder builder(pipelineFor->getContext());
  Location loc = pipelineFor.getLoc();

  Value slotIndex = bsi.producerStage.slotIndex
                        ? bsi.producerStage.slotIndex
                        : bsi.consumerStage.slotIndex;
  if (!slotIndex) {
    LLVM_DEBUG(llvm::dbgs() << "No slot index found for buffer\n");
    return;
  }

  auto insertSyncAtStageStart = [&](scf::IfOp stageIf, hivm::TCoreType coreType,
                                    Value flagId) {
    builder.setInsertionPointToStart(stageIf.thenBlock());
    buildSyncBlockWait(builder, loc, coreType, flagId);
  };

  auto insertSyncAtStageEnd = [&](scf::IfOp stageIf, hivm::TCoreType coreType,
                                  Value flagId) {
    Block *thenBlock = stageIf.thenBlock();
    builder.setInsertionPoint(thenBlock->getTerminator());
    buildSyncBlockSet(builder, loc, coreType, flagId);
  };

  auto createFlagValue = [&](size_t base, Value slot) -> Value {
    Value slotI64 = convertToI64(builder, loc, slot);
    Value flagBase = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(base));
    Value flag = builder.create<arith::AddIOp>(loc, flagBase, slotI64);
    Value syncLimit = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
    return builder.create<arith::RemSIOp>(loc, flag, syncLimit);
  };

  Value freeFlag = createFlagValue(bsi.flagBase + bsi.multiBufferDepth, slotIndex);
  Value readyFlag = createFlagValue(bsi.flagBase, slotIndex);

  insertSyncAtStageStart(bsi.producerStage.stageIf, bsi.producerStage.coreType,
                         freeFlag);

  freeFlag = createFlagValue(bsi.flagBase + bsi.multiBufferDepth, slotIndex);
  readyFlag = createFlagValue(bsi.flagBase, slotIndex);

  insertSyncAtStageEnd(bsi.producerStage.stageIf, bsi.producerStage.coreType,
                       readyFlag);

  readyFlag = createFlagValue(bsi.flagBase, slotIndex);

  insertSyncAtStageStart(bsi.consumerStage.stageIf, bsi.consumerStage.coreType,
                         readyFlag);

  freeFlag = createFlagValue(bsi.flagBase + bsi.multiBufferDepth, slotIndex);

  insertSyncAtStageEnd(bsi.consumerStage.stageIf, bsi.consumerStage.coreType,
                       freeFlag);
}

void TileLangIRInsertSoftSync::insertInitAndCleanup(
    SmallVector<BufferSyncInfo> &buffers, scf::ForOp pipelineFor) {
  if (buffers.empty())
    return;

  OpBuilder builder(pipelineFor->getContext());
  Location loc = pipelineFor.getLoc();

  builder.setInsertionPoint(pipelineFor);

  for (auto &bsi : buffers) {
    for (size_t slot = 0; slot < bsi.multiBufferDepth; ++slot) {
      Value slotVal = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(slot));

      Value freeFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(bsi.flagBase + bsi.multiBufferDepth));
      Value freeFlag = builder.create<arith::AddIOp>(loc, freeFlagBase, slotVal);

      Value syncLimit = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      freeFlag = builder.create<arith::RemSIOp>(loc, freeFlag, syncLimit);

      hivm::TCoreType initCoreType = hivm::TCoreType::VECTOR;
      buildSyncBlockSet(builder, loc, initCoreType, freeFlag);
    }
  }

  builder.setInsertionPointAfter(pipelineFor);

  for (auto &bsi : buffers) {
    for (size_t slot = 0; slot < bsi.multiBufferDepth; ++slot) {
      Value slotVal = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(slot));

      Value freeFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(bsi.flagBase + bsi.multiBufferDepth));
      Value freeFlag = builder.create<arith::AddIOp>(loc, freeFlagBase, slotVal);

      Value syncLimit = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      freeFlag = builder.create<arith::RemSIOp>(loc, freeFlag, syncLimit);

      hivm::TCoreType cleanupCoreType = hivm::TCoreType::CUBE;
      buildSyncBlockWait(builder, loc, cleanupCoreType, freeFlag);
    }
  }
}

#undef DEBUG_TYPE
} // namespace

} // namespace tilelangir
} // namespace mlir
