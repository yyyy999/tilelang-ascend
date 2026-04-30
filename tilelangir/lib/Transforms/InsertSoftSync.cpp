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

struct BufferAccessInfo {
  Value buffer;
  Operation *accessOp;
  scf::IfOp stageIf;
  hivm::TCoreType coreType;
  bool isWrite;
  Value slotIndex;
  int stageOrder;
};

struct ProducerConsumerPair {
  BufferAccessInfo producer;
  BufferAccessInfo consumer;
};

struct CommunicationBuffer {
  Value buffer;
  size_t multiBufferDepth;
  size_t flagBase;
  SmallVector<ProducerConsumerPair> pairs;
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

static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = constOp.getValue().dyn_cast_or_null<IntegerAttr>())
      return intAttr.getInt();
  }
  return std::nullopt;
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

static bool isWriteOperation(Operation *op) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    return true;
  }
  if (op->hasTrait<OpTrait::MemWrite>()) {
    return true;
  }
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    auto resultType = op->getResult(i).getType();
    if (auto memrefType = resultType.dyn_cast<MemRefType>()) {
      return true;
    }
  }
  StringRef opName = op->getName().getStringRef();
  if (opName.contains("fixpipe") || opName.contains("copy") ||
      opName.contains("vbrc") || opName.contains("vcast")) {
    return true;
  }
  return false;
}

static bool isReadOperation(Operation *op) {
  if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
    return true;
  }
  if (op->hasTrait<OpTrait::MemRead>()) {
    return true;
  }
  for (Value operand : op->getOperands()) {
    if (operand.getType().isa<MemRefType>()) {
      return true;
    }
  }
  return false;
}

static bool isCommunicationBuffer(Value val) {
  auto memrefType = val.getType().dyn_cast<MemRefType>();
  if (!memrefType)
    return false;

  if (auto spaceAttr =
          memrefType.getMemorySpace().dyn_cast_or_null<hivm::AddressSpaceAttr>()) {
    auto space = spaceAttr.getValue();
    if (space == hivm::AddressSpace::gm) {
      return true;
    }
  }

  if (memrefType.getShape().size() >= 1 && memrefType.getShape()[0] >= 2) {
    if (auto spaceAttr = memrefType.getMemorySpace()
                             .dyn_cast_or_null<hivm::AddressSpaceAttr>()) {
      if (spaceAttr.getValue() == hivm::AddressSpace::ub) {
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
  SmallVector<CommunicationBuffer>
  collectCommunicationBuffers(scf::ForOp pipelineFor);
  void buildProducerConsumerPairs(CommunicationBuffer &cb,
                                   scf::ForOp pipelineFor);
  void insertSynchronization(CommunicationBuffer &cb, scf::ForOp pipelineFor);
  void insertInitAndCleanup(SmallVector<CommunicationBuffer> &buffers,
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

    auto buffers = collectCommunicationBuffers(pipelineFor);

    for (auto &buffer : buffers) {
      buildProducerConsumerPairs(buffer, pipelineFor);
    }

    for (auto &buffer : buffers) {
      insertSynchronization(buffer, pipelineFor);
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

SmallVector<CommunicationBuffer>
TileLangIRInsertSoftSync::collectCommunicationBuffers(scf::ForOp pipelineFor) {
  SmallVector<CommunicationBuffer> buffers;
  DenseSet<Value> seenBuffers;

  auto stageIfOps = collectStageIfOps(pipelineFor);
  for (scf::IfOp stageIf : stageIfOps) {
    stageIf.thenBlock()->walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        if (isCommunicationBuffer(operand) && !seenBuffers.count(operand)) {
          seenBuffers.insert(operand);
          CommunicationBuffer cb;
          cb.buffer = operand;
          cb.multiBufferDepth = getMultiBufferDepth(operand);
          cb.flagBase = nextFlagBase;
          nextFlagBase += cb.multiBufferDepth * 2;
          buffers.push_back(cb);
        }
      }
      for (Value result : op->getResults()) {
        if (isCommunicationBuffer(result) && !seenBuffers.count(result)) {
          seenBuffers.insert(result);
          CommunicationBuffer cb;
          cb.buffer = result;
          cb.multiBufferDepth = getMultiBufferDepth(result);
          cb.flagBase = nextFlagBase;
          nextFlagBase += cb.multiBufferDepth * 2;
          buffers.push_back(cb);
        }
      }
    });
  }

  return buffers;
}

void TileLangIRInsertSoftSync::buildProducerConsumerPairs(
    CommunicationBuffer &cb, scf::ForOp pipelineFor) {
  auto stageIfOps = collectStageIfOps(pipelineFor);

  SmallVector<BufferAccessInfo> writes;
  SmallVector<BufferAccessInfo> reads;

  int stageOrder = 0;
  for (scf::IfOp stageIf : stageIfOps) {
    auto coreTypeAttr = stageIf->getAttrOfType<hivm::TCoreTypeAttr>(
        hivm::TCoreTypeAttr::name);
    if (!coreTypeAttr)
      continue;

    auto coreType = coreTypeAttr.getTcoretype();

    stageIf.thenBlock()->walk([&](Operation *op) {
      bool isWriteToBuffer = false;
      bool isReadFromBuffer = false;
      Value slotIndex;

      for (Value operand : op->getOperands()) {
        if (operand == cb.buffer) {
          isReadFromBuffer = true;
        }
        if (auto subviewOp = operand.getDefiningOp<memref::SubViewOp>()) {
          if (subviewOp.getSource() == cb.buffer) {
            isReadFromBuffer = true;
            if (subviewOp.getMixedOffsets().size() > 0) {
              auto firstOffset = subviewOp.getMixedOffsets()[0];
              if (auto offsetVal = firstOffset.dyn_cast<Value>()) {
                slotIndex = offsetVal;
              }
            }
          }
        }
      }

      for (Value result : op->getResults()) {
        if (result == cb.buffer) {
          isWriteToBuffer = true;
        }
        for (auto &use : result.getUses()) {
          if (auto subviewOp = dyn_cast<memref::SubViewOp>(use.getOwner())) {
            if (subviewOp.getResult() == cb.buffer) {
              isWriteToBuffer = true;
            }
          }
        }
      }

      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        if (auto subviewOp =
                copyOp.getSource().getDefiningOp<memref::SubViewOp>()) {
          if (subviewOp.getSource() == cb.buffer) {
            isReadFromBuffer = true;
            if (subviewOp.getMixedOffsets().size() > 0) {
              auto firstOffset = subviewOp.getMixedOffsets()[0];
              if (auto offsetVal = firstOffset.dyn_cast<Value>()) {
                slotIndex = offsetVal;
              }
            }
          }
        }
        if (auto subviewOp =
                copyOp.getTarget().getDefiningOp<memref::SubViewOp>()) {
          if (subviewOp.getSource() == cb.buffer) {
            isWriteToBuffer = true;
            if (subviewOp.getMixedOffsets().size() > 0) {
              auto firstOffset = subviewOp.getMixedOffsets()[0];
              if (auto offsetVal = firstOffset.dyn_cast<Value>()) {
                slotIndex = offsetVal;
              }
            }
          }
        }
      }

      if (isWriteToBuffer) {
        BufferAccessInfo info;
        info.buffer = cb.buffer;
        info.accessOp = op;
        info.stageIf = stageIf;
        info.coreType = coreType;
        info.isWrite = true;
        info.slotIndex = slotIndex;
        info.stageOrder = stageOrder;
        writes.push_back(info);
      }

      if (isReadFromBuffer && !isWriteToBuffer) {
        BufferAccessInfo info;
        info.buffer = cb.buffer;
        info.accessOp = op;
        info.stageIf = stageIf;
        info.coreType = coreType;
        info.isWrite = false;
        info.slotIndex = slotIndex;
        info.stageOrder = stageOrder;
        reads.push_back(info);
      }
    });

    stageOrder++;
  }

  for (auto &write : writes) {
    for (auto &read : reads) {
      if (read.stageOrder > write.stageOrder) {
        ProducerConsumerPair pair;
        pair.producer = write;
        pair.consumer = read;
        cb.pairs.push_back(pair);
        break;
      }
    }
  }
}

void TileLangIRInsertSoftSync::insertSynchronization(CommunicationBuffer &cb,
                                                      scf::ForOp pipelineFor) {
  OpBuilder builder(pipelineFor->getContext());
  Location loc = pipelineFor.getLoc();

  for (auto &pair : cb.pairs) {
    auto &producer = pair.producer;
    auto &consumer = pair.consumer;

    Value slotIndex = producer.slotIndex ? producer.slotIndex : consumer.slotIndex;
    if (!slotIndex) {
      LLVM_DEBUG(llvm::dbgs() << "No slot index found for buffer\n");
      continue;
    }

    {
      builder.setInsertionPoint(producer.accessOp);
      Value slotIndexI64 = convertToI64(builder, loc, slotIndex);

      Value readyFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase));
      Value freeFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase + cb.multiBufferDepth));

      Value freeFlag = builder.create<arith::AddIOp>(loc, freeFlagBase, slotIndexI64);

      Value syncLimit = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      freeFlag = builder.create<arith::RemSIOp>(loc, freeFlag, syncLimit);

      buildSyncBlockWait(builder, loc, producer.coreType, freeFlag);
    }

    {
      builder.setInsertionPointAfter(producer.accessOp);
      Value slotIndexI64 = convertToI64(builder, loc, slotIndex);

      Value readyFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase));

      Value readyFlag = builder.create<arith::AddIOp>(loc, readyFlagBase, slotIndexI64);

      Value syncLimit = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      readyFlag = builder.create<arith::RemSIOp>(loc, readyFlag, syncLimit);

      buildSyncBlockSet(builder, loc, producer.coreType, readyFlag);
    }

    {
      builder.setInsertionPoint(consumer.accessOp);
      Value slotIndexI64 = convertToI64(builder, loc, slotIndex);

      Value readyFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase));

      Value readyFlag = builder.create<arith::AddIOp>(loc, readyFlagBase, slotIndexI64);

      Value syncLimit = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      readyFlag = builder.create<arith::RemSIOp>(loc, readyFlag, syncLimit);

      buildSyncBlockWait(builder, loc, consumer.coreType, readyFlag);
    }

    {
      builder.setInsertionPointAfter(consumer.accessOp);
      Value slotIndexI64 = convertToI64(builder, loc, slotIndex);

      Value freeFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase + cb.multiBufferDepth));

      Value freeFlag = builder.create<arith::AddIOp>(loc, freeFlagBase, slotIndexI64);

      Value syncLimit = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      freeFlag = builder.create<arith::RemSIOp>(loc, freeFlag, syncLimit);

      buildSyncBlockSet(builder, loc, consumer.coreType, freeFlag);
    }
  }
}

void TileLangIRInsertSoftSync::insertInitAndCleanup(
    SmallVector<CommunicationBuffer> &buffers, scf::ForOp pipelineFor) {
  if (buffers.empty())
    return;

  OpBuilder builder(pipelineFor->getContext());
  Location loc = pipelineFor.getLoc();

  auto numStagesAttr = pipelineFor->getAttr("tilelangir.num_stages");
  if (!numStagesAttr) {
    pipelineFor.emitError("Missing tilelangir.num_stages attribute");
    return;
  }

  int64_t numStages = numStagesAttr.cast<IntegerAttr>().getInt();

  builder.setInsertionPoint(pipelineFor);

  for (auto &cb : buffers) {
    for (size_t slot = 0; slot < cb.multiBufferDepth; ++slot) {
      Value slotVal = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(slot));

      Value freeFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase + cb.multiBufferDepth));
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

  for (auto &cb : buffers) {
    for (size_t slot = 0; slot < cb.multiBufferDepth; ++slot) {
      Value slotVal = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(slot));

      Value freeFlagBase = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(cb.flagBase + cb.multiBufferDepth));
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
