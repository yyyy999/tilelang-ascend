// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/EnableSoftPipeline.cpp
 * \brief TileLangIR Enable Soft Pipeline pass.
 *
 * Transforms pipelined loops into soft-pipeline mode where Cube and Vector
 * operations from different stages can overlap execution.
 *
 * Instead of creating separate inner loops for each scope (unroll mode),
 * this pass creates a single inner loop with scf.if dispatch, enabling
 * cross-stage CV overlap through embedded synchronization.
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "tilelangir/Transforms/Passes.h.inc"

#define DEBUG_TYPE "tilelangir-enable-soft-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")

using namespace mlir;
using namespace mlir::tilelangir;
using namespace bishengir;

namespace mlir {
namespace tilelangir {

#define GEN_PASS_DEF_TILELANGIRENABLESOFTPIPELINE
#include "tilelangir/Transforms/Passes.h.inc"

static constexpr size_t SYNC_FLAGS_LIMIT = 16;

static MemRefType expandMemRefType(MemRefType oldType, int32_t multiBuffer) {
  ArrayRef<int64_t> oldShape = oldType.getShape();
  SmallVector<int64_t> newShape;
  newShape.push_back(multiBuffer);
  newShape.append(oldShape.begin(), oldShape.end());

  MemRefLayoutAttrInterface newLayout = oldType.getLayout();
  if (auto stridedLayout = dyn_cast<StridedLayoutAttr>(oldType.getLayout())) {
    SmallVector<int64_t> newStrides;
    int64_t leadingStride = 1;
    bool isDynamic = false;
    for (int64_t dim : oldShape) {
      if (dim == ShapedType::kDynamic) {
        isDynamic = true;
        break;
      }
      leadingStride *= dim;
    }
    if (!isDynamic) {
      newStrides.push_back(leadingStride);
    } else {
      newStrides.push_back(ShapedType::kDynamic);
    }
    ArrayRef<int64_t> oldStrides = stridedLayout.getStrides();
    newStrides.append(oldStrides.begin(), oldStrides.end());
    newLayout = StridedLayoutAttr::get(oldType.getContext(),
                                       stridedLayout.getOffset(), newStrides);
  }
  return MemRefType::get(newShape, oldType.getElementType(), newLayout,
                         oldType.getMemorySpace());
}

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

static hivm::TCoreType anotherCoreType(const hivm::TCoreType &current) {
  return current == hivm::TCoreType::VECTOR ? hivm::TCoreType::CUBE
                                            : hivm::TCoreType::VECTOR;
}

static void buildCVSyncSet(OpBuilder &builder, Location loc,
                           hivm::TCoreType coreSrc, Value flagId) {
  auto coreSrcAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), coreSrc);
  auto sync_mode = hivm::SyncBlockInstrModeAttr::get(
      builder.getContext(),
      hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr = hivm::PipeAttr::get(builder.getContext(),
                                         coreSrc == hivm::TCoreType::VECTOR
                                             ? hivm::PIPE::PIPE_MTE3
                                             : hivm::PIPE::PIPE_FIX);

  builder.create<hivm::SyncBlockSetOp>(loc, coreSrcAttr, pipTypeAttr,
                                       tPipTypeAttr, flagId, mlir::Value(),
                                       sync_mode);
}

static void buildCVSyncWait(OpBuilder &builder, Location loc,
                            hivm::TCoreType coreSrc, Value flagId) {
  auto coreSrcAttr =
      mlir::hivm::TCoreTypeAttr::get(builder.getContext(), coreSrc);
  auto tPipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto pipTypeAttr =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_MTE2);

  builder.create<hivm::SyncBlockWaitOp>(loc, coreSrcAttr, tPipTypeAttr,
                                        pipTypeAttr, flagId);
}

class WorkspaceExpander {
public:
  static bool expand(Operation *op) {
    bool changed = false;
    SmallVector<memref_ext::AllocWorkspaceOp> targets;

    op->walk([&](memref_ext::AllocWorkspaceOp allocOp) {
      Value res = allocOp.getResult();
      for (Operation *user : res.getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
          if (markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
            targets.push_back(allocOp);
            break;
          }
        }
      }
    });

    for (auto allocOp : targets) {
      Value workspaceValue = allocOp.getResult();
      int32_t multiBuffer = 0;
      annotation::MarkOp markOpToRemove = nullptr;

      for (Operation *user : workspaceValue.getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
          if (auto attr =
                  markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
            multiBuffer = static_cast<int32_t>(attr.getInt());
            markOpToRemove = markOp;
            break;
          }
        }
      }

      if (!markOpToRemove)
        continue;

      LLVM_DEBUG(DBGS() << "Expanding workspace with factor=" << multiBuffer
                        << "\n");
      markOpToRemove->erase();

      MemRefType oldType = workspaceValue.getType().cast<MemRefType>();
      MemRefType newType = expandMemRefType(oldType, multiBuffer);

      OpBuilder builder(allocOp);
      auto newAlloc = builder.create<memref_ext::AllocWorkspaceOp>(
          allocOp.getLoc(), newType, allocOp.getWorkspaceArg(),
          allocOp.getDynamicSize(), allocOp.getOffset());

      workspaceValue.replaceAllUsesWith(newAlloc.getResult());
      allocOp.erase();
      changed = true;
    }
    return changed;
  }
};

class SoftPipelineConverter {
public:
  SoftPipelineConverter(scf::ForOp outerFor,
                        ArrayRef<scope::ScopeOp> scopeOps,
                        ArrayRef<Value> workspaceValues, int32_t numStages)
      : outerFor_(outerFor), scopeOps_(scopeOps),
        workspaceValues_(workspaceValues), numStages_(numStages) {}

  bool convert() {
    if (scopeOps_.empty())
      return false;

    OpBuilder builder(outerFor_);
    Location loc = outerFor_.getLoc();
    Value outerIV = outerFor_.getInductionVar();

    Value c0 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                                 builder.getI32IntegerAttr(0));
    Value cNumStages = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(numStages_));
    Value c1 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                                 builder.getI32IntegerAttr(1));

    auto innerFor = builder.create<scf::ForOp>(loc, c0, cNumStages, c1,
                                               ValueRange{});
    innerFor->setAttr("hivm.soft_pipeline", builder.getUnitAttr());
    innerFor->setAttr("tilelangir.num_stages",
                      builder.getI32IntegerAttr(numStages_));

    Block *innerBody = innerFor.getBody();
    Operation *terminator = innerBody->getTerminator();
    Value innerIV = innerFor.getInductionVar();

    builder.setInsertionPoint(innerBody, innerBody->begin());

    Value outerIVi64 = convertToI64(builder, loc, outerIV);
    Value innerIVi64 = convertToI64(builder, loc, innerIV);
    Value numStagesI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages_));
    Value syncLimitI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));

    Value flagBase = builder.create<arith::MulIOp>(loc, outerIVi64, numStagesI64);
    Value flagIdI64 = builder.create<arith::AddIOp>(loc, flagBase, innerIVi64);
    Value flagIdNextI64 =
        builder.create<arith::AddIOp>(loc, flagIdI64, numStagesI64);
    Value flagIdMod =
        builder.create<arith::RemSIOp>(loc, flagIdI64, syncLimitI64);
    Value flagIdNextMod =
        builder.create<arith::RemSIOp>(loc, flagIdNextI64, syncLimitI64);

    Value stageIdx =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), innerIV);

    newOuterExprI32_ = builder.create<arith::AddIOp>(
        loc,
        builder.create<arith::MulIOp>(
            loc, outerIV,
            builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                              builder.getI32IntegerAttr(numStages_))),
        innerIV);
    Value newOuterExprIdx = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), newOuterExprI32_);

    innerFor_ = innerFor;

    for (size_t stageIdx_ = 0; stageIdx_ < scopeOps_.size(); ++stageIdx_) {
      auto scopeOp = scopeOps_[stageIdx_];
      auto coreTypeAttr =
          scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
      if (!coreTypeAttr) {
        scopeOp->emitWarning("ScopeOp without tcore_type, skipping");
        continue;
      }
      hivm::TCoreType coreType = coreTypeAttr.getTcoretype();

      builder.setInsertionPoint(terminator);

      Value cmpVal = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, innerIV,
          builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                            builder.getI32IntegerAttr(stageIdx_)));

      auto ifOp = builder.create<scf::IfOp>(loc, cmpVal, false);
      Block *thenBlock = &ifOp.getThenRegion().front();
      builder.setInsertionPointToStart(thenBlock);

      hivm::TCoreType waitCoreType = anotherCoreType(coreType);
      Value waitFlagId = flagIdMod;
      Value setFlagId = flagIdNextMod;

      if (stageIdx_ == 0) {
        waitFlagId = flagIdI64;
        setFlagId = flagIdI64;
      }

      buildCVSyncWait(builder, loc, waitCoreType, waitFlagId);

      Region *scopeRegion = &scopeOp.getRegion();
      if (!scopeRegion->empty()) {
        Block *scopeBody = &scopeRegion->front();
        SmallVector<Operation *> opsToMove;
        for (Operation &op : scopeBody->getOperations()) {
          if (!isa<scope::ReturnOp>(op)) {
            opsToMove.push_back(&op);
          }
        }
        for (Operation *op : opsToMove) {
          op->moveBefore(thenBlock->getTerminator());
        }
      }

      adjustOperationsInBranch(thenBlock, newOuterExprI32_, newOuterExprIdx,
                               stageIdx_, builder);

      builder.setInsertionPoint(thenBlock->getTerminator());
      buildCVSyncSet(builder, loc, coreType, setFlagId);
    }

    for (auto scopeOp : scopeOps_) {
      scopeOp->erase();
    }

    return true;
  }

private:
  void adjustOperationsInBranch(Block *thenBlock, Value newOuterExprI32,
                                Value newOuterExprIdx, size_t stageIdx,
                                OpBuilder &builder) {
    Value innerIV = innerFor_.getInductionVar();
    Value outerIV = outerFor_.getInductionVar();

    for (Operation &op : llvm::make_early_inc_range(*thenBlock)) {
      if (auto subview = dyn_cast<memref::SubViewOp>(&op)) {
        bool isWorkspace = false;
        for (Value ws : workspaceValues_) {
          if (subview.getSource() == ws) {
            isWorkspace = true;
            break;
          }
        }
        if (isWorkspace) {
          adjustWorkspaceSubview(subview, builder);
        } else {
          adjustGlobalSubviewOffset(subview, newOuterExprI32, builder);
        }
      } else if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        for (Value ws : workspaceValues_) {
          if (copyOp.getSource() == ws || copyOp.getTarget() == ws) {
            adjustCopyOp(copyOp, builder);
            break;
          }
        }
      }

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        if (op.getOperand(i) == outerIV) {
          op.setOperand(i, newOuterExprIdx);
        }
      }
    }
  }

  void adjustWorkspaceSubview(memref::SubViewOp subview, OpBuilder &builder) {
    Location loc = subview.getLoc();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    Value innerIV = innerFor_.getInductionVar();

    Value stageIndex =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), innerIV);

    auto origOffsets = subview.getMixedOffsets();
    auto origSizes = subview.getMixedSizes();
    auto origStrides = subview.getMixedStrides();

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    newOffsets.push_back(stageIndex);
    newSizes.push_back(builder.getIndexAttr(1));
    newStrides.push_back(builder.getIndexAttr(1));

    for (size_t i = 0; i < origOffsets.size(); ++i) {
      auto ofr = origOffsets[i];
      if (auto val = ofr.dyn_cast<Value>()) {
        if (val == outerFor_.getInductionVar()) {
          newOffsets.push_back(
              builder.create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                 newOuterExprI32_).getResult());
        } else {
          newOffsets.push_back(val);
        }
      } else {
        newOffsets.push_back(ofr);
      }
      newSizes.push_back(origSizes[i]);
      newStrides.push_back(origStrides[i]);
    }

    auto newSubview = builder.create<memref::SubViewOp>(loc, source, newOffsets,
                                                        newSizes, newStrides);
    auto subviewType = newSubview.getResult().getType().cast<MemRefType>();

    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices currentGroup;
    bool mergedLeadingOnes = false;
    for (int i = 0; i < subviewType.getRank(); ++i) {
      int64_t dimSize = subviewType.getDimSize(i);
      if (!mergedLeadingOnes && dimSize == 1) {
        currentGroup.push_back(i);
      } else {
        if (!currentGroup.empty()) {
          currentGroup.push_back(i);
          reassociation.push_back(currentGroup);
          currentGroup.clear();
          mergedLeadingOnes = true;
        } else {
          reassociation.push_back({i});
        }
      }
    }
    if (!currentGroup.empty())
      reassociation.push_back(currentGroup);

    Value collapsed = builder.create<memref::CollapseShapeOp>(
        loc, newSubview.getResult(), reassociation);
    subview.getResult().replaceAllUsesWith(collapsed);
    subview.erase();
  }

  void adjustGlobalSubviewOffset(memref::SubViewOp subview, Value newOuterI32,
                                 OpBuilder &builder) {
    auto origOffsets = subview.getMixedOffsets();
    SmallVector<OpFoldResult> newOffsets;

    for (size_t i = 0; i < origOffsets.size(); ++i) {
      auto ofr = origOffsets[i];
      if (auto val = ofr.dyn_cast<Value>()) {
        if (val == outerFor_.getInductionVar()) {
          Value newIdx = builder.create<arith::IndexCastOp>(
              subview.getLoc(), builder.getIndexType(), newOuterI32);
          newOffsets.push_back(newIdx);
        } else {
          newOffsets.push_back(val);
        }
      } else {
        newOffsets.push_back(ofr);
      }
    }

    auto newSubview = builder.create<memref::SubViewOp>(
        subview.getLoc(), subview.getSource(), newOffsets,
        subview.getMixedSizes(), subview.getMixedStrides());
    subview.getResult().replaceAllUsesWith(newSubview.getResult());
    subview.erase();
  }

  void adjustCopyOp(memref::CopyOp copyOp, OpBuilder &builder) {
    Location loc = copyOp.getLoc();
    Value innerIV = innerFor_.getInductionVar();
    Value stageIndex =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), innerIV);

    Value ws = nullptr;
    bool fixSource = false;
    for (Value candidate : workspaceValues_) {
      if (copyOp.getSource() == candidate) {
        ws = candidate;
        fixSource = true;
        break;
      }
      if (copyOp.getTarget() == candidate) {
        ws = candidate;
        fixSource = false;
        break;
      }
    }
    if (!ws)
      return;

    auto wsType = ws.getType().cast<MemRefType>();

    SmallVector<OpFoldResult> offsets, sizes, strides;
    offsets.push_back(stageIndex);
    sizes.push_back(builder.getIndexAttr(1));
    strides.push_back(builder.getIndexAttr(1));

    for (int i = 1; i < wsType.getRank(); ++i) {
      offsets.push_back(builder.getIndexAttr(0));
      if (wsType.isDynamicDim(i)) {
        sizes.push_back(builder.createOrFold<memref::DimOp>(loc, ws, i));
      } else {
        sizes.push_back(builder.getIndexAttr(wsType.getDimSize(i)));
      }
      strides.push_back(builder.getIndexAttr(1));
    }

    auto subviewOp = builder.create<memref::SubViewOp>(loc, ws, offsets, sizes,
                                                       strides);
    auto subviewType = subviewOp.getResult().getType().cast<MemRefType>();

    SmallVector<ReassociationIndices> reassociation;
    ReassociationIndices currentGroup;
    bool mergedLeadingOnes = false;
    for (int i = 0; i < subviewType.getRank(); ++i) {
      int64_t dimSize = subviewType.getDimSize(i);
      if (!mergedLeadingOnes && dimSize == 1) {
        currentGroup.push_back(i);
      } else {
        if (!currentGroup.empty()) {
          currentGroup.push_back(i);
          reassociation.push_back(currentGroup);
          currentGroup.clear();
          mergedLeadingOnes = true;
        } else {
          reassociation.push_back({i});
        }
      }
    }
    if (!currentGroup.empty())
      reassociation.push_back(currentGroup);

    Value slicedWs = builder.create<memref::CollapseShapeOp>(
        loc, subviewOp.getResult(), reassociation);

    if (fixSource) {
      builder.create<memref::CopyOp>(loc, slicedWs, copyOp.getTarget());
    } else {
      builder.create<memref::CopyOp>(loc, copyOp.getSource(), slicedWs);
    }
    copyOp.erase();
  }

  scf::ForOp outerFor_;
  scf::ForOp innerFor_;
  SmallVector<scope::ScopeOp> scopeOps_;
  SmallVector<Value> workspaceValues_;
  int32_t numStages_;
  Value newOuterExprI32_;
};

class SoftPipelineProcessor {
public:
  SoftPipelineProcessor(scf::ForOp pipelineLoop,
                        ArrayRef<Value> workspaceValues)
      : pipelineLoop_(pipelineLoop), workspaceValues_(workspaceValues) {}

  bool process() {
    auto attr =
        pipelineLoop_->getAttrOfType<IntegerAttr>("tilelangir.num_stages");
    if (!attr)
      return false;

    int32_t numStages = static_cast<int32_t>(attr.getInt());
    LLVM_DEBUG(DBGS() << "Processing pipeline loop with num_stages="
                      << numStages << "\n");

    SmallVector<scope::ScopeOp> scopeOps;
    for (Operation &op : pipelineLoop_.getBody()->getOperations()) {
      if (auto scopeOp = dyn_cast<scope::ScopeOp>(&op)) {
        scopeOps.push_back(scopeOp);
      }
    }

    if (scopeOps.empty()) {
      LLVM_DEBUG(DBGS() << "No scope ops found, skipping\n");
      return false;
    }

    hivm::TCoreType beginCoreType = hivm::TCoreType::CUBE_OR_VECTOR;
    hivm::TCoreType endCoreType = hivm::TCoreType::CUBE_OR_VECTOR;
    auto firstCoreTypeAttr =
        scopeOps.front()->getAttrOfType<hivm::TCoreTypeAttr>(
            hivm::TCoreTypeAttr::name);
    if (firstCoreTypeAttr)
      beginCoreType = firstCoreTypeAttr.getTcoretype();
    auto lastCoreTypeAttr =
        scopeOps.back()->getAttrOfType<hivm::TCoreTypeAttr>(
            hivm::TCoreTypeAttr::name);
    if (lastCoreTypeAttr)
      endCoreType = lastCoreTypeAttr.getTcoretype();

    Value oldUpper = pipelineLoop_.getUpperBound();
    OpBuilder builder(pipelineLoop_);
    Location loc = pipelineLoop_.getLoc();
    Value constNumStages = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(numStages));
    Value newUpper =
        builder.create<arith::DivSIOp>(loc, oldUpper, constNumStages);
    pipelineLoop_.setUpperBound(newUpper);

    SoftPipelineConverter converter(pipelineLoop_, scopeOps, workspaceValues_,
                                    numStages);
    bool changed = converter.convert();

    if (changed) {
      insertInitAndClear(pipelineLoop_, numStages, beginCoreType, endCoreType);
    }

    return changed;
  }

private:
  void insertInitAndClear(scf::ForOp outerFor, int32_t numStages,
                          hivm::TCoreType beginCoreType,
                          hivm::TCoreType endCoreType) {
    if (beginCoreType == hivm::TCoreType::CUBE_OR_VECTOR)
      return;

    hivm::TCoreType initCoreType = anotherCoreType(beginCoreType);
    hivm::TCoreType clearCoreType = anotherCoreType(endCoreType);

    Block *parentBlock = outerFor->getBlock();
    OpBuilder builder(outerFor->getContext());

    Value c0 = builder.create<arith::ConstantOp>(
        outerFor.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0));
    Value cNumStages = builder.create<arith::ConstantOp>(
        outerFor.getLoc(), builder.getI32Type(),
        builder.getI32IntegerAttr(numStages));
    Value c1 = builder.create<arith::ConstantOp>(
        outerFor.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(1));

    builder.setInsertionPoint(outerFor);
    auto initForOp =
        builder.create<scf::ForOp>(outerFor.getLoc(), c0, cNumStages, c1);
    auto initCoreTypeAttr =
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), initCoreType);
    initForOp->setAttr(hivm::TCoreTypeAttr::name, initCoreTypeAttr);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *initBody = initForOp.getBody();
      builder.setInsertionPointToStart(initBody);
      Value initId = initForOp.getInductionVar();
      initId = convertToI64(builder, initForOp->getLoc(), initId);
      buildCVSyncSet(builder, initForOp->getLoc(), initCoreType, initId);
    }

    builder.setInsertionPointAfter(outerFor);
    auto clearForOp =
        builder.create<scf::ForOp>(outerFor.getLoc(), c0, cNumStages, c1);
    auto clearCoreTypeAttr =
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), clearCoreType);
    clearForOp->setAttr(hivm::TCoreTypeAttr::name, clearCoreTypeAttr);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *clearBody = clearForOp.getBody();
      builder.setInsertionPointToStart(clearBody);
      Value clearId = clearForOp.getInductionVar();
      clearId = convertToI64(builder, clearForOp->getLoc(), clearId);
      buildCVSyncWait(builder, clearForOp->getLoc(), clearCoreType, clearId);
    }
  }

  scf::ForOp pipelineLoop_;
  SmallVector<Value> workspaceValues_;
};

namespace {
struct TileLangIREnableSoftPipeline
    : public impl::TileLangIREnableSoftPipelineBase<
          TileLangIREnableSoftPipeline> {
  using Base =
      impl::TileLangIREnableSoftPipelineBase<TileLangIREnableSoftPipeline>;
  using Base::Base;

public:
  void runOnOperation() override;
};
} // end anonymous namespace

void TileLangIREnableSoftPipeline::runOnOperation() {
  ModuleOp module = getOperation();
  if (!module)
    return;

  LLVM_DEBUG(DBGS() << "Starting EnableSoftPipeline pass\n");

  if (!WorkspaceExpander::expand(module)) {
    LLVM_DEBUG(DBGS() << "No workspaces to expand.\n");
    return;
  }

  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    SmallVector<Value> expandedWorkspaces;
    func.walk([&](memref_ext::AllocWorkspaceOp allocOp) {
      auto type = allocOp.getType().cast<MemRefType>();
      if (type.getRank() > 1 && type.getShape()[0] > 1) {
        expandedWorkspaces.push_back(allocOp.getResult());
      }
    });

    if (expandedWorkspaces.empty())
      continue;

    LLVM_DEBUG(DBGS() << "Found " << expandedWorkspaces.size()
                      << " expanded workspaces in function "
                      << func.getSymName() << "\n");

    SmallVector<scf::ForOp> pipelineLoops;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->getAttr("tilelangir.num_stages")) {
        bool usesAnyWs = false;
        forOp.walk([&](Operation *op) {
          if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
            for (Value ws : expandedWorkspaces) {
              if (sv.getSource() == ws) {
                usesAnyWs = true;
                return;
              }
            }
          } else if (auto cp = dyn_cast<memref::CopyOp>(op)) {
            for (Value ws : expandedWorkspaces) {
              if (cp.getSource() == ws || cp.getTarget() == ws) {
                usesAnyWs = true;
                return;
              }
            }
          }
        });
        if (usesAnyWs) {
          pipelineLoops.push_back(forOp);
        }
      }
    });

    for (auto forOp : pipelineLoops) {
      SoftPipelineProcessor processor(forOp, expandedWorkspaces);
      processor.process();
    }
  }
}

} // namespace tilelangir
} // namespace mlir
