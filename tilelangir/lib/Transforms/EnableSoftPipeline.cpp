// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file tilelangir/lib/Transforms/EnableSoftPipeline.cpp
 * \brief TileLangIR Enable Soft Pipeline pass.
 *
 * Transforms pipelined loops into soft-pipeline mode where Cube and Vector
 * operations from different stages can overlap execution.
 *
 * When preconditions for two-branch software pipelining are met
 * (num_stages==2, even scope count, flag budget), the pass replaces the
 * inner stage loop with two parallel `scf.if` regions: "front" stages for
 * tile k and "back" stages for tile (k-1) in the same outer iteration, with
 * workspace slot = tile % S.  Per-workspace CV sync is inserted around
 * `memref.copy` to/from the expanded workspace.  Otherwise, the pass falls
 * back to a single inner `scf.for` over stages and a linear timestamp flag
 * chain (t = outerIV * S + innerIV) as in the original implementation.
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
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>

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

static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = constOp.getValue().dyn_cast_or_null<IntegerAttr>())
      return intAttr.getInt();
  }
  return std::nullopt;
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

/// sync_block_set: [core, src_pipe, PIPE_S, flag, ...]
static void
buildHivmSyncSet(OpBuilder &builder, Location loc, hivm::TCoreType core,
                 hivm::PIPE srcPipe, Value flagId) {
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), core);
  auto syncMode = hivm::SyncBlockInstrModeAttr::get(
      builder.getContext(),
      hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
  auto pipeS =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto srcPipeAttr = hivm::PipeAttr::get(builder.getContext(), srcPipe);

  builder.create<hivm::SyncBlockSetOp>(loc, coreAttr, srcPipeAttr, pipeS,
                                       flagId, Value(), syncMode);
}

/// sync_block_wait: [core, PIPE_S, dst_pipe, flag]
static void
buildHivmSyncWait(OpBuilder &builder, Location loc, hivm::TCoreType core,
                 hivm::PIPE dstPipe, Value flagId) {
  auto coreAttr = hivm::TCoreTypeAttr::get(builder.getContext(), core);
  auto pipeS =
      hivm::PipeAttr::get(builder.getContext(), hivm::PIPE::PIPE_S);
  auto dstPipeAttr = hivm::PipeAttr::get(builder.getContext(), dstPipe);

  builder.create<hivm::SyncBlockWaitOp>(loc, coreAttr, pipeS, dstPipeAttr,
                                        flagId);
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
  buildHivmSyncWait(builder, loc, coreSrc, hivm::PIPE::PIPE_MTE2, flagId);
}

static std::pair<Value, Value> buildLinearFlagsFromT(OpBuilder &builder,
                                                     Location loc,
                                                     Value tI64) {
  Value syncLimitI64 = builder.create<arith::ConstantOp>(
      loc, builder.getI64Type(),
      builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
  Value oneI64 = builder.create<arith::ConstantOp>(
      loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
  Value flagT = builder.create<arith::RemSIOp>(loc, tI64, syncLimitI64);
  Value tMinus1PlusL = builder.create<arith::AddIOp>(
      loc, builder.create<arith::SubIOp>(loc, tI64, oneI64), syncLimitI64);
  Value flagTm1 = builder.create<arith::RemSIOp>(loc, tMinus1PlusL, syncLimitI64);
  return {flagTm1, flagT};
}

/// First expanded workspace with static leading dim == 2 (S=2, B=1 two-branch).
static std::optional<int32_t> getTwoBranchWorkspaceSlotS(
    ArrayRef<Value> workspaceValues) {
  for (Value ws : workspaceValues) {
    auto ty = ws.getType().dyn_cast<MemRefType>();
    if (!ty || ty.getRank() < 1)
      continue;
    if (ty.isDynamicDim(0))
      return std::nullopt;
    int64_t d0 = ty.getDimSize(0);
    if (d0 == 2)
      return 2;
  }
  return std::nullopt;
}

// ---------------------------------------------------------------------------
// WorkspaceExpander
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// SoftPipelineConverter
// ---------------------------------------------------------------------------
class SoftPipelineConverter {
public:
  SoftPipelineConverter(scf::ForOp outerFor,
                        ArrayRef<scope::ScopeOp> scopeOps,
                        ArrayRef<Value> workspaceValues, int32_t numStages)
      : outerFor_(outerFor), scopeOps_(scopeOps),
        workspaceValues_(workspaceValues), numStages_(numStages) {
    for (Value ws : workspaceValues_)
      workspaceValueSet_.insert(ws);
  }

  bool convert() {
    if (scopeOps_.empty())
      return false;
    if (canUseTwoBranch())
      return convertTwoBranch();
    return convertWithInnerLoop();
  }

private:
  bool canUseTwoBranch() const {
    if (scopeOps_.size() < 2 || (scopeOps_.size() & 1))
      return false;
    if (static_cast<int32_t>(scopeOps_.size()) != numStages_)
      return false;
    if (!getTwoBranchWorkspaceSlotS(workspaceValues_).has_value())
      return false;
    return true;
  }

  static void moveScopeBodyBeforeTerminator(scope::ScopeOp scope,
                                            Block *dest) {
    if (scope.getRegion().empty())
      return;
    Block *scopeBody = &scope.getRegion().front();
    SmallVector<Operation *> ops;
    for (Operation &op : scopeBody->getOperations()) {
      if (!isa<scope::ReturnOp>(op))
        ops.push_back(&op);
    }
    for (Operation *op : ops)
      op->moveBefore(dest->getTerminator());
  }

  static void
  replaceInductionInBlockExcludingDef(Block *block, Value from, Value to,
                                      Operation *excludedUser) {
    if (!from || !to)
      return;
    from.replaceUsesWithIf(to, [&](OpOperand &u) {
      Operation *user = u.getOwner();
      if (user->getBlock() != block)
        return false;
      if (user == excludedUser)
        return false;
      return u.get() == from;
    });
  }

  static Value createIntLikeConstant(OpBuilder &builder, Location loc,
                                        Type t, int64_t v) {
    if (t.isIndex())
      return builder.create<arith::ConstantIndexOp>(loc, v);
    return builder.create<arith::ConstantOp>(loc, t,
                                            builder.getIntegerAttr(t, v));
  }

  /// Software pipeline: two parallel scf.if (tile k "front" / tile k-1 "back"),
  /// no inner stage loop.  Upper bound = N+1, slot = tile % S.  Requires
  /// static 2-wide workspace, scope count == num_stages_.
  bool convertTwoBranch() {
    int32_t half = static_cast<int32_t>(scopeOps_.size() / 2);

    OpBuilder builder(outerFor_);
    Location loc = outerFor_.getLoc();
    Value outerIV = outerFor_.getInductionVar();
    auto ivType = outerIV.getType();
    if (!ivType.isIntOrIndex())
      return false;
    std::optional<int32_t> slotSOpt = getTwoBranchWorkspaceSlotS(workspaceValues_);
    if (!slotSOpt)
      return false; // canUseTwoBranch and workspace shape diverged
    int32_t slotS = *slotSOpt;

    Value originalUB = outerFor_.getUpperBound();
    Value c1 = createIntLikeConstant(builder, loc, ivType, 1);
    Value c0 = createIntLikeConstant(builder, loc, ivType, 0);
    Value cSlot = createIntLikeConstant(builder, loc, ivType, slotS);

    Value newUB = builder.create<arith::AddIOp>(loc, originalUB, c1);
    outerFor_.setUpperBound(newUB);
    outerFor_->setAttr("hivm.soft_pipeline", builder.getUnitAttr());

    if (scopeOps_.empty() || scopeOps_.front()->getBlock() != outerFor_.getBody())
      builder.setInsertionPointToStart(outerFor_.getBody());
    else
      builder.setInsertionPoint(scopeOps_.front());

    Value numStagesI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages_));

    // Branch 1: if k < N
    Value cond1 = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, outerIV, originalUB);
    auto ifB1 = builder.create<scf::IfOp>(loc, cond1, false);
    Block *then1 = &ifB1.getThenRegion().front();
    builder.setInsertionPointToStart(then1);
    Value slotI32B1 = builder.create<arith::RemSIOp>(loc, outerIV, cSlot);
    Value slotIdxB1 = slotI32B1.getType().isIndex()
                        ? slotI32B1
                        : builder.create<arith::IndexCastOp>(
                              loc, builder.getIndexType(), slotI32B1);

    for (int s = 0; s < half; ++s) {
      scope::ScopeOp scopeOp = scopeOps_[s];
      auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name);
      if (!coreTypeAttr) {
        scopeOp->emitWarning("ScopeOp without tcore_type, skipping");
        continue;
      }
      hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
      hivm::TCoreType waitCoreType = anotherCoreType(coreType);

      Value outerI64 = convertToI64(builder, loc, outerIV);
      Value stageI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(s));
      Value tI64 = builder.create<arith::AddIOp>(
          loc, builder.create<arith::MulIOp>(loc, outerI64, numStagesI64), stageI64);
      auto [flagTm1, flagT] = buildLinearFlagsFromT(builder, loc, tI64);

      builder.setInsertionPoint(then1->getTerminator());
      buildCVSyncWait(builder, loc, waitCoreType, flagTm1);
      moveScopeBodyBeforeTerminator(scopeOp, then1);
      builder.setInsertionPoint(then1->getTerminator());
      buildCVSyncSet(builder, loc, coreType, flagT);
    }
    adjustOperationsInBranch(then1, builder, slotIdxB1);

    // Branch 2: if k > 0
    if (scopeOps_.size() > size_t(half) &&
        scopeOps_[half]->getBlock() == outerFor_.getBody())
      builder.setInsertionPoint(scopeOps_[half]);
    else
      builder.setInsertionPointAfter(ifB1);

    Value cond2 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, outerIV, c0);
    auto ifB2 = builder.create<scf::IfOp>(loc, cond2, false);
    Block *then2 = &ifB2.getThenRegion().front();
    builder.setInsertionPointToStart(then2);
    Value kMinus1 = builder.create<arith::SubIOp>(loc, outerIV, c1).getResult();
    Value slotI32B2 = builder.create<arith::RemSIOp>(loc, kMinus1, cSlot);
    Value slotIdxB2 = slotI32B2.getType().isIndex()
                        ? slotI32B2
                        : builder.create<arith::IndexCastOp>(
                              loc, builder.getIndexType(), slotI32B2);

    for (int s = 0; s < half; ++s) {
      scope::ScopeOp scopeOp = scopeOps_[static_cast<unsigned>(half + s)];
      auto coreTypeAttr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(
          hivm::TCoreTypeAttr::name);
      if (!coreTypeAttr) {
        scopeOp->emitWarning("ScopeOp without tcore_type, skipping");
        continue;
      }
      hivm::TCoreType coreType = coreTypeAttr.getTcoretype();
      hivm::TCoreType waitCoreType = anotherCoreType(coreType);

      Value kMinus1I64 = convertToI64(builder, loc, kMinus1);
      Value stageI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(s + half));
      Value tI64 = builder.create<arith::AddIOp>(
          loc, builder.create<arith::MulIOp>(loc, kMinus1I64, numStagesI64), stageI64);
      auto [flagTm1, flagT] = buildLinearFlagsFromT(builder, loc, tI64);

      builder.setInsertionPoint(then2->getTerminator());
      buildCVSyncWait(builder, loc, waitCoreType, flagTm1);
      moveScopeBodyBeforeTerminator(scopeOp, then2);
      builder.setInsertionPoint(then2->getTerminator());
      buildCVSyncSet(builder, loc, coreType, flagT);
    }
    replaceInductionInBlockExcludingDef(then2, outerIV, kMinus1,
                                        kMinus1.getDefiningOp());
    adjustOperationsInBranch(then2, builder, slotIdxB2);

    for (auto scopeOp : scopeOps_) {
      scopeOp->erase();
    }

    innerFor_ = scf::ForOp();
    return true;
  }

  /// Original shape: inner `scf.for` over stages and linear t = outer*S+inner.
  bool convertWithInnerLoop() {
    OpBuilder builder(outerFor_);
    Location loc = outerFor_.getLoc();
    Value outerIV = outerFor_.getInductionVar();

    if (!scopeOps_.empty() &&
        scopeOps_.front()->getBlock() == outerFor_.getBody()) {
      builder.setInsertionPoint(scopeOps_.front());
    } else {
      builder.setInsertionPointToStart(outerFor_.getBody());
    }

    auto ivType = outerIV.getType();
    if (!ivType.isIntOrIndex())
      return false;

    Value c0 = createIntLikeConstant(builder, loc, ivType, 0);
    Value cNumStages = createIntLikeConstant(builder, loc, ivType, numStages_);
    Value c1 = createIntLikeConstant(builder, loc, ivType, 1);

    auto innerFor = builder.create<scf::ForOp>(loc, c0, cNumStages, c1, ValueRange{});
    innerFor->setAttr("hivm.soft_pipeline", builder.getUnitAttr());

    Block *innerBody = innerFor.getBody();
    Operation *terminator = innerBody->getTerminator();
    Value innerIV = innerFor.getInductionVar();

    builder.setInsertionPoint(innerBody, innerBody->begin());

    slotI32_ = builder.create<arith::RemSIOp>(loc, outerIV, cNumStages);
    slotIdx_ = builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), slotI32_);

    Value outerI64 = convertToI64(builder, loc, outerIV);
    Value innerI64 = convertToI64(builder, loc, innerIV);
    Value numStagesI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages_));
    Value syncLimitI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(),
        builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
    Value oneI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
    Value tI64 = builder.create<arith::AddIOp>(
        loc, builder.create<arith::MulIOp>(loc, outerI64, numStagesI64), innerI64);
    Value flagT = builder.create<arith::RemSIOp>(loc, tI64, syncLimitI64);
    Value tMinus1PlusL = builder.create<arith::AddIOp>(
        loc, builder.create<arith::SubIOp>(loc, tI64, oneI64), syncLimitI64);
    Value flagTm1 = builder.create<arith::RemSIOp>(loc, tMinus1PlusL, syncLimitI64);

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

      Value stageConst = createIntLikeConstant(builder, loc, innerIV.getType(),
                                                static_cast<int64_t>(stageIdx_));
      Value cmpVal = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 innerIV, stageConst);

      auto ifOp = builder.create<scf::IfOp>(loc, cmpVal, false);
      Block *thenBlock = &ifOp.getThenRegion().front();
      builder.setInsertionPointToStart(thenBlock);

      hivm::TCoreType waitCoreType = anotherCoreType(coreType);
      buildCVSyncWait(builder, loc, waitCoreType, flagTm1);

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

      adjustOperationsInBranch(thenBlock, builder, slotIdx_);

      builder.setInsertionPoint(thenBlock->getTerminator());
      buildCVSyncSet(builder, loc, coreType, flagT);
    }

    for (auto scopeOp : scopeOps_) {
      scopeOp->erase();
    }

    return true;
  }

  static Value getViewLikeRoot(Value v) {
    while (Operation *def = v.getDefiningOp()) {
      if (auto sv = dyn_cast<memref::SubViewOp>(def)) {
        v = sv.getSource();
        continue;
      }
      if (auto view = dyn_cast<memref::ViewOp>(def)) {
        v = view.getSource();
        continue;
      }
      if (auto rc = dyn_cast<memref::ReinterpretCastOp>(def)) {
        v = rc.getSource();
        continue;
      }
      break;
    }
    return v;
  }

  bool isWorkspaceValue(Value v) const {
    return workspaceValueSet_.contains(getViewLikeRoot(v));
  }

  /// Only adjust workspace subview / copy ops (add buffer-slot dimension).
  /// Global-memory subviews stay unchanged: outerIV IS the original iteration.
  void adjustOperationsInBranch(Block *thenBlock, OpBuilder &builder,
                                 Value slotIdx) {
    SmallVector<Operation *> opsSnapshot;
    opsSnapshot.reserve(thenBlock->getOperations().size());
    for (Operation &op : thenBlock->getOperations())
      opsSnapshot.push_back(&op);

    for (Operation *op : opsSnapshot) {
      if (!op || op->getBlock() != thenBlock)
        continue;
      if (op->hasTrait<OpTrait::IsTerminator>())
        continue;

      if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
        if (isWorkspaceValue(subview.getSource())) {
          builder.setInsertionPoint(subview);
          adjustWorkspaceSubview(subview, builder, slotIdx);
        }
        continue;
      }

      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        if (isWorkspaceValue(copyOp.getSource()) ||
            isWorkspaceValue(copyOp.getTarget())) {
          builder.setInsertionPoint(copyOp);
          adjustCopyOp(copyOp, builder, slotIdx);
        }
        continue;
      }
    }
  }

  /// Workspace subview: prepend `slotIdx` (tile % S) on the buffer-slot dim.
  void adjustWorkspaceSubview(memref::SubViewOp subview, OpBuilder &builder,
                              Value slotIdx) {
    Location loc = subview.getLoc();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();

    auto origOffsets = subview.getMixedOffsets();
    auto origSizes = subview.getMixedSizes();
    auto origStrides = subview.getMixedStrides();

    int expandedSourceRank = sourceType.getRank();
    int originalSourceRank = expandedSourceRank - 1;

    SmallVector<OpFoldResult> normalizedOffsets, normalizedSizes, normalizedStrides;
    normalizedOffsets.append(origOffsets.begin(), origOffsets.end());
    normalizedSizes.append(origSizes.begin(), origSizes.end());
    normalizedStrides.append(origStrides.begin(), origStrides.end());

    int missingDims = originalSourceRank - (int)origOffsets.size();
    if (missingDims < 0)
      missingDims = 0;

    if (missingDims > 0) {
      for (int i = 0; i < missingDims; ++i) {
        int expandedDimIdx = (int)normalizedOffsets.size() + 1;
        normalizedOffsets.push_back(builder.getIndexAttr(0));
        if (sourceType.isDynamicDim(expandedDimIdx)) {
          normalizedSizes.push_back(
              builder.createOrFold<memref::DimOp>(loc, source, expandedDimIdx));
        } else {
          normalizedSizes.push_back(
              builder.getIndexAttr(sourceType.getDimSize(expandedDimIdx)));
        }
        normalizedStrides.push_back(builder.getIndexAttr(1));
      }
    }

    // Prepend buffer-slot dimension (slotIdx = effective tile % S)
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    newOffsets.push_back(slotIdx);
    newSizes.push_back(builder.getIndexAttr(1));
    newStrides.push_back(builder.getIndexAttr(1));

    for (size_t i = 0; i < normalizedOffsets.size(); ++i) {
      newOffsets.push_back(normalizedOffsets[i]);
      newSizes.push_back(normalizedSizes[i]);
      newStrides.push_back(normalizedStrides[i]);
    }

    auto fullSubview = builder.create<memref::SubViewOp>(
        loc, source, newOffsets, newSizes, newStrides);

    Value replacement = fullSubview.getResult();
    int fullRank = replacement.getType().cast<MemRefType>().getRank();
    int targetRank = subview.getType().cast<MemRefType>().getRank();

    if (fullRank > targetRank) {
      SmallVector<ReassociationIndices> reassociation;
      reassociation.reserve(targetRank);
      ReassociationIndices firstGroup;
      for (int i = 0; i <= fullRank - targetRank; ++i)
        firstGroup.push_back(i);
      reassociation.push_back(std::move(firstGroup));
      for (int i = fullRank - targetRank + 1; i < fullRank; ++i)
        reassociation.push_back({i});

      replacement = builder.create<memref::CollapseShapeOp>(
          loc, replacement, reassociation);
    }

    subview.getResult().replaceAllUsesWith(replacement);
    subview.erase();
  }

  /// Workspace copy: slice with `slotIdx` (tile % S on workspace leading dim).
  void adjustCopyOp(memref::CopyOp copyOp, OpBuilder &builder, Value slotIdx) {
    Location loc = copyOp.getLoc();

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
    offsets.push_back(slotIdx);
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
  llvm::DenseSet<Value> workspaceValueSet_;
  int32_t numStages_;
  Value slotI32_;
  Value slotIdx_;
};

// ---------------------------------------------------------------------------
// SoftPipelineProcessor
// ---------------------------------------------------------------------------
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

    // In two-branch mode (S=2 workspace) the pass bumps upper bound to N+1;
    // otherwise the inner stage loop and linear flag chain are unchanged.
    SoftPipelineConverter converter(pipelineLoop_, scopeOps, workspaceValues_,
                                    numStages);
    bool changed = converter.convert();

    if (changed) {
      insertInitAndClear(pipelineLoop_, numStages, beginCoreType, endCoreType);
      stripLocalAllocMultiBuffer(pipelineLoop_);
    }

    return changed;
  }

private:
  /// In soft-pipeline mode all stages execute sequentially within each outer
  /// iteration, so local (UB) allocs do NOT need multi-buffering.  If we
  /// leave the `annotation.mark {hivm.multi_buffer}` in place, the later
  /// EnableLocalBuffer pass will expand the alloc and index it by innerIV
  /// (= stage index), which causes cross-stage buffers (produced by one
  /// stage, consumed by another) to read from the wrong slot.
  /// Stripping the annotation keeps the alloc shared across all stages.
  void stripLocalAllocMultiBuffer(scf::ForOp outerFor) {
    SmallVector<annotation::MarkOp> toErase;
    outerFor.walk([&](annotation::MarkOp markOp) {
      if (!markOp->hasAttr("hivm.multi_buffer"))
        return;
      Value marked = markOp.getOperand(0);
      if (marked.getDefiningOp<memref::AllocOp>())
        toErase.push_back(markOp);
    });
    for (auto markOp : toErase)
      markOp->erase();
  }

  /// Init: set a single flag so (outer=0, inner=0) can pass its wait.
  /// The first stage waits on flag(t-1) = flag(-1+L) = flag(L-1),
  /// so we pre-set anotherCoreType(beginCoreType) on flag L-1.
  /// Wrapped in a single-trip for loop with hivm.tcore_type so that
  /// SplitMixKernel keeps it only in the correct function.
  ///
  /// Clear: wait on the last flag produced by the pipeline.
  /// The last stage sets endCoreType on flag((N*S-1) % L).
  /// The opposite core must wait on that flag before proceeding.
  void insertInitAndClear(scf::ForOp outerFor, int32_t numStages,
                          hivm::TCoreType beginCoreType,
                          hivm::TCoreType endCoreType) {
    if (beginCoreType == hivm::TCoreType::CUBE_OR_VECTOR)
      return;

    OpBuilder builder(outerFor);
    Location loc = outerFor->getLoc();

    Value c0 = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
    Value c1 = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1));

    // --- Init: single set on flag = SYNC_FLAGS_LIMIT - 1 ---
    hivm::TCoreType initCoreType = anotherCoreType(beginCoreType);
    builder.setInsertionPoint(outerFor);

    auto initForOp = builder.create<scf::ForOp>(loc, c0, c1, c1);
    initForOp->setAttr(hivm::TCoreTypeAttr::name,
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), initCoreType));
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(initForOp.getBody());
      Value initFlagI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(
              static_cast<int64_t>(SYNC_FLAGS_LIMIT) - 1));
      buildCVSyncSet(builder, loc, initCoreType, initFlagI64);
    }

    // --- Clear: single wait on flag = (N * numStages - 1) % L ---
    Value upperBound = outerFor.getUpperBound();
    if (std::optional<int64_t> trip = getConstantIntValue(upperBound);
        trip && *trip <= 0) {
      return;
    }

    hivm::TCoreType clearCoreType = anotherCoreType(endCoreType);
    builder.setInsertionPointAfter(outerFor);

    auto clearForOp = builder.create<scf::ForOp>(loc, c0, c1, c1);
    clearForOp->setAttr(hivm::TCoreTypeAttr::name,
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), clearCoreType));
    {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(clearForOp.getBody());
      Value nI64 = convertToI64(builder, loc, upperBound);
      Value numStagesI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages));
      Value syncLimitI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(),
          builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      Value oneI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
      Value nTimesS = builder.create<arith::MulIOp>(loc, nI64, numStagesI64);
      Value lastT = builder.create<arith::SubIOp>(loc, nTimesS, oneI64);
      Value clearFlagI64 =
          builder.create<arith::RemSIOp>(loc, lastT, syncLimitI64);
      buildCVSyncWait(builder, loc, endCoreType, clearFlagI64);
    }
  }

  scf::ForOp pipelineLoop_;
  SmallVector<Value> workspaceValues_;
};

// ---------------------------------------------------------------------------
// Verify sync closure
// ---------------------------------------------------------------------------
namespace {

static LogicalResult verifySoftPipelineCvSync(ModuleOp module) {
  std::array<int64_t, SYNC_FLAGS_LIMIT> setCnt{};
  std::array<int64_t, SYNC_FLAGS_LIMIT> waitCnt{};
  int64_t skippedNonConst = 0;

  module.walk([&](Operation *op) {
    if (!isa<hivm::SyncBlockSetOp>(op) && !isa<hivm::SyncBlockWaitOp>(op))
      return;

    if (op->getNumOperands() < 1) {
      skippedNonConst++;
      return;
    }
    Value flagVal = op->getOperand(0);
    std::optional<int64_t> cst = getConstantIntValue(flagVal);
    if (!cst) {
      skippedNonConst++;
      return;
    }

    int64_t lim = static_cast<int64_t>(SYNC_FLAGS_LIMIT);
    int64_t mod = *cst % lim;
    if (mod < 0)
      mod += lim;
    auto idx = static_cast<size_t>(mod);

    if (isa<hivm::SyncBlockSetOp>(op))
      setCnt[idx]++;
    else
      waitCnt[idx]++;
  });

  bool mismatch = false;
  for (size_t i = 0; i < SYNC_FLAGS_LIMIT; ++i) {
    if (setCnt[i] == waitCnt[i])
      continue;
    module.emitError()
        << "enable-soft-pipeline verify: flag%16==" << i
        << " has sync_block_set=" << setCnt[i] << " vs sync_block_wait="
        << waitCnt[i] << " (constant flags only)";
    mismatch = true;
  }

  if (skippedNonConst)
    module.emitWarning() << "enable-soft-pipeline verify: skipped "
                         << skippedNonConst
                         << " sync op(s) with non-constant flag SSA";

  return mismatch ? failure() : success();
}

// ---------------------------------------------------------------------------
// Pass entry
// ---------------------------------------------------------------------------
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

  // Step 1: expand workspace buffers with multi-buffer dimension
  if (!WorkspaceExpander::expand(module)) {
    LLVM_DEBUG(DBGS() << "No workspaces to expand.\n");
    return;
  }

  // Step 2: process each function's pipeline loops
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
        (void)forOp.walk([&](Operation *op) -> WalkResult {
          if (usesAnyWs)
            return WalkResult::interrupt();

          if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
            Value src = sv.getSource();
            for (Value ws : expandedWorkspaces) {
              if (src == ws) {
                usesAnyWs = true;
                return WalkResult::interrupt();
              }
            }
          } else if (auto cp = dyn_cast<memref::CopyOp>(op)) {
            Value src = cp.getSource();
            Value dst = cp.getTarget();
            for (Value ws : expandedWorkspaces) {
              if (src == ws || dst == ws) {
                usesAnyWs = true;
                return WalkResult::interrupt();
              }
            }
          }
          return WalkResult::advance();
        });
        if (usesAnyWs) {
          pipelineLoops.push_back(forOp);
        }
      }
    });

    // Step 3: convert each pipeline loop
    for (auto forOp : pipelineLoops) {
      SoftPipelineProcessor processor(forOp, expandedWorkspaces);
      processor.process();
    }
  }

  if (verifySoftPipelineSync) {
    if (failed(verifySoftPipelineCvSync(module)))
      signalPassFailure();
  }
}

} // namespace tilelangir
} // namespace mlir
