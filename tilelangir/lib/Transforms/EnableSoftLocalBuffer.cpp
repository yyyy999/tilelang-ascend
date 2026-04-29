//===----------------------------------------------------------------------===//
// TileLangIR enable-soft-local-buffer pass
//===----------------------------------------------------------------------===//

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/MemRefExt/IR/MemRefExt.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "tilelangir/Transforms/Passes.h.inc"

#include <optional>

using namespace mlir;
using namespace mlir::tilelangir;
using namespace bishengir;
using namespace hivm;

namespace mlir {
namespace tilelangir {
#define GEN_PASS_DEF_TILELANGIRENABLESOFTLOCALBUFFER
#include "tilelangir/Transforms/Passes.h.inc"

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

static std::optional<int32_t> getMultiBufferFromMark(Value value) {
  for (Operation *user : value.getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      if (auto attr = markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer"))
        return static_cast<int32_t>(attr.getInt());
    }
  }
  return std::nullopt;
}

static scf::ForOp findNearestFor(Operation *op) {
  Operation *curr = op;
  while (curr) {
    if (auto forOp = dyn_cast<scf::ForOp>(curr))
      return forOp;
    curr = curr->getParentOp();
  }
  return nullptr;
}

static scf::ForOp findParentFor(scf::ForOp inner) {
  if (!inner)
    return nullptr;
  Operation *curr = inner.getOperation()->getParentOp();
  while (curr) {
    if (auto forOp = dyn_cast<scf::ForOp>(curr))
      return forOp;
    curr = curr->getParentOp();
  }
  return nullptr;
}

static scf::ForOp findPipelineFor(Operation *op) {
  Operation *curr = op;
  while (curr) {
    if (auto forOp = dyn_cast<scf::ForOp>(curr)) {
      if (forOp->hasAttr("tilelangir.num_stages"))
        return forOp;
    }
    curr = curr->getParentOp();
  }
  return nullptr;
}

static void collectStageIfOpsInOrder(Block *block, TCoreType coreType,
                                      SmallVector<scf::IfOp> &ifOps) {
  for (Operation &op : *block) {
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      auto tcoreAttr = ifOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type");
      if (tcoreAttr && tcoreAttr.getTcoretype() == coreType) {
        ifOps.push_back(ifOp);
      }
      for (Region &region : ifOp->getRegions()) {
        if (!region.empty())
          collectStageIfOpsInOrder(&region.front(), coreType, ifOps);
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(&op)) {
      collectStageIfOpsInOrder(forOp.getBody(), coreType, ifOps);
    }
  }
}

static SmallVector<scf::IfOp>
getStageIfOpsForCoreType(scf::ForOp pipelineFor, TCoreType coreType) {
  SmallVector<scf::IfOp> ifOps;
  if (!pipelineFor)
    return ifOps;

  collectStageIfOpsInOrder(pipelineFor.getBody(), coreType, ifOps);
  return ifOps;
}

static std::optional<int32_t> getGroupForUserInPipeline(
    Operation *user, scf::ForOp pipelineFor, int32_t G) {
  if (!pipelineFor)
    return std::nullopt;
  // 根据 `user` 这个算子，把它归到 soft-pipeline 顶层 stage 分发的某个
  // `scf.if` 分支里（而不是“最近父 if”）。
  //
  // 原因：stage 分支内部可能还会有其它嵌套 `scf.if`（边界/保护逻辑等），
  // 它们可能不带 `hivm.tcore_type`。如果用最近父 if，很容易把 group 算错，
  // 从而导致 slot 没选对，最终产生 rank 不一致错误。

  TCoreType coreTypes[] = {TCoreType::CUBE, TCoreType::VECTOR};
  for (TCoreType core : coreTypes) {
    auto stageIfOps = getStageIfOpsForCoreType(pipelineFor, core);
    if (stageIfOps.empty())
      continue;

    for (size_t idx = 0; idx < stageIfOps.size(); ++idx) {
      Operation *stageIfOp = stageIfOps[idx].getOperation();
      if (stageIfOp->isProperAncestor(user) ||
          stageIfOp == user->getParentOp()) {
        return static_cast<int32_t>(idx);
      }
    }
  }

  return std::nullopt;
}

static scf::IfOp findStageIfForUser(Operation *user, scf::ForOp pipelineFor) {
  if (!pipelineFor)
    return nullptr;

  TCoreType coreTypes[] = {TCoreType::CUBE, TCoreType::VECTOR};
  for (TCoreType core : coreTypes) {
    auto stageIfOps = getStageIfOpsForCoreType(pipelineFor, core);
    for (auto stageIfOp : stageIfOps) {
      if (stageIfOp->isProperAncestor(user) ||
          stageIfOp == user->getParentOp()) {
        return stageIfOp;
      }
    }
  }
  return nullptr;
}

static Value createSubviewAndCollapse(OpBuilder &builder, Location loc,
                                       Value source,
                                       MemRefType sourceType,
                                       Value slotIndex) {
  // sourceType is expanded type: [M, ...oldDims]
  int64_t rank = sourceType.getRank();
  assert(rank >= 2 && "expanded alloc must have >=2 dims");

  SmallVector<OpFoldResult> offsets, sizes, strides;
  offsets.push_back(slotIndex);
  sizes.push_back(builder.getIndexAttr(1));
  strides.push_back(builder.getIndexAttr(1));

  for (int64_t i = 1; i < rank; ++i) {
    offsets.push_back(builder.getIndexAttr(0));
    if (sourceType.isDynamicDim(i)) {
      sizes.push_back(builder.createOrFold<memref::DimOp>(loc, source, i));
    } else {
      sizes.push_back(builder.getIndexAttr(sourceType.getDimSize(i)));
    }
    strides.push_back(builder.getIndexAttr(1));
  }

  auto subviewOp =
      builder.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);

  // 先固定住扩展出来的 slot 维（dim0），再把前导的 leading dims [0,1]
  // 折叠回原本形状，保证后续 `hivm.*` op 的输入 rank 与期望一致。
  SmallVector<ReassociationIndices> reassociation;
  reassociation.push_back({0, 1});
  for (int64_t i = 2; i < rank; ++i)
    reassociation.push_back({i});

  return builder.create<memref::CollapseShapeOp>(
      loc, subviewOp.getResult(), reassociation);
}

static Value createSlotSubviewForGroup(
    OpBuilder &builder, Location loc, Value allocValue, MemRefType expandedType,
    scf::ForOp pipelineFor, int32_t group, int32_t slotDepth) {
  Value iv = pipelineFor.getInductionVar();
  Value gVal =
      builder.create<arith::ConstantOp>(loc, iv.getType(),
                                        builder.getIntegerAttr(iv.getType(), group));
  Value MVal =
      builder.create<arith::ConstantOp>(loc, iv.getType(),
                                        builder.getIntegerAttr(iv.getType(), slotDepth));

  Value ivMinusG = builder.create<arith::SubIOp>(loc, iv, gVal);
  Value ivMinusGPlusM = builder.create<arith::AddIOp>(loc, ivMinusG, MVal);
  Value slotI = builder.create<arith::RemSIOp>(loc, ivMinusGPlusM, MVal);

  Value slotIndex = slotI;
  if (!slotI.getType().isIndex())
    slotIndex = builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), slotI);

  return createSubviewAndCollapse(builder, loc, allocValue, expandedType, slotIndex);
}

static void replaceOperandWithSubview(Operation *user, Value allocValue, Value subview) {
  for (OpOperand &operand : user->getOpOperands()) {
    if (operand.get() == allocValue)
      operand.set(subview);
  }
}

struct TileLangIREnableSoftLocalBuffer
    : public impl::TileLangIREnableSoftLocalBufferBase<
          TileLangIREnableSoftLocalBuffer> {
  using Base =
      impl::TileLangIREnableSoftLocalBufferBase<TileLangIREnableSoftLocalBuffer>;
  using Base::Base;

  void runOnOperation() override;
};

void TileLangIREnableSoftLocalBuffer::runOnOperation() {
  ModuleOp module = getOperation();
  if (!module)
    return;

  // Collect allocs marked with hivm.multi_buffer.
  SmallVector<std::pair<memref::AllocOp, int32_t>> targets;
  module.walk([&](memref::AllocOp allocOp) {
    if (auto m = getMultiBufferFromMark(allocOp.getResult()))
      targets.emplace_back(allocOp, *m);
  });

  if (targets.empty())
    return;

  for (auto &[allocOp, multiBuffer] : targets) {
    Value oldVal = allocOp.getResult();

    // Remove the mark(s) on this alloc result.
    for (Operation *user : oldVal.getUsers()) {
      if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
        if (markOp->getAttrOfType<IntegerAttr>("hivm.multi_buffer"))
          markOp->erase();
      }
    }

    // Expand alloc type.
    MemRefType oldType = allocOp.getType();
    MemRefType expandedType = expandMemRefType(oldType, multiBuffer);

    // Move alloc out of the innermost loop and one outer loop.
    scf::ForOp innerFor = findNearestFor(allocOp);
    scf::ForOp outerFor = findParentFor(innerFor);
    if (!innerFor || !outerFor)
      continue;

    SmallVector<std::pair<Operation *, int32_t>> usersWithGroup;
    scf::ForOp pipelineFor = nullptr;
    int32_t G = 0;

    for (Operation *user : oldVal.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        continue;

      auto userPipelineFor = findPipelineFor(user);
      if (!userPipelineFor) {
        user->emitError("cannot find pipeline loop for user of multi-buffer alloc");
        continue;
      }

      auto stagesAttr =
          userPipelineFor->getAttrOfType<IntegerAttr>("tilelangir.num_stages");
      if (!stagesAttr) {
        user->emitError("pipeline loop missing tilelangir.num_stages attribute");
        continue;
      }

      int32_t userG = static_cast<int32_t>(stagesAttr.getInt());
      if (userG != multiBuffer) {
        user->emitError("soft local buffer expects tilelangir.num_stages == hivm.multi_buffer");
        continue;
      }

      auto groupOpt = getGroupForUserInPipeline(user, userPipelineFor, userG);
      if (!groupOpt) {
        user->emitError("cannot determine group for user in pipeline");
        continue;
      }

      if (!pipelineFor) {
        pipelineFor = userPipelineFor;
        G = userG;
      }

      usersWithGroup.emplace_back(user, *groupOpt);
    }

    if (!pipelineFor || usersWithGroup.empty())
      continue;

    OpBuilder topBuilder(outerFor.getBody(), outerFor.getBody()->begin());
    auto newAlloc =
        topBuilder.create<memref::AllocOp>(allocOp.getLoc(), expandedType);

    DenseMap<int32_t, Value> groupToSubview;
    Location loc = newAlloc.getLoc();

    for (auto &[user, group] : usersWithGroup) {
      if (!groupToSubview.count(group)) {
        scf::IfOp stageIf = findStageIfForUser(user, pipelineFor);
        if (!stageIf) {
          user->emitError("cannot find stage if for user");
          continue;
        }

        OpBuilder ifBuilder(stageIf.thenBlock(), stageIf.thenBlock()->begin());
        groupToSubview[group] = createSlotSubviewForGroup(
            ifBuilder, loc, newAlloc.getResult(), expandedType,
            pipelineFor, group, multiBuffer);
      }

      replaceOperandWithSubview(user, oldVal, groupToSubview[group]);
    }

    allocOp.erase();
  }
}

} // namespace tilelangir
} // namespace mlir

