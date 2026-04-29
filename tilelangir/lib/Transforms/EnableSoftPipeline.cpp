/*!
 * \file tilelangir/lib/Transforms/EnableSoftPipeline.cpp
 * \brief TileLangIR enable-soft-pipeline pass.
 *
 * Transforms pipelined loops with scope.scope ops into soft-pipeline form,
 * where Cube and Vector stages are dispatched via scf.if conditions.
 * This allows cross-stage overlap on hardware without explicit unrolling.
 *
 * Input shape (MergeCopyChains / MarkMultiBuffer style):
 *   - Inner `scf.for` carries `tilelangir.num_stages = G` (e.g. G = 2).
 *   - Workspace allocs use `annotation.mark { hivm.multi_buffer = G }`; the
 *     intended invariant is **G matches `tilelangir.num_stages`** on the
 *     pipeline loop (same double-buffer depth).
 *
 * Structural transformation (example G = 2, four scopes, two scopes/group):
 *
 *   FOR iv = lb TO ub:
 *     scope ...  // group 0
 *     scope ...
 *     scope ...  // group 1
 *     scope ...
 *   {tilelangir.num_stages = G}
 *
 * becomes:
 *
 *   FOR iv = lb TO ub + (G-1)*step:
 *     slot = iv % G
 *     IF (lb <= iv < ub):           // group 0 — same as iv < ub when lb = 0
 *       ...
 *     IF (lb+step <= iv < ub+step): // group 1 — same as iv > lb when lb = 0, step = 1
 *       ...
 *   {tilelangir.num_stages = G}
 *
 * Workspace: prepend one dimension of static size G for the soft-pipeline slot.
 * No sync ops here — InsertCVSync is a separate pass.
 */

#include "tilelangir/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
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

/// Collect `tilelangir.num_stages` (G) from every `scf.for` under \p root that
/// carries the attribute. All values must agree — the same G is expected on
/// `annotation.mark { hivm.multi_buffer = G }` for workspace allocs.
static std::optional<int32_t>
getConsistentPipelineGroupCount(Operation *root) {
  std::optional<int32_t> g;
  bool conflict = false;
  root->walk([&](scf::ForOp forOp) {
    auto attr = forOp->getAttrOfType<IntegerAttr>("tilelangir.num_stages");
    if (!attr)
      return;
    int32_t v = static_cast<int32_t>(attr.getInt());
    if (v < 2) {
      forOp.emitWarning("tilelangir.num_stages must be >= 2 for soft pipeline");
      conflict = true;
      return;
    }
    if (!g)
      g = v;
    else if (*g != v) {
      forOp.emitError("mismatched tilelangir.num_stages across loops");
      conflict = true;
    }
  });
  if (conflict)
    return std::nullopt;
  return g;
}

// ====================================================================
// Utility: expand a MemRefType by prepending a multi-buffer dimension
// ====================================================================
static MemRefType expandWithMultiBufferDim(MemRefType oldType,
                                           int32_t depth) {
  auto oldShape = oldType.getShape();
  SmallVector<int64_t> newShape;
  newShape.push_back(depth);
  newShape.append(oldShape.begin(), oldShape.end());

  MemRefLayoutAttrInterface layout = oldType.getLayout();
  if (auto strided = dyn_cast<StridedLayoutAttr>(oldType.getLayout())) {
    int64_t leadStride = 1;
    bool isDynamic = false;
    for (int64_t d : oldShape) {
      if (d == ShapedType::kDynamic) {
        isDynamic = true;
        break;
      }
      leadStride *= d;
    }
    SmallVector<int64_t> strides;
    strides.push_back(isDynamic ? ShapedType::kDynamic : leadStride);
    strides.append(strided.getStrides().begin(), strided.getStrides().end());
    layout = StridedLayoutAttr::get(oldType.getContext(),
                                    strided.getOffset(), strides);
  }

  return MemRefType::get(newShape, oldType.getElementType(), layout,
                         oldType.getMemorySpace());
}

// ====================================================================
// Utility: select one slot from the multi-buffer dim via subview,
//          then collapse leading size-1 dimensions.
// ====================================================================
static Value buildSlotAccess(OpBuilder &b, Location loc, Value source,
                             Value slotIdx,
                             ArrayRef<OpFoldResult> origOffsets,
                             ArrayRef<OpFoldResult> origSizes,
                             ArrayRef<OpFoldResult> origStrides) {
  auto srcType = source.getType().cast<MemRefType>();
  int newRank = srcType.getRank();

  SmallVector<OpFoldResult> offsets, sizes, strides;

  offsets.push_back(slotIdx);
  sizes.push_back(b.getIndexAttr(1));
  strides.push_back(b.getIndexAttr(1));

  offsets.append(origOffsets.begin(), origOffsets.end());
  sizes.append(origSizes.begin(), origSizes.end());
  strides.append(origStrides.begin(), origStrides.end());

  int missing = newRank - (int)offsets.size();
  for (int i = 0; i < missing; ++i) {
    int dimIdx = (int)offsets.size();
    offsets.push_back(b.getIndexAttr(0));
    if (srcType.isDynamicDim(dimIdx))
      sizes.push_back(
          b.createOrFold<memref::DimOp>(loc, source, dimIdx));
    else
      sizes.push_back(b.getIndexAttr(srcType.getDimSize(dimIdx)));
    strides.push_back(b.getIndexAttr(1));
  }

  auto svOp =
      b.create<memref::SubViewOp>(loc, source, offsets, sizes, strides);

  auto svType = svOp.getResult().getType().cast<MemRefType>();
  SmallVector<ReassociationIndices> reassoc;
  ReassociationIndices group;
  bool merged = false;
  for (int i = 0; i < svType.getRank(); ++i) {
    if (!merged && svType.getDimSize(i) == 1) {
      group.push_back(i);
    } else {
      if (!group.empty()) {
        group.push_back(i);
        reassoc.push_back(group);
        group.clear();
        merged = true;
      } else {
        reassoc.push_back({i});
      }
    }
  }
  if (!group.empty())
    reassoc.push_back(group);

  return b.create<memref::CollapseShapeOp>(loc, svOp.getResult(), reassoc);
}

// ====================================================================
// Step 1: Expand workspace allocations with multi-buffer dimension
// ====================================================================
static bool expandWorkspaces(Operation *root, int32_t slotDepth,
                             SmallVectorImpl<Value> &expanded) {
  SmallVector<memref_ext::AllocWorkspaceOp> targets;
  root->walk([&](memref_ext::AllocWorkspaceOp alloc) {
    for (auto *user : alloc.getResult().getUsers()) {
      if (auto mark = dyn_cast<annotation::MarkOp>(user)) {
        if (mark->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
          targets.push_back(alloc);
          break;
        }
      }
    }
  });

  if (targets.empty())
    return false;

  for (auto alloc : targets) {
    Value val = alloc.getResult();
    annotation::MarkOp markOp = nullptr;
    for (auto *user : val.getUsers()) {
      if (auto m = dyn_cast<annotation::MarkOp>(user)) {
        if (m->getAttrOfType<IntegerAttr>("hivm.multi_buffer")) {
          markOp = m;
          break;
        }
      }
    }
    if (!markOp)
      continue;

    markOp->erase();

    auto oldType = val.getType().cast<MemRefType>();
    auto newType = expandWithMultiBufferDim(oldType, slotDepth);

    OpBuilder b(alloc);
    auto newAlloc = b.create<memref_ext::AllocWorkspaceOp>(
        alloc.getLoc(), newType, alloc.getWorkspaceArg(),
        alloc.getDynamicSize(), alloc.getOffset());

    val.replaceAllUsesWith(newAlloc.getResult());
    alloc.erase();
    expanded.push_back(newAlloc.getResult());

    LLVM_DEBUG(DBGS() << "Expanded workspace → " << newType << "\n");
  }
  return true;
}

// ====================================================================
// Step 2a: Fix workspace subview ops inside a block
// ====================================================================
static void fixWorkspaceSubviews(Block *block, Value slotIdx,
                                 ArrayRef<Value> wsList) {
  SmallVector<memref::SubViewOp> toFix;
  for (auto &op : *block) {
    if (auto sv = dyn_cast<memref::SubViewOp>(&op))
      if (llvm::is_contained(wsList, sv.getSource()))
        toFix.push_back(sv);
  }

  for (auto sv : toFix) {
    OpBuilder b(sv);
    Value result =
        buildSlotAccess(b, sv.getLoc(), sv.getSource(), slotIdx,
                        sv.getMixedOffsets(), sv.getMixedSizes(),
                        sv.getMixedStrides());
    sv.replaceAllUsesWith(result);
    sv.erase();
  }
}

// ====================================================================
// Step 2b: Process one pipeline loop
// ====================================================================
static void processLoop(scf::ForOp forOp, ArrayRef<Value> wsList,
                        int32_t numGroups) {
  auto stagesAttr =
      forOp->getAttrOfType<IntegerAttr>("tilelangir.num_stages");
  if (!stagesAttr)
    return;

  if (static_cast<int32_t>(stagesAttr.getInt()) != numGroups) {
    forOp.emitWarning(
        "tilelangir.num_stages differs from module-wide G; skipping loop");
    return;
  }
  int32_t G = numGroups;

  SmallVector<scope::ScopeOp> scopes;
  for (auto &op : forOp.getBody()->getOperations())
    if (auto s = dyn_cast<scope::ScopeOp>(&op))
      scopes.push_back(s);

  if ((int)scopes.size() < G)
    return;

  int32_t perGroup = (int32_t)scopes.size() / G;
  if (perGroup * G != (int32_t)scopes.size()) {
    forOp.emitWarning(
        "scope count not evenly divisible by tilelangir.num_stages; skipping");
    return;
  }

  Location loc = forOp.getLoc();
  Value iv = forOp.getInductionVar();
  Value origUpper = forOp.getUpperBound();

  // --- Extend upper bound by (G - 1) * step (pipeline drain) ---
  OpBuilder preBuilder(forOp);
  Value step = forOp.getStep();
  Value newUpper = origUpper;
  if (G > 1) {
    Value extra = preBuilder.create<arith::ConstantOp>(
        loc, step.getType(),
        preBuilder.getIntegerAttr(step.getType(), G - 1));
    Value delta = preBuilder.create<arith::MulIOp>(loc, extra, step);
    newUpper = preBuilder.create<arith::AddIOp>(loc, origUpper, delta);
  }
  forOp.setUpperBound(newUpper);

  // --- Insert per-group conditions & slot before the first scope ---
  OpBuilder b(scopes.front());
  Value lowerBound = forOp.getLowerBound();

  SmallVector<Value> condPerGroup;
  condPerGroup.reserve(G);
  for (int32_t g = 0; g < G; ++g) {
    Value gv = b.create<arith::ConstantOp>(
        loc, iv.getType(), b.getIntegerAttr(iv.getType(), g));
    Value gStep = b.create<arith::MulIOp>(loc, gv, step);
    Value start = b.create<arith::AddIOp>(loc, lowerBound, gStep);
    Value end = b.create<arith::AddIOp>(loc, origUpper, gStep);
    Value ge = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, iv,
                                       start);
    Value lt =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, iv, end);
    condPerGroup.push_back(b.create<arith::AndIOp>(loc, ge, lt));
  }

  Value cG =
      b.create<arith::ConstantOp>(loc, iv.getType(),
                                  b.getIntegerAttr(iv.getType(), G));
  Value slotI = b.create<arith::RemSIOp>(loc, iv, cG);
  Value slotIdx = slotI;
  if (!slotI.getType().isIndex())
    slotIdx = b.create<arith::IndexCastOp>(loc, b.getIndexType(), slotI);

  LLVM_DEBUG(DBGS() << "Processing loop: " << scopes.size() << " scopes, G="
                    << G << ", " << perGroup << " scopes/group\n");

  // --- Convert each scope → scf.if ---
  for (int i = 0; i < (int)scopes.size(); ++i) {
    auto scopeOp = scopes[i];
    int group = i / perGroup;
    Value cond = condPerGroup[group];

    b.setInsertionPoint(scopeOp);
    auto ifOp = b.create<scf::IfOp>(loc, cond, /*withElseRegion=*/false);

    for (auto attr : scopeOp->getAttrs()) {
      StringRef name = attr.getName().getValue();
      if (name == "operandSegmentSizes" ||
          name == "operand_segment_sizes" ||
          name == "resultSegmentSizes" ||
          name == "result_segment_sizes")
        continue;
      ifOp->setAttr(attr.getName(), attr.getValue());
    }

    Block *scopeBody = &scopeOp.getRegion().front();
    Block *ifBody = ifOp.thenBlock();
    Operation *yield = ifBody->getTerminator();

    SmallVector<Operation *> ops;
    for (auto &op : scopeBody->getOperations())
      if (!isa<scope::ReturnOp>(op))
        ops.push_back(&op);

    for (auto *op : ops)
      op->moveBefore(yield);

    fixWorkspaceSubviews(ifBody, slotIdx, wsList);

    scopeOp.erase();
  }

  forOp->setAttr("tilelangir.num_stages", preBuilder.getI32IntegerAttr(G));
}

// ====================================================================
// Pass definition
// ====================================================================
namespace {
struct TileLangIREnableSoftPipeline
    : public impl::TileLangIREnableSoftPipelineBase<
          TileLangIREnableSoftPipeline> {
  using Base =
      impl::TileLangIREnableSoftPipelineBase<TileLangIREnableSoftPipeline>;
  using Base::Base;

  void runOnOperation() override;
};
} // namespace

void TileLangIREnableSoftPipeline::runOnOperation() {
  ModuleOp module = getOperation();
  if (!module)
    return;

  LLVM_DEBUG(DBGS() << "=== EnableSoftPipeline pass start ===\n");

  std::optional<int32_t> gOpt = getConsistentPipelineGroupCount(module);
  if (!gOpt) {
    LLVM_DEBUG(DBGS() << "No tilelangir.num_stages on any scf.for; nothing to "
                         "do.\n");
    return;
  }
  int32_t numGroups = *gOpt;

  SmallVector<Value> expandedWS;
  if (!expandWorkspaces(module, numGroups, expandedWS)) {
    LLVM_DEBUG(DBGS() << "No annotated workspaces found; nothing to do.\n");
    return;
  }

  for (auto func : module.getOps<func::FuncOp>()) {
    SmallVector<scf::ForOp> loops;
    func.walk([&](scf::ForOp f) {
      if (f->getAttr("tilelangir.num_stages"))
        loops.push_back(f);
    });

    for (auto f : loops)
      processLoop(f, expandedWS, numGroups);
  }

  LLVM_DEBUG(DBGS() << "=== EnableSoftPipeline pass done ===\n");
}

} // namespace tilelangir
} // namespace mlir
