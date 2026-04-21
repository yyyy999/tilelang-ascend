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

// 同步标志数量上限，用于CV同步机制
// 限制同步flag的数量，避免资源耗尽
static constexpr size_t SYNC_FLAGS_LIMIT = 16;

/// 扩展MemRef类型，添加多缓冲维度
/// 在原有shape前面添加一个维度用于多缓冲
/// 例如: [M, N] -> [multiBuffer, M, N]
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

/// 若值为 arith.constant，返回其整数值（用于静态核对 sync flag）
static std::optional<int64_t> getConstantIntValue(Value value) {
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = constOp.getValue().dyn_cast_or_null<IntegerAttr>())
      return intAttr.getInt();
  }
  return std::nullopt;
}

/// 将值转换为i64类型，用于同步标志计算
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

/// 获取另一种核心类型（CUBE <-> VECTOR）
static hivm::TCoreType anotherCoreType(const hivm::TCoreType &current) {
  return current == hivm::TCoreType::VECTOR ? hivm::TCoreType::CUBE
                                            : hivm::TCoreType::VECTOR;
}

/// 构建CV同步设置操作 (sync_block_set)
/// 用于通知另一种核心类型当前操作已完成
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

/// 构建CV同步等待操作 (sync_block_wait)
/// 用于等待另一种核心类型的操作完成
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

/// Workspace扩展器
/// 负责将workspace buffer扩展为多缓冲结构
/// 例如: 将 [M, N] 扩展为 [num_stages, M, N]
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

/// Soft Pipeline转换器
/// 将多个ScopeOp转换为单个内层循环 + scf.if分支调度的结构
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

  /// 核心转换方法
  /// 将外层循环内的多个ScopeOp转换为软流水结构
  bool convert() {
    if (scopeOps_.empty())
      return false;

    OpBuilder builder(outerFor_);
    Location loc = outerFor_.getLoc();
    Value outerIV = outerFor_.getInductionVar();

    // 关键：innerFor 必须插入到 outerFor 的 body 内部。
    //
    // 同时，为了保持原先 scope 之前定义的值（例如 index_cast 得到的 offset）对 scope 内部使用的 dominance，
    // innerFor 应当插入在“第一个 ScopeOp 的位置”，而不是 outer body 的开头。
    // 否则会把 scope 内部的 use 移到这些定义之前，触发 dominance 错误。
    if (!scopeOps_.empty() && scopeOps_.front()->getBlock() == outerFor_.getBody()) {
      builder.setInsertionPoint(scopeOps_.front());
    } else {
      builder.setInsertionPointToStart(outerFor_.getBody());
    }

    // 创建常量: 0, num_stages, 1
    Value c0 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                                 builder.getI32IntegerAttr(0));
    Value cNumStages = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(numStages_));
    Value c1 = builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                                 builder.getI32IntegerAttr(1));

    // 创建内层循环: for innerIV = 0 to num_stages step 1
    auto innerFor = builder.create<scf::ForOp>(loc, c0, cNumStages, c1,
                                               ValueRange{});
    innerFor->setAttr("hivm.soft_pipeline", builder.getUnitAttr());
    // 注意：不要在 soft-pipeline 内层循环上再打 tilelangir.num_stages，
    // 该属性语义属于外层 pipeline loop。打在这里会让下游/其他扫描逻辑把内层也当成 pipeline loop，
    // 增加不必要的遍历成本，甚至导致重复处理的风险。

    Block *innerBody = innerFor.getBody();
    Operation *terminator = innerBody->getTerminator();
    Value innerIV = innerFor.getInductionVar();

    builder.setInsertionPoint(innerBody, innerBody->begin());

    // 计算同步标志ID: flag_id = outerIV * num_stages + innerIV
    Value outerIVi64 = convertToI64(builder, loc, outerIV);
    Value innerIVi64 = convertToI64(builder, loc, innerIV);
    Value numStagesI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages_));
    Value syncLimitI64 = builder.create<arith::ConstantOp>(
        loc, builder.getI64Type(), builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));

    Value flagBase = builder.create<arith::MulIOp>(loc, outerIVi64, numStagesI64);
    Value flagIdI64 = builder.create<arith::AddIOp>(loc, flagBase, innerIVi64);
    // 计算下一个外层迭代的flag_id: flag_id_next = flag_id + num_stages
    Value flagIdNextI64 =
        builder.create<arith::AddIOp>(loc, flagIdI64, numStagesI64);
    // 对flag_id取模，限制在SYNC_FLAGS_LIMIT范围内
    Value flagIdMod =
        builder.create<arith::RemSIOp>(loc, flagIdI64, syncLimitI64);
    Value flagIdNextMod =
        builder.create<arith::RemSIOp>(loc, flagIdNextI64, syncLimitI64);

    Value stageIdx =
        builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), innerIV);

    // 计算新的外层索引表达式: newOuterExpr = outerIV * num_stages + innerIV
    // 用于调整全局内存的subview偏移
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

    // 为每个ScopeOp创建scf.if分支
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

      // 创建条件判断: if (innerIV == stageIdx_)
      Value cmpVal = builder.create<arith::CmpIOp>(
          loc, arith::CmpIPredicate::eq, innerIV,
          builder.create<arith::ConstantOp>(loc, builder.getI32Type(),
                                            builder.getI32IntegerAttr(stageIdx_)));

      auto ifOp = builder.create<scf::IfOp>(loc, cmpVal, false);
      Block *thenBlock = &ifOp.getThenRegion().front();
      builder.setInsertionPointToStart(thenBlock);

      // 确定同步的核心类型和flag_id
      hivm::TCoreType waitCoreType = anotherCoreType(coreType);
      // 统一使用取模后的 flag，避免 outer*stages 增大后 wait/set 永远匹配不上导致死锁。
      //
      // 约定：
      // - stage 0/1：等待本次迭代 flagIdMod（由 init 或 stage0 set 提供）
      // - stage >=2：等待 flagIdNextMod（由前一 stage 的 set 提供，形成跨外层迭代叠加）
      // - stage 0：set flagIdMod
      // - stage >0：set flagIdNextMod
      Value waitFlagId = (stageIdx_ <= 1) ? flagIdMod : flagIdNextMod;
      Value setFlagId = (stageIdx_ == 0) ? flagIdMod : flagIdNextMod;

      // 插入同步等待操作: 等待前一个stage完成
      buildCVSyncWait(builder, loc, waitCoreType, waitFlagId);

      // 将ScopeOp内的操作移动到scf.if分支中
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

      // 调整分支内的操作（workspace subview, 全局内存偏移等）
      adjustOperationsInBranch(thenBlock, newOuterExprI32_, newOuterExprIdx,
                               stageIdx_, builder);

      builder.setInsertionPoint(thenBlock->getTerminator());
      // 插入同步设置操作: 通知下一个stage当前操作已完成
      buildCVSyncSet(builder, loc, coreType, setFlagId);
    }

    // 删除原始的ScopeOp
    for (auto scopeOp : scopeOps_) {
      scopeOp->erase();
    }

    return true;
  }

private:
  // 追踪 view-like 链（subview/view/reinterpret_cast），返回根 source。
  // 用于快速判断某个值是否源自 workspace，避免在大 IR 上反复线性扫描 workspaceValues_。
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

  /// 调整scf.if分支内的操作
  /// 包括: workspace subview调整、全局内存偏移调整、copy操作调整
  void adjustOperationsInBranch(Block *thenBlock, Value newOuterExprI32,
                                Value newOuterExprIdx, size_t stageIdx,
                                OpBuilder &builder) {
    Value innerIV = innerFor_.getInductionVar();
    Value outerIV = outerFor_.getInductionVar();

    // 重要：不要在遍历 block 的同时在同一 block 中插入新的 subview/copy，
    // 否则新插入的 op 可能在同一轮遍历中再次被命中，造成重复改写甚至指令膨胀（opt 看起来像“卡住”）。
    SmallVector<Operation *> opsSnapshot;
    opsSnapshot.reserve(thenBlock->getOperations().size());
    for (Operation &op : thenBlock->getOperations())
      opsSnapshot.push_back(&op);

    // Step 1: 先处理 subview/copy（会插入新 op 并 erase 原 op）
    for (Operation *op : opsSnapshot) {
      if (!op || op->getBlock() != thenBlock)
        continue; // 已被 erase 或搬走
      if (op->hasTrait<OpTrait::IsTerminator>())
        continue;

      if (auto subview = dyn_cast<memref::SubViewOp>(op)) {
        builder.setInsertionPoint(subview);
        if (isWorkspaceValue(subview.getSource()))
          adjustWorkspaceSubview(subview, builder);
        else
          adjustGlobalSubviewOffset(subview, newOuterExprI32, builder);
        continue;
      }

      if (auto copyOp = dyn_cast<memref::CopyOp>(op)) {
        if (isWorkspaceValue(copyOp.getSource()) ||
            isWorkspaceValue(copyOp.getTarget())) {
          builder.setInsertionPoint(copyOp);
          adjustCopyOp(copyOp, builder);
        }
        continue;
      }
    }

    // Step 2: 再做通用的 outerIV operand 替换（只针对快照里的原始 op）
    for (Operation *op : opsSnapshot) {
      if (!op || op->getBlock() != thenBlock)
        continue;
      if (op->hasTrait<OpTrait::IsTerminator>())
        continue;
      for (OpOperand &operand : op->getOpOperands()) {
        if (operand.get() == outerIV)
          operand.set(newOuterExprIdx);
      }
    }
  }

  /// 调整workspace的subview操作
  /// 添加stage维度索引，用于多缓冲访问
  /// 例如: subview %ws[%i, ...] -> subview %ws[%stageIdx, %i, ...]
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

    int expandedSourceRank = sourceType.getRank();
    int originalSourceRank = expandedSourceRank - 1;

    SmallVector<OpFoldResult> normalizedOffsets, normalizedSizes, normalizedStrides;
    normalizedOffsets.append(origOffsets.begin(), origOffsets.end());
    normalizedSizes.append(origSizes.begin(), origSizes.end());
    normalizedStrides.append(origStrides.begin(), origStrides.end());

    // 一些 subview 可能是 rank-reducing 的，mixedOffsets 数量可能大于/小于期望值。
    // 这里缺失维度只在为 source 追加隐式维度时才有意义，因此保证它不为负，避免刷屏/异常行为。
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

    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    newOffsets.push_back(stageIndex);
    newSizes.push_back(builder.getIndexAttr(1));
    newStrides.push_back(builder.getIndexAttr(1));

    for (size_t i = 0; i < normalizedOffsets.size(); ++i) {
      auto ofr = normalizedOffsets[i];
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
      newSizes.push_back(normalizedSizes[i]);
      newStrides.push_back(normalizedStrides[i]);
    }

    // 先创建全 rank 的 subview，让 MLIR 推导出正确的 strided layout / offset。
    // 然后再通过 collapse_shape 折叠“前缀维度”，得到和原 subview 等 rank 的结果，
    // 避免直接指定 result type 导致 layout mismatch（尤其是 offset: ?>）。
    auto fullSubview =
        builder.create<memref::SubViewOp>(loc, source, newOffsets, newSizes, newStrides);

    Value replacement = fullSubview.getResult();
    int fullRank =
        replacement.getType().cast<MemRefType>().getRank();
    int targetRank = subview.getType().cast<MemRefType>().getRank();

    if (fullRank > targetRank) {
      // 折叠前缀维度，使 rank 变为 targetRank：
      // firstGroup = [0 .. fullRank-targetRank]
      // remaining = singleton groups
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

  /// 调整全局内存的subview偏移
  /// 使用新的外层索引表达式替换原始的外层循环变量
  void adjustGlobalSubviewOffset(memref::SubViewOp subview, Value newOuterI32,
                                 OpBuilder &builder) {
    Location loc = subview.getLoc();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    
    auto origOffsets = subview.getMixedOffsets();
    auto origSizes = subview.getMixedSizes();
    auto origStrides = subview.getMixedStrides();

    // 处理隐式维度：填充缺失的维度
    SmallVector<OpFoldResult> normalizedOffsets, normalizedSizes, normalizedStrides;
    normalizedOffsets.append(origOffsets.begin(), origOffsets.end());
    normalizedSizes.append(origSizes.begin(), origSizes.end());
    normalizedStrides.append(origStrides.begin(), origStrides.end());

    int missingDims = sourceType.getRank() - (int)origOffsets.size();
    if (missingDims < 0)
      missingDims = 0;
    if (missingDims > 0) {
      for (int i = 0; i < missingDims; ++i) {
        int dimIdx = (int)normalizedOffsets.size();
        normalizedOffsets.push_back(builder.getIndexAttr(0));
        if (sourceType.isDynamicDim(dimIdx)) {
          normalizedSizes.push_back(
              builder.createOrFold<memref::DimOp>(loc, source, dimIdx));
        } else {
          normalizedSizes.push_back(
              builder.getIndexAttr(sourceType.getDimSize(dimIdx)));
        }
        normalizedStrides.push_back(builder.getIndexAttr(1));
      }
    }

    SmallVector<OpFoldResult> newOffsets;
    for (size_t i = 0; i < normalizedOffsets.size(); ++i) {
      auto ofr = normalizedOffsets[i];
      if (auto val = ofr.dyn_cast<Value>()) {
        if (val == outerFor_.getInductionVar()) {
          Value newIdx = builder.create<arith::IndexCastOp>(
              loc, builder.getIndexType(), newOuterI32);
          newOffsets.push_back(newIdx);
        } else {
          newOffsets.push_back(val);
        }
      } else {
        newOffsets.push_back(ofr);
      }
    }

    auto newSubview = builder.create<memref::SubViewOp>(
        loc, source, newOffsets, normalizedSizes, normalizedStrides);
    subview.getResult().replaceAllUsesWith(newSubview.getResult());
    subview.erase();
  }

  /// 调整copy操作
  /// 为workspace相关的copy操作添加stage维度索引
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
  llvm::DenseSet<Value> workspaceValueSet_;
  int32_t numStages_;
  Value newOuterExprI32_;
};

/// Soft Pipeline处理器
/// 负责处理整个pipeline循环的转换
class SoftPipelineProcessor {
public:
  SoftPipelineProcessor(scf::ForOp pipelineLoop,
                        ArrayRef<Value> workspaceValues)
      : pipelineLoop_(pipelineLoop), workspaceValues_(workspaceValues) {}

  /// 处理pipeline循环
  /// 1. 收集ScopeOp列表
  /// 2. 修改外层循环迭代次数
  /// 3. 调用SoftPipelineConverter进行转换
  /// 4. 插入Init/Clear同步循环
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
      insertInitAndClear(pipelineLoop_, numStages, beginCoreType, endCoreType, newUpper);
    }

    return changed;
  }

private:
  /// 插入Init和Clear同步循环
  /// 
  /// Init循环: 在主循环前预先设置同步标志，让第一个stage不被阻塞
  ///   for j = 0 to num_stages:
  ///     sync_block_set[initCoreType](j)
  ///
  /// Clear：主循环后单次 wait，对应最后一轮外层迭代、最后一级 stage 的 set 所用 flag
  ///（末级 stage>0 使用 flag_id + num_stages，线性下标为 newUpper*numStages + numStages - 1）。
  void insertInitAndClear(scf::ForOp outerFor, int32_t numStages,
                          hivm::TCoreType beginCoreType,
                          hivm::TCoreType endCoreType,
                          Value newUpperBound) {
    if (beginCoreType == hivm::TCoreType::CUBE_OR_VECTOR)
      return;

    // Builder 必须有合法插入点，否则 create(...) 可能触发断言/崩溃。
    OpBuilder builder(outerFor);
    Location loc = outerFor->getLoc();

    // 创建常量
    builder.setInsertionPoint(outerFor);
    Value c0 = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
    Value cNumStages = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(numStages));
    Value c1 = builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1));

    // Init核心类型: 与第一个stage的核心类型相反
    // 如果第一个stage是CUBE，则需要初始化VECTOR标志
    hivm::TCoreType initCoreType = anotherCoreType(beginCoreType);

    // 创建Init循环
    builder.setInsertionPoint(outerFor);
    auto initForOp =
        builder.create<scf::ForOp>(loc, c0, cNumStages, c1);
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

    // Clear：等待最后一个 stage 的 set 生效，确保流水线排空。
    // 最后一个 stage 会 set 自己的 coreType 的 flag，因此这里等待 endCoreType。
    hivm::TCoreType clearCoreType = endCoreType;

    if (std::optional<int64_t> trip = getConstantIntValue(newUpperBound);
        trip && *trip <= 0) {
      // 外层 trip 为 0 时不插入会永久阻塞的 clear
      return;
    }

    // 单次迭代的 scf.for（0 到 1 step 1），便于挂上 hivm.tcore_type，与 InsertCVSync 习惯一致
    builder.setInsertionPointAfter(outerFor);
    auto clearForOp = builder.create<scf::ForOp>(loc, c0, c1, c1);
    auto clearCoreTypeAttr =
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), clearCoreType);
    clearForOp->setAttr(hivm::TCoreTypeAttr::name, clearCoreTypeAttr);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *clearBody = clearForOp.getBody();
      builder.setInsertionPointToStart(clearBody);

      Value numStagesI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages));
      Value newUpperI64 = convertToI64(builder, loc, newUpperBound);
      Value prod =
          builder.create<arith::MulIOp>(loc, newUpperI64, numStagesI64);
      Value oneI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(1));
      // 末级 stage 的 set：stage==0 用 flagId；stage>0 用 flagId+numStages。
      // last outer = newUpper-1、last inner = numStages-1 时：
      // - numStages==1：仅 stage0 → lastLinear = newUpper*S - 1
      // - numStages>1：末级 stage>0 → lastLinear = newUpper*S + S - 1
      Value lastLinear;
      if (numStages > 1) {
        Value withStage =
            builder.create<arith::AddIOp>(loc, prod, numStagesI64);
        lastLinear =
            builder.create<arith::SubIOp>(loc, withStage, oneI64);
      } else {
        lastLinear = builder.create<arith::SubIOp>(loc, prod, oneI64);
      }

      Value syncLimitI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      Value flagIdMod =
          builder.create<arith::RemSIOp>(loc, lastLinear, syncLimitI64);

      buildCVSyncWait(builder, clearForOp->getLoc(), clearCoreType, flagIdMod);
    }
  }

  scf::ForOp pipelineLoop_;
  SmallVector<Value> workspaceValues_;
};

namespace {

/// 静态核对：对每个 flag 下标（对 SYNC_FLAGS_LIMIT 取模），constant 的
/// sync_block_set / sync_block_wait 次数是否一致。仅统计 flag 为 arith.constant 的 op；
/// 非常量 flag 记一次 warning 并跳过，不判失败。
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

  // Step 1: 扩展workspace buffer，添加多缓冲维度
  if (!WorkspaceExpander::expand(module)) {
    LLVM_DEBUG(DBGS() << "No workspaces to expand.\n");
    return;
  }

  // Step 2: 遍历每个函数，处理pipeline循环
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    // 收集扩展后的workspace
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

    // 收集使用workspace的pipeline循环
    SmallVector<scf::ForOp> pipelineLoops;
    func.walk([&](scf::ForOp forOp) {
      if (forOp->getAttr("tilelangir.num_stages")) {
        bool usesAnyWs = false;
        // 注意：这里在大 IR 上可能非常慢。找到任意 workspace 使用后应立即中断 walk，
        // 否则会继续遍历整个 loop body，看起来像 opt “卡住”。
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

    // Step 3: 处理每个pipeline循环
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
