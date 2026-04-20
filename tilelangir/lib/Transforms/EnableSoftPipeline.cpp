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
        workspaceValues_(workspaceValues), numStages_(numStages) {}

  /// 核心转换方法
  /// 将外层循环内的多个ScopeOp转换为软流水结构
  bool convert() {
    if (scopeOps_.empty())
      return false;

    OpBuilder builder(outerFor_);
    Location loc = outerFor_.getLoc();
    Value outerIV = outerFor_.getInductionVar();

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
    innerFor->setAttr("tilelangir.num_stages",
                      builder.getI32IntegerAttr(numStages_));

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
      Value waitFlagId = flagIdMod;
      Value setFlagId = flagIdNextMod;

      // 第一个stage使用原始flag_id（不跨外层迭代）
      if (stageIdx_ == 0) {
        waitFlagId = flagIdI64;
        setFlagId = flagIdI64;
      }

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
  /// 调整scf.if分支内的操作
  /// 包括: workspace subview调整、全局内存偏移调整、copy操作调整
  void adjustOperationsInBranch(Block *thenBlock, Value newOuterExprI32,
                                Value newOuterExprIdx, size_t stageIdx,
                                OpBuilder &builder) {
    Value innerIV = innerFor_.getInductionVar();
    Value outerIV = outerFor_.getInductionVar();

    for (Operation &op : llvm::make_early_inc_range(*thenBlock)) {
      bool opErased = false;
      if (auto subview = dyn_cast<memref::SubViewOp>(&op)) {
        bool isWorkspace = false;
        Value source = subview.getSource();
        auto sourceType = source.getType().cast<MemRefType>();
        
        for (Value ws : workspaceValues_) {
          if (source == ws) {
            isWorkspace = true;
            break;
          }
          if (auto definingOp = source.getDefiningOp()) {
            if (auto subviewSource = dyn_cast<memref::SubViewOp>(definingOp)) {
              if (subviewSource.getSource() == ws) {
                isWorkspace = true;
                break;
              }
            }
          }
        }
        
        if (isWorkspace) {
          adjustWorkspaceSubview(subview, builder);
          opErased = true;
        } else {
          adjustGlobalSubviewOffset(subview, newOuterExprI32, builder);
          opErased = true;
        }
      } else if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        for (Value ws : workspaceValues_) {
          if (copyOp.getSource() == ws || copyOp.getTarget() == ws) {
            adjustCopyOp(copyOp, builder);
            opErased = true;
            break;
          }
        }
      }

      if (opErased)
        continue;

      for (unsigned i = 0; i < op.getNumOperands(); ++i) {
        if (op.getOperand(i) == outerIV) {
          op.setOperand(i, newOuterExprIdx);
        }
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

    int missingDims = originalSourceRank - (int)origOffsets.size();
    
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
  /// Clear循环: 在主循环后等待最后的操作完成
  ///   for j = 0 to num_stages:
  ///     sync_block_wait[clearCoreType]((newUpper - 1) * num_stages + j)
  void insertInitAndClear(scf::ForOp outerFor, int32_t numStages,
                          hivm::TCoreType beginCoreType,
                          hivm::TCoreType endCoreType,
                          Value newUpperBound) {
    if (beginCoreType == hivm::TCoreType::CUBE_OR_VECTOR)
      return;

    OpBuilder builder(outerFor->getContext());
    Location loc = outerFor->getLoc();

    // 创建常量
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

    // Clear核心类型: 与最后一个stage的核心类型相反
    // 如果最后一个stage是VECTOR，则需要等待CUBE标志
    hivm::TCoreType clearCoreType = anotherCoreType(endCoreType);

    // 创建Clear循环
    builder.setInsertionPointAfter(outerFor);
    auto clearForOp =
        builder.create<scf::ForOp>(loc, c0, cNumStages, c1);
    auto clearCoreTypeAttr =
        mlir::hivm::TCoreTypeAttr::get(builder.getContext(), clearCoreType);
    clearForOp->setAttr(hivm::TCoreTypeAttr::name, clearCoreTypeAttr);
    {
      OpBuilder::InsertionGuard guard(builder);
      Block *clearBody = clearForOp.getBody();
      builder.setInsertionPointToStart(clearBody);

      // 计算clear的flag_id: (newUpper - 1) * num_stages + innerIV
      // 等待最后一次外层迭代的对应stage完成
      Value innerIV = clearForOp.getInductionVar();

      // 计算 lastOuterIV = newUpperBound - 1
      Value c1I32 = builder.create<arith::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(1));
      Value lastOuterIV = builder.create<arith::SubIOp>(loc, newUpperBound, c1I32);

      Value numStagesI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(numStages));

      Value lastOuterIVi64 = convertToI64(builder, loc, lastOuterIV);
      Value innerIVi64 = convertToI64(builder, loc, innerIV);

      Value flagBase = builder.create<arith::MulIOp>(loc, lastOuterIVi64, numStagesI64);
      Value flagIdI64 = builder.create<arith::AddIOp>(loc, flagBase, innerIVi64);

      Value syncLimitI64 = builder.create<arith::ConstantOp>(
          loc, builder.getI64Type(), builder.getI64IntegerAttr(SYNC_FLAGS_LIMIT));
      Value flagIdMod =
          builder.create<arith::RemSIOp>(loc, flagIdI64, syncLimitI64);

      buildCVSyncWait(builder, clearForOp->getLoc(), clearCoreType, flagIdMod);
    }
  }

  scf::ForOp pipelineLoop_;
  SmallVector<Value> workspaceValues_;
};

/// Pass入口: TileLangIREnableSoftPipeline
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

    // Step 3: 处理每个pipeline循环
    for (auto forOp : pipelineLoops) {
      SoftPipelineProcessor processor(forOp, expandedWorkspaces);
      processor.process();
    }
  }
}

} // namespace tilelangir
} // namespace mlir
