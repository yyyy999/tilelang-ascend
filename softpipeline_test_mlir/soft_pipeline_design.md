# Soft Pipeline 设计文档

## 1. 目标

将 FA 算子中 CVCV 四阶段串行执行模式，改为软流水模式，实现 **Cube 和 Vector 核的跨阶段叠加执行**。

### 1.1 期望执行时间线

```
Unroll 模式（当前）:
Cube:   [QK_0][QK_1][QK_2][QK_3]        [PV_0][PV_1][PV_2][PV_3]
Vector:                    [Soft_0][Soft_1][Soft_2][Soft_3]  [Acc_0]...

Soft Pipeline 模式（目标）:
Cube:   [QK_0][QK_1][QK_2][QK_3]  [PV_0][PV_1][PV_2][PV_3]
Vector:       [Soft_0][Soft_1][Soft_2][Soft_3]  [Acc_0][Acc_1][Acc_2][Acc_3]
               ↑ V:Soft_0 与 C:QK_1 叠加！
```

---

## 2. 核心设计

### 2.1 IR 结构对比

#### Unroll 模式（当前）

```mlir
scf.for %arg14 = %c0 to %c8 step %c1 { tilelangir.num_stages = 4 }
  // 4 个独立内层循环，串行执行
  scf.for %arg15 = %c0 to %c4 step %c1 { hivm.tcore_type = CUBE }
    // QK matmul
  }
  scf.for %arg15 = %c0 to %c4 step %c1 { hivm.tcore_type = VECTOR }
    // Softmax
  }
  scf.for %arg15 = %c0 to %c4 step %c1 { hivm.tcore_type = CUBE }
    // PV matmul
  }
  scf.for %arg15 = %c0 to %c4 step %c1 { hivm.tcore_type = VECTOR }
    // acc_o 累加
  }
}
```

#### Soft Pipeline 模式（目标）

```mlir
scf.for %arg14 = %c0 to %c8 step %c1 { tilelangir.num_stages = 4 }
  // 1 个内层循环，4 次迭代，每次迭代通过 scf.if 分发到不同阶段
  scf.for %arg15 = %c0 to %c4 step %c1 { hivm.soft_pipeline }
    // 计算 flag_id = outerIV * num_stages + innerIV
    %flag_id = ...
    %flag_id_mod = rem %flag_id, 16   // SYNC_FLAGS_LIMIT = 16

    scf.if (%arg15 == 0) {            // j==0: CUBE - QK matmul
      sync_block_wait[VECTOR](%flag_id_mod)
      // QK matmul 操作...
      sync_block_set[CUBE](%flag_id_mod)
    }
    scf.if (%arg15 == 1) {            // j==1: VECTOR - Softmax
      sync_block_wait[CUBE](%flag_id_mod)
      // Softmax 操作...
      sync_block_set[VECTOR](%flag_id_next_mod)
    }
    scf.if (%arg15 == 2) {            // j==2: CUBE - PV matmul
      sync_block_wait[VECTOR](%flag_id_next_mod)
      // PV matmul 操作...
      sync_block_set[CUBE](%flag_id_next_mod)
    }
    scf.if (%arg15 == 3) {            // j==3: VECTOR - acc_o
      sync_block_wait[CUBE](%flag_id_next_mod)
      // acc_o 操作...
      sync_block_set[VECTOR](%flag_id_next_mod)
    }
  }
}
```

### 2.2 关键设计决策

| 决策 | 选择 | 原因 |
|------|------|------|
| 循环结构 | 1 个内层循环 + `scf.if` 分发 | 硬件层面 Cube/Vector 是同一核的不同单元，共享同一控制流 |
| 同步方式 | 内嵌在 `scf.if` 分支内 | 同步标志与阶段绑定，不同阶段有不同的 wait/set 对 |
| 标志计算 | `flag_id = outerIV * num_stages + innerIV` | 确保每次迭代有唯一标志，避免冲突 |
| 标志取模 | `% 16` (SYNC_FLAGS_LIMIT) | 硬件同步标志数量有限，需要循环复用 |
| 新 Pass 文件 | `EnableSoftPipeline.cpp` | 独立于原有 `EnableMultiBuffer.cpp`，不影响 unroll 模式 |

---

## 3. 新增 Pass: `enable_soft_pipeline`

### 3.1 文件清单

| 文件 | 作用 |
|------|------|
| `tilelangir/lib/Transforms/EnableSoftPipeline.cpp` | Pass 实现 |
| `tilelangir/include/tilelangir/Transforms/Passes.td` | 添加 TableGen 定义 |
| `tilelangir/lib/Transforms/CMakeLists.txt` | 添加编译源文件 |
| `tilelang/engine/lower.py` | 注册 Pass 到 pipeline |

### 3.2 Pass 定义 (Passes.td)

```tablegen
def TileLangIREnableSoftPipeline : Pass<"tilelangir-enable-soft-pipeline", "::mlir::ModuleOp"> {
  let summary = "TileLangIR Enable Soft Pipeline pass";
  let description = [{
    This pass transforms pipelined loops into soft-pipeline mode where
    Cube and Vector operations from different stages can overlap execution.

    Instead of creating separate inner loops for each scope (unroll mode),
    this pass creates a single inner loop with scf.if dispatch, enabling
    cross-stage CV overlap through embedded synchronization.

    Transformation:
    1. Expands workspace buffers with multi-buffer dimension.
    2. Merges all ScopeOps into a single inner loop.
    3. Dispatches each stage to a scf.if branch based on inner IV.
    4. Inserts sync_block_wait/set within each branch.
    5. Computes flag IDs as outerIV * num_stages + innerIV with modulo.
  }];
}
```

### 3.3 核心算法

```
EnableSoftPipeline::runOnOperation():
  1. 查找带 tilelangir.num_stages 属性的 scf::ForOp
  2. 收集其内部的 scope::ScopeOp 列表
  3. 调用 expandWorkspaces() 扩展 workspace
  4. 调用 convertToSoftPipeline() 转换为软流水

expandWorkspaces():
  同 enable_multi_buffer 的 WorkspaceExpander:
  - 查找 hivm.multi_buffer 标注的 alloc_workspace
  - 扩展类型，添加 multi_buffer 维度
  - 删除 annotation.mark

convertToSoftPipeline():
  输入: 外层循环 + N 个 ScopeOp (N = num_stages)
  输出: 外层循环 + 1 个内层循环 + N 个 scf.if 分支

  步骤:
  a. 修改外层循环迭代次数: upperBound = upperBound / num_stages
  b. 创建内层循环: for j = 0 to num_stages step 1
     - 标记 hivm.soft_pipeline 属性
  c. 在内层循环体内:
     - 计算 flag_id = extsi(outerIV, i64) * num_stages_i64 + extsi(innerIV, i64)
     - 计算 flag_id_mod = rem flag_id, SYNC_FLAGS_LIMIT
     - 计算 flag_id_next = flag_id + num_stages_i64
     - 计算 flag_id_next_mod = rem flag_id_next, SYNC_FLAGS_LIMIT
  d. 对每个 ScopeOp (index = stageIdx):
     - 创建 scf.if (innerIV == stageIdx)
     - 在 scf.if 内:
       i.   插入 sync_block_wait (等待前一个阶段的核)
       ii.  移入 ScopeOp 内的操作（调整 workspace subview）
       iii. 插入 sync_block_set (通知后一个阶段的核)
  e. 删除原始 ScopeOp

  同步规则:
  - stage 0 (CUBE): wait VECTOR, set CUBE    (flag_id_mod)
  - stage 1 (VECTOR): wait CUBE, set VECTOR  (flag_id_next_mod)
  - stage 2 (CUBE): wait VECTOR, set CUBE    (flag_id_next_mod)
  - stage 3 (VECTOR): wait CUBE, set VECTOR  (flag_id_next_mod)
  - 奇数 stage: wait CUBE, set VECTOR
  - 偶数 stage: wait VECTOR, set CUBE

  workspace subview 调整:
  - 原始: subview %ws[%newOuterExpr, ...]
  - 软流水: subview %ws[%innerIV, %newOuterExpr, ...]
    其中 newOuterExpr = outerIV * num_stages + innerIV
```

### 3.4 同步标志设计

```
flag_id = outerIV * num_stages + innerIV

对于 num_stages = 4, SYNC_FLAGS_LIMIT = 16:

外层 i=0:
  j=0 (CUBE):  flag=0,  wait VECTOR(0),  set CUBE(0)
  j=1 (VECTOR): flag=1,  wait CUBE(0),   set VECTOR(1+4=5)
  j=2 (CUBE):  flag=2,  wait VECTOR(5),  set CUBE(5)
  j=3 (VECTOR): flag=3,  wait CUBE(5),   set VECTOR(3+4=7)

外层 i=1:
  j=0 (CUBE):  flag=4,  wait VECTOR(4),  set CUBE(4)
  j=1 (VECTOR): flag=5,  wait CUBE(4),   set VECTOR(5+4=9)
  ...

跨外层迭代叠加:
  i=0, j=3 (VECTOR:Acc_0) set VECTOR(7)
  i=1, j=0 (CUBE:QK_4)   wait VECTOR(4) ← 与 i=0,j=3 的 VECTOR 操作叠加！
```

### 3.5 Init/Clear 同步

```
init: for j=0..num_stages {
  sync_block_set[VECTOR](j)   // 初始化 Vector 标志，让第一个 CUBE 不被阻塞
}

clear: for j=0..num_stages {
  sync_block_wait[CUBE](outerIV*num_stages + j)  // 等待最后的 CUBE 完成
}
```

---

## 4. 受影响 Pass 的修改方案

### 4.1 `insert_cv_sync` — 需要修改

**问题**: 当前逻辑为每个内层循环插入同步，但 soft pipeline 的同步已内嵌在 `scf.if` 分支中。

**修改方案**:
- 检测内层循环是否有 `hivm.soft_pipeline` 属性
- 如果有，跳过该循环的同步插入（同步已在 `enable_soft_pipeline` 中完成）
- 仍然处理 init/clear 循环（或由 `enable_soft_pipeline` 自行生成）

```cpp
// InsertCVSync.cpp 修改点
void InsertCVSyncInSinglePipeline(scf::ForOp outerFor) {
  SmallVector<scf::ForOp> innerFors;
  outerFor.getBody()->walk([&](scf::ForOp innerFor) {
    if (innerFor->hasAttr("hivm.soft_pipeline"))
      return;  // 跳过 soft pipeline 循环
    if (innerFor->hasAttr("hivm.tcore_type"))
      innerFors.push_back(innerFor);
  });
  // ... 原有逻辑
}
```

### 4.2 `enable_local_buffer` — 需要修改

**问题**: `findStageLoop` 查找 `upperBound == multiBuffer` 的内层循环，但 soft pipeline 的内层循环结构不同。

**修改方案**:
- 在 `findStageLoop` 中增加对 `hivm.soft_pipeline` 循环的识别
- Soft pipeline 循环的 `upperBound == num_stages`，父循环有 `tilelangir.num_stages`
- subview 的 stage index 使用内层循环变量

```cpp
// EnableLocalBuffer.cpp 修改点
static scf::ForOp findStageLoop(Operation *op, int32_t multiBuffer) {
  Operation *curr = op;
  while (curr) {
    if (auto forOp = dyn_cast<scf::ForOp>(curr)) {
      Value upper = forOp.getUpperBound();
      APInt upperInt;
      if (!matchPattern(upper, m_ConstantInt(&upperInt)))
        break;
      if (upperInt.getZExtValue() != static_cast<uint64_t>(multiBuffer))
        break;
      Value step = forOp.getStep();
      APInt stepInt;
      if (!matchPattern(step, m_ConstantInt(&stepInt)) || stepInt != 1)
        break;
      Operation *parent = forOp->getParentOp();
      while (parent && !isa<scf::ForOp>(parent))
        parent = parent->getParentOp();
      if (auto outerFor = dyn_cast_or_null<scf::ForOp>(parent)) {
        if (outerFor->hasAttr("tilelangir.num_stages"))
          return forOp;  // 同时匹配 unroll 和 soft_pipeline 循环
      }
      break;
    }
    curr = curr->getParentOp();
  }
  return nullptr;
}
```

**注意**: 实际上 `findStageLoop` 的逻辑对 soft pipeline 循环也是兼容的，因为 soft pipeline 内层循环同样满足 `upperBound == num_stages == multiBuffer`，且父循环有 `tilelangir.num_stages`。**可能不需要修改**，但需要测试验证。

### 4.3 `split_mix_kernel` — 需要修改

**问题**: 当前按 `hivm.tcore_type` 属性决定循环保留/删除。Soft pipeline 的内层循环没有单一 `tcore_type`，包含混合的 `scf.if` 分支。

**修改方案**:
- 检测 `hivm.soft_pipeline` 循环
- AIC 函数：保留 `scf.if` 中 Cube 分支的操作，删除 Vector 分支
- AIV 函数：保留 `scf.if` 中 Vector 分支的操作，删除 Cube 分支
- 删除空 `scf.if` 和无用操作

```cpp
// SplitMixKernel.cpp 修改点

// 新增: 判断 scf.if 分支是否属于指定 core type
static bool isBranchOfCoreType(scf::IfOp ifOp, bool isAIC) {
  // 检查 scf.if 内部操作的 core type
  // 通过检查操作数 address_space 或 HIVM op 的 core type 接口判断
  for (auto &op : ifOp.getThenBlock()->getOperations()) {
    if (auto ctIface = dyn_cast<hivm::CoreTypeInterface>(&op)) {
      auto ct = ctIface.getCoreType();
      if (ct) {
        if (isAIC && *ct == hivm::TCoreType::VECTOR) return false;
        if (!isAIC && *ct == hivm::TCoreType::CUBE) return false;
      }
    }
  }
  return true;
}

// 修改: shouldDeleteComputationOp 增加对 soft pipeline 循环的处理
static bool shouldDeleteComputationOp(Operation *op, bool isAIC) {
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    // 原有 tcore_type 逻辑
    if (auto attr = forOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type")) {
      if (isAIC) return attr.getTcoretype() == TCoreType::VECTOR;
      else return attr.getTcoretype() == TCoreType::CUBE;
    }
    // 新增: soft pipeline 循环不直接删除，而是处理其内部 scf.if
    if (forOp->hasAttr("hivm.soft_pipeline")) {
      return false;  // 不删除循环本身，后续在 filterSoftPipelineLoop 中处理
    }
    return false;
  }
  // ... 原有逻辑
}

// 新增: 处理 soft pipeline 循环内部的 scf.if
static void filterSoftPipelineLoop(scf::ForOp forOp, bool isAIC) {
  forOp.getBody()->walk<WalkOrder::PostOrder>([&](scf::IfOp ifOp) {
    // 检查 scf.if 内部操作是否属于当前核
    bool shouldKeep = isBranchOfCoreType(ifOp, isAIC);
    if (!shouldKeep) {
      // 清空 scf.if 的 then 块，只保留 scf.yield
      ifOp.getThenBlock()->clear();
      OpBuilder builder(ifOp);
      builder.setInsertionPointToEnd(ifOp.getThenBlock());
      builder.create<scf::YieldOp>(ifOp.getLoc());
    }
  });
}
```

### 4.4 不需要修改的 Pass

| Pass | 原因 |
|------|------|
| `insert_workspace` | 在 cv_split 之前，不关心循环结构 |
| `mark_multibuffer` | 在 cv_split 之前，不关心循环结构 |
| `cv_split` | 生成 ScopeOp，soft pipeline 在其后处理 |
| `infer_mem_scope` | 按 memref 类型和操作类型传播，不关心循环结构 |
| `merge_copy_chains` | 按 copy 操作类型匹配，不关心循环结构 |
| `specialize_cube` | 按 address_space 替换 copy，不关心循环结构 |
| `bind_workspace_arg` | 绑定 workspace 参数，不关心循环结构 |
| `plan_workspace_memory` | 分配 workspace offset，不关心循环结构 |
| `infer_workspace_size_func` | 在 insert_cv_sync 之后 |
| `lower_memref_ext` | 在 insert_cv_sync 之后 |
| `wrap_host_function` | 在 split_mix_kernel 之后 |

---

## 5. Pipeline 执行顺序

### 5.1 新的 Pass 顺序

```python
# lower.py 修改
pipeline.add(transforms.mlir.canonicalize, top_down=True)
pipeline.add(transforms.bishengir.adapt_triton_kernel)
pipeline.add(transforms.tilelangir.insert_workspace)
pipeline.add(transforms.tilelangir.mark_multibuffer)
pipeline.add(transforms.tilelangir.cv_split)
pipeline.add(transforms.tilelangir.infer_mem_scope)
pipeline.add(transforms.tilelangir.merge_copy_chains)
# 选择 unroll 或 soft pipeline 模式
if use_soft_pipeline:
    pipeline.add(transforms.tilelangir.enable_soft_pipeline)
else:
    pipeline.add(transforms.tilelangir.enable_multi_buffer)
pipeline.add(transforms.tilelangir.enable_local_buffer)
pipeline.add(transforms.tilelangir.specialize_cube)
pipeline.add(transforms.bishengir.bind_workspace_arg)
pipeline.add(transforms.tilelangir.plan_workspace_memory)
pipeline.add(transforms.tilelangir.insert_cv_sync)       # 需适配 soft pipeline
pipeline.add(transforms.bishengir.infer_workspace_size_func)
pipeline.add(transforms.bishengir.lower_memref_ext)
pipeline.add(transforms.tilelangir.split_mix_kernel)      # 需适配 soft pipeline
pipeline.add(transforms.tilelangir.wrap_host_function)
```

### 5.2 模式选择

通过参数控制使用 unroll 还是 soft pipeline：
- Python 层: `T.Pipelined(num_stages=2, soft_pipeline=True)`
- C++ 层: Pass 构造参数 `soft_pipeline`

---

## 6. 实现步骤

### Step 1: 创建 `EnableSoftPipeline.cpp`

1. 实现 `WorkspaceExpander`（复用 `EnableMultiBuffer` 的逻辑）
2. 实现 `SoftPipelineConverter`:
   - 创建内层循环
   - 为每个 ScopeOp 创建 `scf.if` 分支
   - 在分支内插入同步操作
   - 调整 workspace subview
3. 实现 init/clear 循环生成
4. 注册 Pass

### Step 2: 修改 `Passes.td`

添加 `TileLangIREnableSoftPipeline` 定义。

### Step 3: 修改 `CMakeLists.txt`

添加 `EnableSoftPipeline.cpp` 到编译列表。

### Step 4: 修改 `insert_cv_sync`

检测 `hivm.soft_pipeline` 属性，跳过已处理循环。

### Step 5: 修改 `split_mix_kernel`

处理 soft pipeline 循环内的 `scf.if` 分支拆分。

### Step 6: 修改 `lower.py`

添加模式选择逻辑和 Pass 注册。

### Step 7: 测试验证

- 生成 IR 对比验证
- 运行 FA 算子性能测试
- 验证 CV 叠加效果

---

## 7. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| `scf.if` 分支内操作过多 | 代码膨胀 | 仅在 soft pipeline 模式下使用 |
| 同步标志冲突 | 死锁 | 仔细验证 flag_id 计算和取模逻辑 |
| `enable_local_buffer` 兼容性 | buffer 访问错误 | 测试 `findStageLoop` 对 soft pipeline 循环的识别 |
| `split_mix_kernel` 拆分正确性 | 运行时错误 | 增加 soft pipeline 循环的专门处理逻辑 |
| `scf.if` 条件判断开销 | 性能损失 | 硬件层面条件判断开销极小，可忽略 |
