# TileLangIR Pipeline Pass 原始逻辑文档

## 1. Pass 执行顺序

```
1.  canonicalize
2.  adapt_triton_kernel
3.  insert_workspace
4.  mark_multibuffer
5.  cv_split
6.  infer_mem_scope
7.  merge_copy_chains
8.  enable_multi_buffer
9.  enable_local_buffer
10. specialize_cube
11. bind_workspace_arg
12. plan_workspace_memory
13. insert_cv_sync
14. infer_workspace_size_func
15. lower_memref_ext
16. split_mix_kernel
17. wrap_host_function
```

---

## 2. 各 Pass 详细逻辑

### 2.1 `insert_workspace` (InsertWorkspace.cpp)

**作用**: 检测跨核（Cube/Vector）共享的 `memref.alloc` 缓冲区，将其替换为 `memref_ext.alloc_workspace`。

**逻辑**:
1. 遍历函数中的 `memref.alloc` 操作
2. 检查该 buffer 是否同时被不同 core type 的 HIVM 操作访问
3. 如果是跨核共享 buffer：
   - 创建 `memref_ext.alloc_workspace` 替代原始 `memref.alloc`
   - 在写入端插入 `memref.copy` 将数据写入 workspace
   - 在读取端插入 `memref.copy` 从 workspace 读取数据

**IR 变化**:
```
之前:
  %buf = memref.alloc() : memref<64x256xf32>
  hivm.hir.mmadL1 ... outs(%buf)        // Cube 写入
  hivm.hir.vmul ins(%buf, ...)           // Vector 读取

之后:
  %ws = memref_ext.alloc_workspace() : memref<64x256xf32>
  %buf_c = memref.alloc() : memref<64x256xf32>  // Cube 本地
  hivm.hir.mmadL1 ... outs(%buf_c)
  memref.copy %buf_c, %ws                         // Cube 写入 workspace
  %buf_v = memref.alloc() : memref<64x256xf32>  // Vector 本地
  memref.copy %ws, %buf_v                         // Vector 读取 workspace
  hivm.hir.vmul ins(%buf_v, ...)
```

---

### 2.2 `mark_multibuffer` (MarkMultiBuffer.cpp)

**作用**: 为流水线循环中使用的 workspace 标注 `hivm.multi_buffer` 属性。

**逻辑**:
1. 查找带有 `tilelangir.num_stages` 属性的 `scf::ForOp`
2. 遍历循环体内所有操作的操作数，通过 `ViewLikeOpInterface` 追溯到根定义
3. 如果根定义是 `memref_ext::AllocWorkspaceOp`，则标记该 workspace
4. 在 workspace 后插入 `annotation.mark` 操作，设置 `hivm.multi_buffer = num_stages`

**IR 变化**:
```
之前:
  %ws = memref_ext.alloc_workspace() : memref<2x64x256xf32>
  scf.for ... { tilelangir.num_stages = 4 }
    ... 使用 %ws ...

之后:
  %ws = memref_ext.alloc_workspace() : memref<2x64x256xf32>
  annotation.mark %ws {hivm.multi_buffer = 4 : i32}
  scf.for ... { tilelangir.num_stages = 4 }
    ... 使用 %ws ...
```

---

### 2.3 `cv_split` (CVSplit.cpp)

**作用**: 将循环体内的操作按数据依赖分组，每组包装为 `scope::ScopeOp` 并标注 core type。

**逻辑**:
1. 遍历 `scf::ForOp` 循环体
2. 从 `CopyOpInterface`（涉及 workspace 的 copy 操作）出发
3. 通过 `visitGroupOfOps` 沿数据依赖（用户和定义者）遍历，将相关操作分到同一组
4. 确定每组的 core type：
   - 如果组内所有 HIVM 操作都是同一 core type → 标记为 CUBE 或 VECTOR
   - 如果组内混合了不同 core type → 标记为 CUBE_AND_VECTOR
5. 将每组操作包装进 `scope::ScopeOp`，设置 `hivm.tcore_type` 属性

**分组算法**:
```
对于每个涉及 workspace 的 CopyOp:
  1. 沿 def-use 链向上追溯（操作数定义者）
  2. 沿 def-use 链向下追溯（结果使用者）
  3. 跳过标量操作和 scf::YieldOp
  4. 将遍历到的所有操作分到同一组
```

**FA 算子的分组结果**:
```
scf.for %arg14 {
  scope.scope { hivm.tcore_type = CUBE }    // Scope 1: Load K + QK matmul
  scope.scope { hivm.tcore_type = VECTOR }  // Scope 2: Softmax
  scope.scope { hivm.tcore_type = CUBE }    // Scope 3: Load V + PV matmul
  scope.scope { hivm.tcore_type = VECTOR }  // Scope 4: acc_o 累加
}
```

---

### 2.4 `infer_mem_scope` (InferMemScope.cpp)

**作用**: 推断并传播 memref 的内存地址空间属性。

**5 个阶段**:

| 阶段 | 规则 | 目标 |
|------|------|------|
| Phase 1 | `alloc_workspace` → GM | Workspace 在全局内存 |
| Phase 2 | `mmadL1` 的 mA/mB → L1(cbuf), mC → L0C(cc) | Cube 矩阵乘操作数 |
| Phase 3 | VECTOR-core HIVM op 的所有 memref 操作数 → UB | Vector 操作数 |
| Phase 4 | 函数参数 → GM | 全局输入输出 |
| Phase 5 | 剩余 `memref.alloc` → 根据 `scope.scope` 的 `tcore_type` 推断 | Cube→L1, Vector→UB |

**传播机制**: `MemScopePropagator` 沿 def-use 链传播 address_space，支持 `scf::ForOp`、`scf::YieldOp`、`ViewLikeOpInterface` 等。

---

### 2.5 `merge_copy_chains` (MergeCopyChains.cpp)

**作用**: 合并 `cc→cbuf→gm` 的 copy 链，减少中间拷贝。

**逻辑**:
1. 收集所有 `L0C(cc)→L1(cbuf)` 的 copy 操作
2. 对于每个 cc→cbuf copy，查找 cbuf→gm 的后续 copy
3. 如果中间没有被修改（安全检查），则将 cc→cbuf→gm 合并为 cc→gm
4. 删除中间的 cbuf 分配和 copy

---

### 2.6 `enable_multi_buffer` (EnableMultiBuffer.cpp)

**作用**: 将 workspace 扩展为多缓冲，并将 ScopeOp 转换为内层循环（Unroll 模式）。

**核心类**:

#### `WorkspaceExpander`
1. 查找带 `hivm.multi_buffer` 标注的 `alloc_workspace`
2. 删除 `annotation.mark` 操作
3. 扩展 workspace 类型，在前面添加 `multi_buffer` 维度
   ```
   memref<2x64x256xf32> → memref<4x2x64x256xf32>
   ```
4. 用新的 `alloc_workspace` 替换旧的

#### `ScopeToForConverter`
1. 将每个 `scope::ScopeOp` 转换为 `scf::ForOp`
2. 内层循环: `for j = 0 to num_stages step 1`
3. 将 ScopeOp 内的操作移入内层循环体
4. 计算 `newOuterExpr = outerIV * num_stages + innerIV`，替换对外层循环变量的引用
5. 调整 workspace 的 subview：添加 stage 维度的索引
6. 调整全局内存的 subview：用 `newOuterExpr` 替换偏移量计算

#### `PipelineLoopProcessor`
1. 查找带 `tilelangir.num_stages` 属性的 `scf::ForOp`
2. 收集其内部的 `scope::ScopeOp`
3. 调用 `WorkspaceExpander::expand()` 扩展 workspace
4. 修改外层循环迭代次数：`upperBound = upperBound / num_stages`
5. 对每个 ScopeOp 调用 `ScopeToForConverter::convert()` 转换

**IR 变化**:
```
之前 (cv_split 后):
  scf.for %arg14 = 0 to 32 {        // 外层循环 32 次
    tilelangir.num_stages = 4
    scope.scope { CUBE }   // Scope 1
    scope.scope { VECTOR } // Scope 2
    scope.scope { CUBE }   // Scope 3
    scope.scope { VECTOR } // Scope 4
  }

之后 (enable_multi_buffer 后):
  scf.for %arg14 = 0 to 8 {         // 外层循环 8 次 (= 32/4)
    scf.for %arg15 = 0 to 4 {       // 内层循环 1 (CUBE)
      hivm.tcore_type = CUBE
      // Scope 1 的操作，workspace subview 使用 %arg15 作为 stage index
    }
    scf.for %arg15 = 0 to 4 {       // 内层循环 2 (VECTOR)
      hivm.tcore_type = VECTOR
      // Scope 2 的操作
    }
    scf.for %arg15 = 0 to 4 {       // 内层循环 3 (CUBE)
      hivm.tcore_type = CUBE
      // Scope 3 的操作
    }
    scf.for %arg15 = 0 to 4 {       // 内层循环 4 (VECTOR)
      hivm.tcore_type = VECTOR
      // Scope 4 的操作
    }
  }
```

**关键点**: 4 个 ScopeOp 变成 4 个独立的内层循环，**串行执行**，无法实现 CV 叠加。

---

### 2.7 `enable_local_buffer` (EnableLocalBuffer.cpp)

**作用**: 将带 `hivm.multi_buffer` 标注的 `memref.alloc`（本地缓冲区）扩展为多缓冲。

**逻辑**:
1. 查找带 `hivm.multi_buffer` 标注的 `memref::AllocOp`
2. 删除 `annotation.mark` 操作
3. 扩展 alloc 类型，添加 multi_buffer 维度
   ```
   memref<32x1xf32> → memref<4x32x1xf32>
   ```
4. 通过 `findStageLoop` 查找内层 stage 循环
5. 在使用点创建 `subview + collapse_shape` 按 stage index 访问

**`findStageLoop` 逻辑**:
- 向上遍历操作层次
- 查找 `upperBound == multiBuffer` 且 `step == 1` 的 `scf::ForOp`
- 验证其父循环有 `tilelangir.num_stages` 属性
- 返回找到的 stage 循环

**IR 变化**:
```
之前:
  %alloc_12 = memref.alloc() : memref<32x1xf32>
  annotation.mark %alloc_12 {hivm.multi_buffer = 4}
  scf.for %arg15 = 0 to 4 {
    hivm.hir.vexp ... outs(%alloc_12)   // 直接使用
  }

之后:
  %alloc_12 = memref.alloc() : memref<4x32x1xf32>   // 扩展维度
  scf.for %arg15 = 0 to 4 {
    %subview = memref.subview %alloc_12[%arg15, 0, 0] [1, 32, 1] [1, 1, 1]
    %collapsed = memref.collapse_shape %subview [[0, 1], [2]]
    hivm.hir.vexp ... outs(%collapsed)  // 通过 subview 访问
  }
```

---

### 2.8 `specialize_cube` (SpecializeCube.cpp)

**作用**: 将 Cube 核相关的 `memref.copy` 替换为硬件专用操作。

**逻辑**:
1. 遍历所有 `memref::CopyOp`
2. 检查 source 和 target 的 address_space：
   - `GM → L1(cbuf)`: 替换为 `hivm::ND2NZOp`（数据格式转换，ND→NZ）
   - `L0C(cc) → GM`: 替换为 `hivm::FixpipeOp`（Cube 输出管道）
3. 其他 copy 保持不变

---

### 2.9 `bind_workspace_arg` (bishengir)

**作用**: 将 workspace 的分配与函数参数绑定。

---

### 2.10 `plan_workspace_memory` (PlanWorkspaceMemory.cpp)

**作用**: 为 `alloc_workspace` 操作分配字节偏移量。

**逻辑**:
1. 收集所有没有 offset 的 `alloc_workspace` 操作
2. 按 workspace 参数分组
3. 对每组内的 workspace 按顺序分配偏移量
4. 偏移量 = 前面所有 workspace 的总大小之和
5. 创建带 offset 的新 `alloc_workspace` 替换旧的

**IR 变化**:
```
之前:
  %4 = memref_ext.alloc_workspace() : memref<4x2x64x256xf32>
  %5 = memref_ext.alloc_workspace() : memref<4x2x64x256xf16>
  %6 = memref_ext.alloc_workspace() : memref<4x2x64x128xf32>

之后:
  %4 = memref_ext.alloc_workspace() from %arg2 offset = [0]       : ... memref<4x2x64x256xf32>
  %5 = memref_ext.alloc_workspace() from %arg2 offset = [524288]  : ... memref<4x2x64x256xf16>
  %6 = memref_ext.alloc_workspace() from %arg2 offset = [786432]  : ... memref<4x2x64x128xf32>
```

---

### 2.11 `insert_cv_sync` (InsertCVSync.cpp)

**作用**: 在 Cube/Vector 内层循环中插入同步操作，实现 CV 核间通信。

**核心逻辑**:

#### `InsertCVSyncInSinglePipeline`
1. 查找外层循环（有 `tilelangir.num_stages` 属性）内带有 `hivm.tcore_type` 属性的内层循环
2. 如果内层循环数 < 2，无需同步
3. 对每个内层循环：
   - 在循环体开头插入 `sync_block_wait`（等待另一个核完成）
   - 在循环体末尾插入 `sync_block_set`（通知另一个核完成）

#### `GetFinalFlagIds`
计算同步标志 ID：
- 维护 `vectorFlagCnt` 和 `cubeFlagCnt` 计数器
- Vector 循环 wait Cube 的 flag，set 自己的 flag
- Cube 循环 wait Vector 的 flag，set 自己的 flag
- 使用 `SYNC_FLAGS_LIMIT = 16` 取模实现循环复用

#### `InsertInitAndClearInSinglePipeline`
1. 在外层循环之前插入 init 循环：set 初始 flag，让第一个 Cube 操作不被阻塞
2. 在外层循环之后插入 clear 循环：wait 最后的 flag，确保所有操作完成

**同步模式**:
```
init: for j=0..4 { sync_block_set[VECTOR](j) }  // 初始化 Vector 标志

外层循环:
  内层循环1 (CUBE):  for j=0..4 { wait(VECTOR, j), [C:QK_j], set(CUBE, j) }
  内层循环2 (VECTOR): for j=0..4 { wait(CUBE, j), [V:Softmax_j], set(VECTOR, j+4) }
  内层循环3 (CUBE):  for j=0..4 { wait(VECTOR, j+4), [C:PV_j], set(CUBE, j+4) }
  内层循环4 (VECTOR): for j=0..4 { wait(CUBE, j+4), [V:Acc_j], set(VECTOR, j) }

clear: for j=0..4 { sync_block_wait[CUBE](j) }  // 清除 Cube 标志
```

**问题**: 4 个内层循环串行执行，同步标志在同一循环内递增，无法实现跨循环的 CV 叠加。

---

### 2.12 `split_mix_kernel` (SplitMixKernel.cpp)

**作用**: 将混合内核（同时包含 Cube 和 Vector 操作）拆分为 AIC 和 AIV 两个独立函数。

**逻辑**:
1. 查找带 `mix_mode = "aic"` 的函数
2. 克隆原函数，创建 AIC 和 AIV 两个版本
3. `filterOpsRobust` 迭代删除不属于当前核的操作：
   - AIC: 删除 `hivm.tcore_type = VECTOR` 的循环、操作数含 UB 的操作
   - AIV: 删除 `hivm.tcore_type = CUBE` 的循环、操作数含 CBUF/CC 的操作
4. 迭代删除死代码（无用的 Alloc/View/SubView）
5. 删除循环上的 `tcore_type` 属性

---

## 3. 数据流图

```
FA 源码 (Python)
    │
    ▼
T.Pipelined(num_stages=2)  ──→  scf.for { tilelangir.num_stages = 4 }
    │
    ▼
insert_workspace  ──→  跨核 buffer → alloc_workspace + copy
    │
    ▼
mark_multibuffer  ──→  workspace 标注 hivm.multi_buffer = 4
    │
    ▼
cv_split  ──→  4 个 ScopeOp: CUBE, VECTOR, CUBE, VECTOR
    │
    ▼
infer_mem_scope  ──→  所有 memref 添加 address_space
    │
    ▼
merge_copy_chains  ──→  合并 cc→cbuf→gm 链
    │
    ▼
enable_multi_buffer  ──→  workspace 扩展 + ScopeOp → 4 个内层循环
    │
    ▼
enable_local_buffer  ──→  本地 buffer 扩展 + subview 访问
    │
    ▼
specialize_cube  ──→  copy → nd2nz/fixpipe
    │
    ▼
bind_workspace_arg  ──→  workspace 与函数参数绑定
    │
    ▼
plan_workspace_memory  ──→  workspace 分配 offset
    │
    ▼
insert_cv_sync  ──→  插入 sync_block_wait/set
    │
    ▼
split_mix_kernel  ──→  拆分为 AIC + AIV 函数
```

---

## 4. 当前 Unroll 模式的 CV 叠加问题

### 执行时间线

```
外层 i=0:
  内层循环1 j=0..3: [C:QK_0][C:QK_1][C:QK_2][C:QK_3]
  内层循环2 j=0..3: [V:Soft_0][V:Soft_1][V:Soft_2][V:Soft_3]
  内层循环3 j=0..3: [C:PV_0][C:PV_1][C:PV_2][C:PV_3]
  内层循环4 j=0..3: [V:Acc_0][V:Acc_1][V:Acc_2][V:Acc_3]

时间线:
Cube:   [QK_0][QK_1][QK_2][QK_3]        [PV_0][PV_1][PV_2][PV_3]
Vector:                    [Soft_0][Soft_1][Soft_2][Soft_3]  [Acc_0]...

❌ Cube 和 Vector 串行执行，没有叠加
```

### 根因

`enable_multi_buffer` 将 4 个 ScopeOp 转换为 4 个独立的内层循环，它们在外层循环体内串行执行。`insert_cv_sync` 在每个内层循环内部插入同步，但同步只保证循环内部的数据依赖，无法实现跨循环的 CV 叠加。
