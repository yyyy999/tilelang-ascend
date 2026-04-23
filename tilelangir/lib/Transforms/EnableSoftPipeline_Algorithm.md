# EnableSoftPipeline 算法实现文档

## 1. 概述

`EnableSoftPipeline.cpp` 实现了一个 MLIR Pass，用于将流水线循环转换为**软流水模式**。在该模式下，来自不同阶段的 Cube 和 Vector 操作可以重叠执行，从而提高硬件利用率。

## 2. 核心算法架构

### 2.1 主要组件

| 组件 | 职责 |
|------|------|
| `WorkspaceExpander` | 扩展工作空间的内存维度，添加多缓冲支持 |
| `SoftPipelineConverter` | 核心转换逻辑，实现软流水转换 |
| `SoftPipelineProcessor` | 处理整个 Pass 的流程控制 |

### 2.2 同步标志限制

```cpp
static constexpr size_t SYNC_FLAGS_LIMIT = 16;
```

系统最多支持 16 个同步标志。

## 3. 工作空间扩展算法 (WorkspaceExpander)

### 3.1 目标

为工作空间添加多缓冲维度，支持流水线并行。

### 3.2 实现步骤

1. **遍历所有 `AllocWorkspaceOp`**，查找带有 `hivm.multi_buffer` 属性标记的工作空间
2. **提取扩展因子** `multiBuffer`，从 `annotation::MarkOp` 获取
3. **扩展 MemRef 类型**：
   - 在原有形状前添加新维度 `multiBuffer`
   - 更新 stride layout 以保持内存布局正确
4. **替换原操作**：创建新的 `AllocWorkspaceOp`，替换所有使用点

### 3.3 类型扩展示例

```
原始类型: memref<1024xf32>
扩展后:   memref<multiBuffer x 1024xf32>
```

## 4. 软流水转换算法 (SoftPipelineConverter)

### 4.1 两种转换模式

#### 模式一：两分支转换 (Two-Branch)

**前置条件**：
- Scope 数量为偶数且 ≥ 2
- 工作空间缓冲槽位数 S 为偶数且 ≥ 2
- W × S ≤ 16（W 为工作空间数量）

**核心思想**：
将内层阶段循环替换为两个并行的 `scf.if` 分支：
- **前分支**：处理 tile `k*B + b`（b ∈ [0, B)）
- **后分支**：处理 tile `(k-1)*B + b`

其中 `B = S/2`，`slot = tile % S`。

**实现流程**：
```
for outerIV in [0, N):
  ┌─ if front_branch (slot < B):
  │    for b in [0, B):
  │      执行前半部分阶段 (stages 0..B-1)
  └─ else:
       for b in [0, B):
         执行后半部分阶段 (stages B..S-1)
```

#### 模式二：带内部循环转换 (WithInnerLoop)

**适用场景**：不满足两分支条件时的通用方案。

**实现方式**：
- 保留内层循环结构
- 使用线性标志链进行同步
- 标志计算：`t = outerIV * num_stages + innerIV`

### 4.2 同步机制

#### 4.2.1 线性标志同步

```cpp
flag = t % SYNC_FLAGS_LIMIT
flag_prev = (t - 1 + SYNC_FLAGS_LIMIT) % SYNC_FLAGS_LIMIT
```

#### 4.2.2 参数化工作空间标志同步（Flash Attention 专用）

**标志计算公式**（§7.3.1）：
```
flag_id = (workspaceIndex * S + (tile % S)) % SYNC_FLAGS_LIMIT
```

**Flash Attention 特定配置**：
- 4 个阶段 (numStages = 4)
- 3 个工作空间 (qk, pv, softmax)
- 每个工作空间的等待/设置标志映射：

| Stage | Wait Workspace | Set Workspace |
|-------|---------------|---------------|
| 0     | 0 (qk)        | 0 (qk)        |
| 1     | 0 (qk)        | 2 (softmax)   |
| 2     | 2 (softmax)   | 1 (pv)        |
| 3     | 1 (pv)        | 1 (pv)        |

### 4.3 初始化与清理算法 (§7.4)

针对 Flash Attention (W=3, S=共享缓冲槽位数)：

**初始化阶段**：
1. **AIC/CUBE 初始化循环**：设置标志 `[2S, 3S)` 
   - 对应 softmax 工作空间
2. **AIV/VECTOR 初始化循环**：设置标志 `[0, 2S)`
   - 对应 qk 和 pv 工作空间

**清理阶段**：
1. **AIC/CUBE 清理循环**：等待标志 `[0, 2S)` (PIPE_FIX)
   - 排空 qk 和 pv
2. **AIV/VECTOR 清理循环**：等待标志 `[2S, 3S)` (PIPE_MTE3)
   - 排空 softmax

## 5. 同步原语

### 5.1 SyncBlockSet

```
sync_block_set: [core, src_pipe, PIPE_S, flag, ...]
```

设置同步标志，通知其他核心操作已完成。

### 5.2 SyncBlockWait

```
sync_block_wait: [core, PIPE_S, dst_pipe, flag]
```

等待同步标志，阻塞直到依赖操作完成。

### 5.3 Cube-Vector 同步

| 操作 | 源核心 | 源管道 | 目标管道 |
|------|--------|--------|----------|
| Set (Vector) | VECTOR | PIPE_MTE3 | PIPE_S |
| Set (Cube) | CUBE | PIPE_FIX | PIPE_S |
| Wait (Vector) | VECTOR | PIPE_S | PIPE_MTE2 |
| Wait (Cube) | CUBE | PIPE_S | PIPE_MTE2 |

## 6. 算法流程图

```
┌─────────────────────────────────────────┐
│         EnableSoftPipeline Pass         │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    1. WorkspaceExpander::expand()       │
│    - 扩展工作空间维度                    │
│    - 添加 multi_buffer 支持             │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    2. 收集 ScopeOps 和 Workspaces        │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│    3. 检查两分支条件                     │
│    canUseTwoBranch()?                   │
└────────┬────────────────────┬───────────┘
         │                    │
    YES  │                    │ NO
         ▼                    ▼
┌──────────────────┐  ┌──────────────────┐
│ convertTwoBranch │  │convertWithInner  │
│   (两分支模式)    │  │Loop (通用模式)   │
└────────┬─────────┘  └────────┬─────────┘
         │                     │
         └──────────┬──────────┘
                    │
                    ▼
┌─────────────────────────────────────────┐
│    4. 插入同步操作                       │
│    - Wait/Set 标志管理                  │
│    - 初始化/清理循环                    │
└─────────────────────────────────────────┘
```

## 7. 关键数据结构

### 7.1 Scope 信息收集

```cpp
struct ScopeInfo {
  scope::ScopeOp scopeOp;
  hivm::TCoreType coreType;
  int32_t stageIndex;
};
```

### 7.2 转换上下文

```cpp
class SoftPipelineConverter {
  scf::ForOp outerFor_;           // 外层循环
  ArrayRef<scope::ScopeOp> scopeOps_;  // 阶段操作
  ArrayRef<Value> workspaceValues_;    // 工作空间
  int32_t numStages_;             // 阶段数
  DenseSet<Value> workspaceValueSet_;
};
```

## 8. 设计要点

### 8.1 扩展因子来源

扩展因子 S 仅来自 `annotation::MarkOp` 的 `hivm.multi_buffer` 属性，由 `MarkMultiBuffer` 从前端的 `tilelangir.num_stages` 复制到各工作空间。

### 8.2 静态分析约束

- 工作空间的首维必须是静态已知的大小
- 所有工作空间的首维必须相同
- 同步标志总数不能超过 16

### 8.3 Cube/Vector 交替执行

通过同步机制确保：
- Cube 操作完成后通知 Vector
- Vector 操作完成后通知 Cube
- 不同阶段的操作可以流水线重叠

## 9. 性能优化策略

1. **两分支模式**：减少分支开销，提高指令流水效率
2. **参数化标志**：避免标志冲突，支持更复杂的流水线模式
3. **预初始化**：提前设置标志，减少运行时同步开销
4. **延迟清理**：在循环结束后统一清理，避免循环内开销
