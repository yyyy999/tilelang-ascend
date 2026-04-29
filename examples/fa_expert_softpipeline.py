# Copyright (c) Huawei Technologies Co., Ltd. 2025.
import argparse
import torch

import tilelang
import tilelang.language as T

# ---------- 命令行参数 ----------
parser = argparse.ArgumentParser(description="NPU Kernel Compilation")

parser.add_argument(
    "--dtype",
    type=str,
    default="float16",
    help="矩阵运算数据类型 (如 float16)",
)
parser.add_argument(
    "--accum_dtype",
    type=str,
    default="float32",
    help="累加器及 Vector 端数据类型 (高精度，保证数值稳定)",
)
parser.add_argument(
    "--seq_len", type=int, default=8192, help="输入张量的序列长度"
)
parser.add_argument(
    "--dim", type=int, default=128, help="特征维度 (隐层大小)"
)
parser.add_argument(
    "--block_m",
    type=int,
    default=128,
    help="分块中 M 方向 (序列长度) 的块大小",
)
parser.add_argument(
    "--block_n",
    type=int,
    default=256,
    help="分块中 K/V 序列长度方向的块大小",
)


# ---------- FlashAttention 内核定义 ----------
@tilelang.jit(target="npuir", out_idx=[3])
def flash_attn_kernel(dtype, accum_dtype, seq_len, dim, block_m, block_n):
    # ================== 流水线多缓冲配置 ==================
    # Workspace 在 Core 间共享的倍数
    multi_ws_s = 2   # softmax 前 scores 的 workspace
    multi_ws_p = 2   # softmax 后 probabilities 的 workspace
    multi_ws_o = 2   # 输出 partial sum 的 workspace

    # L1 多缓冲 (Cube 侧)
    multi_l0_c = 1   # L0C 缓冲
    multi_l1_q = 1   # Query 的 L1 缓冲
    multi_l1_p = 1   # Probability 的 L1 缓冲
    multi_l1_kv = 1  # Key/Value 的 L1 缓冲

    # UB 多缓冲 (Vector 侧)
    multi_ub_cross = 1          # 跨步缓冲
    multi_ub_inner_cross = 2    # 内部跨步缓冲 (用于 scale/sum 等)

    # ================== 流水线同步标志索引定义 ==================
    # 标志位于不同管道之间，用于生产者/消费者同步。
    # 命名规则：flag_base_{源}_{目标}_{数据}
    # 方向：M (Cube矩阵单元), MTE2 (数据搬运至L1), FIX (标量/搬运单元),
    #       V (Vector向量单元), MTE3 (数据搬运至UB或DDR)

    # Cube - MTE2 -> M: 加载完成，计算开始
    flag_base_m2_to_m_q = 0        # Query 就绪
    flag_base_m2_to_m_p = flag_base_m2_to_m_q + multi_l1_q   # Prob 就绪
    flag_base_m2_to_m_k = flag_base_m2_to_m_q + multi_l1_p   # Key 就绪
    flag_base_m2_to_m_v = flag_base_m2_to_m_k + multi_l1_kv  # Value 就绪

    # Cube - M -> MTE2: 计算完成，加载下一块 (反压)
    flag_base_m_to_m2_q = 0
    flag_base_m_to_m2_k = flag_base_m_to_m2_q + multi_l1_q
    flag_base_m_to_m2_v = flag_base_m_to_m2_k + multi_l1_kv

    # Cube - M -> FIX: 计算完成，存储开始
    flag_base_m_to_fix_s = 0                                   # Score 存回
    flag_base_m_to_fix_o = flag_base_m_to_fix_s + multi_l0_c  # 输出 partial sum 存回

    # Cube - FIX -> M: 存储完成，计算下一块 (反压)
    flag_base_fix_to_m_s = 0
    flag_base_fix_to_m_o = flag_base_fix_to_m_s + multi_l0_c

    # 跨核同步：C-FIX -> V-MTE2 (scores): 存储完成，Vector 可以加载
    flag_base_c_fix_to_v_m2_s = 0
    flag_base_c_fix_to_v_m2_o = flag_base_c_fix_to_v_m2_s + multi_ws_s # 跨步 o

    # 跨核同步：V-MTE2 -> C-FIX: Vector 加载完成，Cube 可以重新写入
    flag_base_v_m2_to_c_fix_s = 0
    flag_base_v_m2_to_c_fix_o = flag_base_v_m2_to_c_fix_s + multi_ws_s

    # Vector - MTE2 -> V: 加载完成，计算开始
    flag_base_m2_to_v_s = 0
    flag_base_m2_to_v_o = flag_base_m2_to_v_s + multi_ub_cross

    # Vector - V -> MTE2: 计算完成，可以再次加载
    flag_base_v_to_m2_s = 0
    flag_base_v_to_m2_o = flag_base_v_to_m2_s + multi_ub_cross

    # Vector - V -> MTE3: 计算完成，存储概率
    flag_base_v_to_m3_p = 0
    flag_base_v_to_m3_o = flag_base_v_to_m3_p + multi_ub_cross

    # Vector - MTE3 -> V: 存储完成，可重新使用 UB
    flag_base_m3_to_v_p = 0
    flag_base_m3_to_v_o = flag_base_m3_to_v_p + multi_ub_cross

    # 跨核同步：V-MTE3 -> C-MTE2 (概率): Vector 写完，Cube 可以加载
    flag_base_v_m3_to_c_m2_p = flag_base_v_m2_to_c_fix_o + multi_ws_o

    # 跨核同步：C-MTE2 -> V-MTE3: Cube 加载完成，Vector 可以再写
    flag_base_c_m2_to_v_m3_p = flag_base_c_fix_to_v_m2_o + multi_ws_o

    # ================== 计算形状相关常量 ==================
    scale = (1.0 / dim) ** 0.5

    num_physical_kernels = 24           # 物理核数（同时运行的kernel实例）
    num_logic_kernels = (seq_len - 1) // block_m + 1   # 逻辑任务数（M方向分块数）
    num_blocks = (seq_len - 1) // block_n + 1          # N方向分块数

    # 输入张量形状
    shape_qkv = [seq_len, dim]
    # workspace 形状
    shape_ws_s = [num_physical_kernels, multi_ws_s, block_m, block_n]
    shape_ws_p = [num_physical_kernels, multi_ws_p, block_m, block_n]
    shape_ws_o = [num_physical_kernels, multi_ws_o, block_m, dim]

    # Vector 端半块大小 (将 block_m 拆成两半由两个 Vector lane 处理)
    block_m_half = (block_m + 1) // 2
    block_share = max(block_n, dim)     # UB 共享空间维度

    # ================== 辅助初始化/清理标志宏 ==================
    @T.macro
    def c_init_flags():
        """Cube 初始化跨核同步标志"""
        with T.rs("PIPE_MTE2"):
            for i in T.serial(multi_ws_p):
                T.sync_block_set(flag_base_c_m2_to_v_m3_p + i)

    @T.macro
    def c_clear_flags():
        """Cube 清理跨核同步标志"""
        with T.rs("PIPE_FIX"):
            for i in T.serial(multi_ws_s + multi_ws_o):
                T.sync_block_wait(flag_base_v_m2_to_c_fix_s + i)

    @T.macro
    def v_init_flags():
        """Vector 初始化跨核同步标志"""
        with T.rs("PIPE_MTE2"):
            for i in T.serial(multi_ws_s + multi_ws_o):
                T.sync_block_set(flag_base_v_m2_to_c_fix_s + i)

    @T.macro
    def v_clear_flags():
        """Vector 清理跨核同步标志"""
        with T.rs("PIPE_MTE3"):
            for i in T.serial(multi_ws_p):
                T.sync_block_wait(flag_base_c_m2_to_v_m3_p + i)

    # ================== Cube 单元宏 ==================
    @T.macro
    def C1L(Q, K, l1_q, l1_kv, total_id, kernel_id):
        """
        Cube 第一阶段加载：Q, K 搬入 L1
        total_id: 全局循环迭代编号
        """
        task_id = total_id // num_blocks  # 当前 M 块索引
        block_id = total_id % num_blocks  # 当前 K/V 块索引

        cid = task_id * num_physical_kernels + kernel_id  # 全局 M 块编号
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_MTE2"):
            if block_id == 0:
                # 只在第一个 K/V 块加载 Q（Q 被复用）
                T.copy(
                    Q[offset_m:offset_m + tail_size_m, :],
                    l1_q[:tail_size_m, :],
                )
            # 每个 K/V 块均需加载对应的 K 分块
            T.copy(
                K[offset_n: offset_n + tail_size_n, :],
                l1_kv[:tail_size_n, :],
            )

    @T.macro
    def C1P(l1_q, l1_kv, l0_c, total_id, kernel_id):
        """
        Cube 第一阶段计算：S = Q * K^T （矩阵乘）
        结果存入 L0C
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_M"):  # 在 Matrix 单元执行
            T.gemm(
                l1_q, l1_kv, l0_c,
                initC=True, b_transpose=True,
                size=[tail_size_m, dim, tail_size_n],
            )

    @T.macro
    def C1S(workspace_s, l0_c, total_id, kernel_id):
        """
        Cube 第一阶段存储：将 S 分数存入 workspace_s
        （用于与 Vector 单元交互）
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_FIX"):  # 使用 FIX 流水线搬运
            # 等待 Vector 端空闲（workspace 可被覆盖）
            flag_v_m2_to_c_fix_s = flag_base_v_m2_to_c_fix_s + total_id % multi_ws_s
            T.sync_block_wait(flag_v_m2_to_c_fix_s)
            T.copy(
                l0_c[:tail_size_m, :tail_size_n],
                workspace_s[kernel_id, total_id % multi_ws_s, :tail_size_m, :tail_size_n],
            )
            # 通知 Vector 端数据已就绪
            flag_c_fix_to_v_m2_s = flag_base_c_fix_to_v_m2_s + total_id % multi_ws_s
            T.sync_block_set(flag_c_fix_to_v_m2_s)

    @T.macro
    def C2L(V, workspace_p, l1_p, l1_kv, total_id, kernel_id):
        """
        Cube 第二阶段加载：加载 Value 分块和 softmax 后的概率 P
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_MTE2"):
            # 加载 V 分块
            T.copy(
                V[offset_n:offset_n + tail_size_n, :],
                l1_kv[:tail_size_n, :],
            )
            # 等待 Vector 端写出 P 完毕
            flag_v_m3_to_c_m2_p = flag_base_v_m3_to_c_m2_p + total_id % multi_ws_p
            T.sync_block_wait(flag_v_m3_to_c_m2_p)
            # 加载 P
            T.copy(
                workspace_p[kernel_id, total_id % multi_ws_p, :tail_size_m, :tail_size_n],
                l1_p[:tail_size_m, :tail_size_n],
            )
            # 通知 Vector 端 P 已被读取，可复用 workspace
            flag_c_m2_to_v_m3_p = flag_base_c_m2_to_v_m3_p + total_id % multi_ws_p
            T.sync_block_set(flag_c_m2_to_v_m3_p)

    @T.macro
    def C2P(l1_p, l1_kv, l0_c, total_id, kernel_id):
        """
        Cube 第二阶段计算：O += P * V （矩阵乘）
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_M"):
            # 注意：这里每次循环都会 initC=True，并非累加，而是每次得到独立的 partial O。
            # 真正的累加逻辑在 Vector 端通过 rescale 实现。
            T.gemm(
                l1_p, l1_kv, l0_c,
                initC=True,
                size=[tail_size_m, tail_size_n, dim],
            )

    @T.macro
    def C2S(workspace_o, l0_c, total_id, kernel_id):
        """
        Cube 第二阶段存储：将 partial O 存入 workspace_o
        """
        task_id = total_id // num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        with T.rs("PIPE_FIX"):
            flag_v_m2_to_c_fix_o = flag_base_v_m2_to_c_fix_o + total_id % multi_ws_o
            T.sync_block_wait(flag_v_m2_to_c_fix_o)
            T.copy(
                l0_c[:tail_size_m, :],
                workspace_o[kernel_id, total_id % multi_ws_o, :tail_size_m, :],
            )
            flag_c_fix_to_v_m2_o = flag_base_c_fix_to_v_m2_o + total_id % multi_ws_o
            T.sync_block_set(flag_c_fix_to_v_m2_o)

    # ================== Vector 单元宏 ==================
    @T.macro
    def V1L(workspace_s, cross_kernel_f16, total_id, kernel_id, vid):
        """
        Vector 第一阶段加载：从 workspace_s 加载 S 分数到 UB
        vid: Vector lane id (0/1)，负责处理 block_m 的一半
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        # 计算当前 lane 对应的 M 范围和有效行数
        real_m = (tail_size_m + 1) // 2
        offset_m_sub = vid * real_m
        real_m = real_m - (tail_size_m % 2) * vid  # 处理最后一块奇数情况

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_MTE2"):
            flag_c_fix_to_v_m2_s = flag_base_c_fix_to_v_m2_s + total_id % multi_ws_s
            T.sync_block_wait(flag_c_fix_to_v_m2_s)
            T.copy(
                workspace_s[kernel_id, total_id % multi_ws_s, offset_m_sub:offset_m_sub + real_m, :tail_size_n],
                cross_kernel_f16[:real_m, :tail_size_n],
            )
            flag_v_m2_to_c_fix_s = flag_base_v_m2_to_c_fix_s + total_id % multi_ws_s
            T.sync_block_set(flag_v_m2_to_c_fix_s)

    @T.macro
    def V1P(cross_kernel_f16, cross_kernel_f32, scores_max, scores_max_prev,
            scores_scale, scores_sum, total_id, kernel_id, vid):
        """
        Vector 第一阶段计算：safe softmax（online rescale）
        - 计算当前块的 max，更新全局 max
        - 使用 scale 对旧的概率进行 rescale，再 exp 得到新概率
        - 最后将 f32 转回 f16 并计算 row sum
        """
        acc_c_scale = scale

        block_id = total_id % num_blocks

        with T.rs("PIPE_V"):
            T.copy(scores_max, scores_max_prev)          # 保存旧的 max
            T.vcast(cross_kernel_f16, cross_kernel_f32, round_mode="rint")  # f16 -> f32

            T.vmul(cross_kernel_f32, acc_c_scale, cross_kernel_f32)  # S * scale
            T.reduce(
                cross_kernel_f32[:, :tail_size_n], scores_max, dims=[1], reduce_mode="max"
            )   # 当前块 row max
            if block_id != 0:
                # 计算 rescale 系数 exp(m_prev - m)
                T.vsub(scores_max_prev, scores_max, scores_scale[total_id % multi_ub_inner_cross, :, :])
                T.vexp(scores_scale[total_id % multi_ub_inner_cross, :, :],
                       scores_scale[total_id % multi_ub_inner_cross, :, :])

            # S - max (safe exp)
            T.vsub(cross_kernel_f32, scores_max, cross_kernel_f32)
            T.vexp(cross_kernel_f32, cross_kernel_f32)
            T.vcast(cross_kernel_f32, cross_kernel_f16, round_mode="rint")  # exp 后转回 f16

            # 计算 row sum
            offset_tmp = (total_id % multi_ub_inner_cross) * block_m_half
            T.reduce(cross_kernel_f32[:, :tail_size_n],
                     scores_sum[offset_tmp:offset_tmp + block_m_half, :],
                     dims=[1], reduce_mode="sum")

    @T.macro
    def V1S(workspace_p, cross_kernel_f16, total_id, kernel_id, vid):
        """
        Vector 第一阶段存储：将概率 P 写入 workspace_p，供 Cube 使用
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        real_m = (tail_size_m + 1) // 2
        offset_m_sub = vid * real_m
        real_m = real_m - (tail_size_m % 2) * vid

        offset_n = block_id * block_n
        tail_size_n = T.min(block_n, seq_len - offset_n)

        with T.rs("PIPE_MTE3"):
            # 等待 Cube 端读取完毕
            flag_c_m2_to_v_m3_p = flag_base_c_m2_to_v_m3_p + total_id % multi_ws_p
            T.sync_block_wait(flag_c_m2_to_v_m3_p)
            T.copy(
                cross_kernel_f16[:real_m, :tail_size_n],
                workspace_p[kernel_id, total_id % multi_ws_p, offset_m_sub:offset_m_sub + real_m, :tail_size_n],
            )
            # 通知 Cube 数据就绪
            flag_v_m3_to_c_m2_p = flag_base_v_m3_to_c_m2_p + total_id % multi_ws_p
            T.sync_block_set(flag_v_m3_to_c_m2_p)

    @T.macro
    def V2L(workspace_o, cross_kernel_f16, total_id, kernel_id, vid):
        """
        Vector 第二阶段加载：从 workspace_o 读取 partial O
        """
        task_id = total_id // num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        real_m = (tail_size_m + 1) // 2
        offset_m_sub = vid * real_m
        real_m = real_m - (tail_size_m % 2) * vid

        with T.rs("PIPE_MTE2"):
            flag_c_fix_to_v_m2_o = flag_base_c_fix_to_v_m2_o + total_id % multi_ws_o
            T.sync_block_wait(flag_c_fix_to_v_m2_o)
            T.copy(
                workspace_o[kernel_id, total_id % multi_ws_o, offset_m_sub:offset_m_sub + real_m, :],
                cross_kernel_f16[:real_m, :dim],
            )
            flag_v_m2_to_c_fix_o = flag_base_v_m2_to_c_fix_o + total_id % multi_ws_o
            T.sync_block_set(flag_v_m2_to_c_fix_o)

    @T.macro
    def V2P(cross_kernel_f16, cross_kernel_f32, acc_o, logsum,
            scores_scale, scores_sum, total_id, kernel_id, vid):
        """
        Vector 第二阶段计算：online partial sum 累加
        利用 logsum 和 scores_scale 实现正确的 rescale 累加：
        O_new = O_old * exp(m_old - m_new) + P_new * V
        logsum 同理 rescale 后累加 row sum
        最后 block 结束时，做 O / logsum 得到最终 softmax 结果
        """
        value_zero = 0
        block_id = total_id % num_blocks

        with T.rs("PIPE_V"):
            if block_id == 0:
                # 第一块时初始化累加器
                T.vbrc(value_zero, logsum)
                T.vbrc(value_zero, acc_o)

            # 对旧的 logsum 和 acc_o 进行 rescale
            T.vmul(logsum, scores_scale[total_id % multi_ub_inner_cross, :, :], logsum)
            offset_tmp = (total_id % multi_ub_inner_cross) * block_m_half
            T.vadd(logsum, scores_sum[offset_tmp:offset_tmp + block_m_half, :], logsum)

            # 累加 O
            T.vcast(cross_kernel_f16, cross_kernel_f32, round_mode="rint")
            T.vmul(acc_o, scores_scale[total_id % multi_ub_inner_cross, :, :], acc_o)
            T.vadd(acc_o[:, :], cross_kernel_f32[:, :dim], acc_o[:, :])

            # 最后一个 K/V 块时，最终归一化
            if block_id == num_blocks - 1:
                T.vdiv(acc_o, logsum, acc_o)
                T.vcast(acc_o, cross_kernel_f16[:, :dim], round_mode="rint")

    @T.macro
    def V2S(Output, cross_kernel_f16, total_id, kernel_id, vid):
        """
        Vector 第二阶段存储：将最终结果写回全局内存
        """
        task_id = total_id // num_blocks
        block_id = total_id % num_blocks

        cid = task_id * num_physical_kernels + kernel_id
        offset_m = cid * block_m
        tail_size_m = T.min(block_m, seq_len - offset_m)

        real_m = (tail_size_m + 1) // 2
        offset_m_sub = vid * real_m
        offset_m_tt = offset_m + offset_m_sub
        real_m = real_m - (tail_size_m % 2) * vid

        with T.rs("PIPE_MTE3"):
            if block_id == num_blocks - 1:
                T.copy(
                    cross_kernel_f16[:real_m, :dim],
                    Output[offset_m_tt:offset_m_tt + real_m, :dim],
                )

    # ================== 顶层 PrimFunc ==================
    @T.prim_func
    def FlashAttnExp(
        Q: T.Tensor(shape_qkv, dtype),
        K: T.Tensor(shape_qkv, dtype),
        V: T.Tensor(shape_qkv, dtype),
        Output: T.Tensor(shape_qkv, dtype),
        workspace_s: T.Tensor(shape_ws_s, dtype),
        workspace_p: T.Tensor(shape_ws_p, dtype),
        workspace_o: T.Tensor(shape_ws_o, dtype),
    ):
        """
        一个 NPU Core 执行的主函数，Cube 和 Vector 在同一核内并行执行，
        通过 slice 迭代实现 M 方向分块，通过 stream 循环实现软件流水。
        """
        with T.Kernel(num_physical_kernels, is_npu=True) as (kernel_id, vid):
            # ---------- Cube 作用域 ----------
            with T.Scope("Cube"):
                # 片上缓冲区分配
                l1_q = T.alloc_L1([block_m, dim], dtype)
                l1_p = T.alloc_L1([block_m, block_n], dtype)
                l1_kv = T.alloc_L1([block_n, dim], dtype)
                l0_c = T.alloc_L0C([block_m, block_share], accum_dtype)

                num_local_tasks = T.ceildiv(num_logic_kernels - kernel_id, num_physical_kernels)
                num_loops_total = num_local_tasks * num_blocks

                c_init_flags()   # 初始化跨核同步标志

                # 软件流水主循环：stream_id 表示时间步
                for stream_id in T.serial(num_loops_total + 1):
                    if stream_id < num_loops_total:
                        total_id = stream_id
                        # 第一阶段：S = Q * K^T，写入 workspace_s
                        C1L(Q, K, l1_q, l1_kv, total_id, kernel_id)
                        C1P(l1_q, l1_kv, l0_c, total_id, kernel_id)
                        C1S(workspace_s, l0_c, total_id, kernel_id)

                    if stream_id > 0:
                        total_id = stream_id - 1
                        # 第二阶段：O = P * V，写入 workspace_o
                        C2L(V, workspace_p, l1_p, l1_kv, total_id, kernel_id)
                        C2P(l1_p, l1_kv, l0_c, total_id, kernel_id)
                        C2S(workspace_o, l0_c, total_id, kernel_id)

                c_clear_flags()

            # ---------- Vector 作用域 ----------
            with T.Scope("Vector"):
                # UB 缓冲区
                logsum = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_max = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_max_prev = T.alloc_ub([block_m_half, 1], accum_dtype)
                scores_scale = T.alloc_ub([multi_ub_inner_cross, block_m_half, 1], accum_dtype)
                scores_sum = T.alloc_ub([multi_ub_inner_cross * block_m_half, 1], accum_dtype)

                cross_kernel_f16 = T.alloc_ub([block_m_half, block_share], dtype)
                cross_kernel_f32 = T.alloc_ub([block_m_half, block_share], accum_dtype)
                acc_o = T.alloc_ub([block_m_half, dim], accum_dtype)

                num_local_tasks = T.ceildiv(num_logic_kernels - kernel_id, num_physical_kernels)
                num_loops_total = num_local_tasks * num_blocks

                v_init_flags()

                # 软件流水
                for stream_id in T.serial(num_loops_total + 1):
                    if stream_id < num_loops_total:
                        total_id = stream_id
                        # 第一阶段 softmax：加载 S，online rescale，写回 P
                        V1L(workspace_s, cross_kernel_f16, total_id, kernel_id, vid)
                        V1P(cross_kernel_f16, cross_kernel_f32, scores_max, scores_max_prev,
                            scores_scale, scores_sum, total_id, kernel_id, vid)
                        V1S(workspace_p, cross_kernel_f16, total_id, kernel_id, vid)

                    if stream_id > 0:
                        total_id = stream_id - 1
                        # 第二阶段累加：加载 partial O，rescale 并累加，最后写回
                        V2L(workspace_o, cross_kernel_f16, total_id, kernel_id, vid)
                        V2P(cross_kernel_f16, cross_kernel_f32, acc_o, logsum,
                            scores_scale, scores_sum, total_id, kernel_id, vid)
                        V2S(Output, cross_kernel_f16, total_id, kernel_id, vid)

                v_clear_flags()

    return FlashAttnExp


# ---------- 测试入口 ----------
def run_test(main_args, verify=True):
    # 获取 JIT 编译后的 kernel
    kernel = flash_attn_kernel(
        main_args.dtype,
        main_args.accum_dtype,
        main_args.seq_len,
        main_args.dim,
        main_args.block_m,
        main_args.block_n,
    )
    num_logic_kernels = (main_args.seq_len - 1) // main_args.block_m + 1
    num_blocks = (main_args.seq_len - 1) // main_args.block_n + 1

    shape_qkv = [main_args.seq_len, main_args.dim]
    shape_sp = [num_logic_kernels, main_args.block_m, main_args.block_n]
    shape_o = [main_args.seq_len, main_args.dim * num_blocks]

    torch.manual_seed(88888888)

    dtype = getattr(torch, main_args.dtype)

    # 随机输入
    q = torch.randn(shape_qkv, dtype=dtype).npu()
    k = torch.randn(shape_qkv, dtype=dtype).npu()
    v = torch.randn(shape_qkv, dtype=dtype).npu()

    # workspace 张量
    w1 = torch.empty(shape_sp, dtype=dtype).npu()
    w2 = torch.empty(shape_sp, dtype=dtype).npu()
    w3 = torch.empty(shape_o, dtype=dtype).npu()

    # 运行自定义 kernel
    o = kernel(q, k, v, w1, w2, w3)

    if verify:
        # 参考 PyTorch 实现
        scale = (1.0 / main_args.dim) ** 0.5
        ref_output = (
            torch.nn.functional.softmax((q @ k.T).to(torch.float32) * scale, dim=-1)
            .to(dtype) @ v
        )
        torch.testing.assert_close(o, ref_output, rtol=1e-2, atol=1e-2)
        print("\033[92mAll check passed!\033[0m")
    else:
        assert o is not None
        print("\033[92mFinished.\033[0m")


if __name__ == "__main__":
    torch.npu.set_device(0)
    tilelang.cache.clear_cache()
    args = parser.parse_args()
    run_test(args)

'''
核心架构总结：

分块与并行

将 Q 按 block_m 分块，K/V 按 block_n 分块。

num_physical_kernels 个 NPU 核并行处理不同的 Q 分块，每个核内部 Cube 和 Vector 异步流水。

Vector 端将 block_m 再拆分为两半，由 vid=0/1 两个向量 lane 处理，提升计算密度。

软件流水

stream_id 从 0 到 num_loops_total，形成前后依赖的流水线：

当 stream_id == t 时，Cube 执行第 t 个 N 块的第一阶段（S 计算），Vector 执行第 t 个 N 块的第一阶段（softmax）。

当 stream_id > 0 时，Cube 执行第 t-1 个 N 块的第二阶段（P×V），Vector 执行第 t-1 个 N 块的第二阶段（累加 O）。

通过 workspace (s, p, o) 进行 Cube ↔ Vector 的数据传递。

Online Softmax

Vector 端维护 scores_max、logsum、scores_scale，实现数值稳定的增量 softmax：

新 max m_new，旧 max m_old，计算 rescale 因子 exp(m_old - m_new)。

对过去的 O 和 logsum 进行 rescale，再累加当前块的贡献。

所有 K/V 块处理完毕后进行 O = O / logsum。

同步机制

使用硬件同步原语 T.sync_block_wait/set 管理 Cube 与 Vector 之间的数据依赖。

多缓冲 (multi_ws_s/p/o) 允许生产者/消费者同时访问不同 buffer，实现 overlap。

此例展示了在昇腾 NPU 上使用 TileLang 实现高性能 FlashAttention 的典型模式。
'''