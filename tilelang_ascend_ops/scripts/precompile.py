#!/usr/bin/env python3
"""
预编译内核脚本

在开发机上运行，编译内核并保存到 kernels 目录。

Usage:
    python scripts/precompile.py
"""

import os
import sys
import shutil
from pathlib import Path

# 设置环境变量
os.environ['TILELANG_ASCEND_MODE'] = 'Developer'

import torch
import tilelang
import tilelang.language as T
import cloudpickle

# 确保在 NPU 上运行
torch.npu.set_device(0)

# 内核目录
SCRIPT_DIR = Path(__file__).parent
PACKAGE_DIR = SCRIPT_DIR.parent / "tilelang_ascend_ops"
KERNELS_DIR = PACKAGE_DIR / "kernels"


def compile_flash_attention():
    """编译 Flash Attention 内核"""
    print("=" * 60)
    print("编译 Flash Attention 内核")
    print("=" * 60)
    
    # 使用动态 shape
    @tilelang.jit(out_idx=[-1], target="npuir")
    def flash_attention_kernel(block_M=64, block_N=64, dtype="float16", accum_dtype="float32"):
        # 动态 shape
        M = T.symbolic("M")
        N = T.symbolic("N")
        
        @T.prim_func
        def main(
            Q: T.Tensor((M, N), dtype),
            K: T.Tensor((M, N), dtype),
            V: T.Tensor((M, N), dtype),
            Output: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(M, block_M), is_npu=True) as (cid, _):
                offset = cid * block_M
                Q_shared = T.alloc_shared([block_M, N], dtype)
                T.copy(Q[offset : offset + block_M, 0 : N], Q_shared)

                K_shared = T.alloc_shared([block_N, N], dtype)
                V_shared = T.alloc_shared([block_N, N], dtype)
                scores = T.alloc_fragment([block_M, block_N], accum_dtype)
                scores_cast = T.alloc_fragment([block_M, block_N], dtype)
                correction = T.alloc_fragment([block_M, 1], accum_dtype)
                local_max = T.alloc_fragment([block_M, 1], accum_dtype)
                local_sum = T.alloc_fragment([block_M, 1], accum_dtype)
                acc_m = T.alloc_fragment([block_M, 1], accum_dtype)
                acc_l = T.alloc_fragment([block_M, 1], accum_dtype)
                acc_o = T.alloc_fragment([block_M, N], accum_dtype)
                tmp = T.alloc_fragment([block_M, block_N], accum_dtype)
                tmp1 = T.alloc_fragment([block_M, 1], accum_dtype)
                new_max = T.alloc_fragment([block_M, 1], accum_dtype)
                scales = T.alloc_fragment([block_M, block_N], accum_dtype)

                value_zero = 0
                scale = (1.0 / N) ** 0.5
                value_min = -T.infinity(accum_dtype)
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, acc_l)
                T.vbrc(value_min, acc_m)
                T.vbrc(scale, scales)

                for k in T.Pipelined(T.ceildiv(M, block_N), num_stages=2):
                    T.copy(K[k * block_N : (k + 1) * block_N, 0 : N], K_shared)
                    T.gemm(Q_shared, K_shared, scores, initC=True, b_transpose=True)

                    T.vmul(scores, scales, scores)
                    T.reduce_max(scores, local_max, dim=1)
                    T.vmax(acc_m, local_max, new_max)
                    T.vsub(acc_m, new_max, tmp1)
                    T.vexp(tmp1, correction)
                    T.vsub(scores, new_max, tmp)
                    T.vexp(tmp, scores)
                    T.reduce_sum(scores, local_sum, dim=1)
                    T.vmul(acc_l, correction, acc_l)
                    T.vadd(acc_l, local_sum, acc_l)
                    T.vmul(acc_o, correction, acc_o)
                    T.vcast(scores, scores_cast, round_mode="rint")
                    T.vbrc(value_zero, tmp1)
                    T.vadd(tmp1, new_max, acc_m)

                    T.copy(V[k * block_N : (k + 1) * block_N, 0 : N], V_shared)
                    T.gemm(scores_cast, V_shared, acc_o, initC=False)

                T.vdiv(acc_o, acc_l, acc_o)
                O_cast = T.alloc_shared([block_M, N], dtype)
                T.vcast(acc_o, O_cast, round_mode="rint")
                real_m = T.min(block_M, M - cid * block_M)
                T.copy(O_cast, Output[cid * block_M : cid * block_m + real_m, 0 : N])

        return main
    
    # 触发编译
    print("正在编译...")
    kernel = flash_attention_kernel()
    
    # 测试运行
    print("测试运行...")
    M, N = 512, 128
    q = torch.randn(M, N, dtype=torch.float16, device="npu")
    k = torch.randn(M, N, dtype=torch.float16, device="npu")
    v = torch.randn(M, N, dtype=torch.float16, device="npu")
    
    output = kernel(q, k, v)
    
    # 验证结果
    scale = (1.0 / N) ** 0.5
    ref = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1
    ).to(torch.float16) @ v
    
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("✓ 验证通过")
    
    return kernel


def save_kernel(kernel, name: str):
    """保存内核到 kernels 目录"""
    kernel_dir = KERNELS_DIR / name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存内核到: {kernel_dir}")
    
    # 保存 metadata
    metadata_path = kernel_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        cloudpickle.dump(kernel.metadata, f)
    print(f"  ✓ metadata.pkl")
    
    # 复制 main.so (启动器)
    launcher_src = Path(kernel.so_launcher_path)
    if launcher_src.exists():
        shutil.copy2(launcher_src, kernel_dir / "main.so")
        print(f"  ✓ main.so (from {launcher_src})")
    else:
        print(f"  ✗ main.so 不存在: {launcher_src}")
    
    # 复制 npu_utils.so (工具库)
    utils_src = Path(kernel.so_utils_path)
    if utils_src.exists():
        shutil.copy2(utils_src, kernel_dir / "npu_utils.so")
        print(f"  ✓ npu_utils.so (from {utils_src})")
    else:
        print(f"  ✗ npu_utils.so 不存在: {utils_src}")
    
    print(f"✓ 内核保存完成: {kernel_dir}")


def main():
    print("TileLang Ascend Operators - 预编译脚本")
    print("=" * 60)
    
    # 清理旧内核
    if KERNELS_DIR.exists():
        shutil.rmtree(KERNELS_DIR)
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 编译 Flash Attention
    kernel = compile_flash_attention()
    save_kernel(kernel, "flash_attention")
    
    print("\n" + "=" * 60)
    print("✓ 所有内核预编译完成！")
    print(f"内核目录: {KERNELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
