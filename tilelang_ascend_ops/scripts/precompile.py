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


def compile_flash_attention_kernel():
    """编译 Flash Attention 内核"""
    print("=" * 60)
    print("编译 Flash Attention 内核")
    print("=" * 60)
    
    seq_len = 512
    dim = 128
    block_m = 64
    block_n = 64
    
    @tilelang.jit(out_idx=[-1], target="npuir")
    def flash_attention_kernel(dtype="float16", accum_dtype="float32"):
        @T.prim_func
        def main(
            Q: T.Tensor((seq_len, dim), dtype),
            K: T.Tensor((seq_len, dim), dtype),
            V: T.Tensor((seq_len, dim), dtype),
            Output: T.Tensor((seq_len, dim), dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_m), is_npu=True) as (cid, _):
                offset = cid * block_m
                Q_shared = T.alloc_shared([block_m, dim], dtype)
                T.copy(Q[offset : offset + block_m, 0 : dim], Q_shared)

                K_shared = T.alloc_shared([block_n, dim], dtype)
                V_shared = T.alloc_shared([block_n, dim], dtype)
                scores = T.alloc_fragment([block_m, block_n], accum_dtype)
                scores_cast = T.alloc_fragment([block_m, block_n], dtype)
                correction = T.alloc_fragment([block_m, 1], accum_dtype)
                local_max = T.alloc_fragment([block_m, 1], accum_dtype)
                local_sum = T.alloc_fragment([block_m, 1], accum_dtype)
                acc_m = T.alloc_fragment([block_m, 1], accum_dtype)
                acc_l = T.alloc_fragment([block_m, 1], accum_dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                tmp = T.alloc_fragment([block_m, block_n], accum_dtype)
                tmp1 = T.alloc_fragment([block_m, 1], accum_dtype)
                new_max = T.alloc_fragment([block_m, 1], accum_dtype)
                scales = T.alloc_fragment([block_m, block_n], accum_dtype)

                value_zero = 0
                scale = (1.0 / dim) ** 0.5
                value_min = -T.infinity(accum_dtype)
                T.vbrc(value_zero, acc_o)
                T.vbrc(value_zero, acc_l)
                T.vbrc(value_min, acc_m)
                T.vbrc(scale, scales)

                for k in T.Pipelined(T.ceildiv(seq_len, block_n), num_stages=2):
                    T.copy(K[k * block_n : (k + 1) * block_n, 0 : dim], K_shared)
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

                    T.copy(V[k * block_n : (k + 1) * block_n, 0 : dim], V_shared)
                    T.gemm(scores_cast, V_shared, acc_o, initC=False)

                T.vdiv(acc_o, acc_l, acc_o)
                O_cast = T.alloc_shared([block_m, dim], dtype)
                T.vcast(acc_o, O_cast, round_mode="rint")
                real_m = T.min(block_m, seq_len - cid * block_m)
                T.copy(O_cast, Output[cid * block_m : cid * block_m + real_m, 0 : dim])

        return main
    
    # 触发编译
    print("正在编译...")
    kernel = flash_attention_kernel()
    
    print(f"Kernel 编译完成")
    print(f"  - symbolic: {kernel.symbolic}")
    print(f"  - param_info: {kernel.param_info}")
    print(f"  - out_idx: {kernel.out_idx}")
    
    # 测试运行
    print("\n测试运行...")
    q = torch.randn(seq_len, dim, dtype=torch.float16, device="npu")
    k = torch.randn(seq_len, dim, dtype=torch.float16, device="npu")
    v = torch.randn(seq_len, dim, dtype=torch.float16, device="npu")
    
    output = kernel(q, k, v)
    
    # 验证结果
    scale = (1.0 / dim) ** 0.5
    ref = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1
    ).to(torch.float16) @ v
    
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("✓ 验证通过")
    
    return kernel


def compile_dynamic_gemm_kernel():
    """编译动态shape的GEMM kernel"""
    print("=" * 60)
    print("编译动态shape GEMM 内核")
    print("=" * 60)
    
    @tilelang.jit(target="npuir")
    def matmul_dynamic(block_M=128, block_N=256, K_L1=16, dtype="float16", accum_dtype="float32"):
        M = T.symbolic("M")
        N = T.symbolic("N")
        K = T.symbolic("K")

        @T.prim_func
        def main(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype)
        ):
            with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
                with T.Scope("Cube"):
                    bx = cid // T.ceildiv(N, block_N) * block_M
                    by = cid % T.ceildiv(N, block_N) * block_N
                    A_BUF = T.alloc_L1([block_M, K_L1], dtype)
                    B_BUF = T.alloc_L1([K_L1, block_N], dtype)
                    C_BUF = T.alloc_L0C([block_M, block_N], accum_dtype)

                    remain_M = T.min(M - bx, block_M)
                    remain_N = T.min(N - by, block_N)

                    for i in T.serial(T.ceildiv(K, K_L1)):
                        remain_K = T.min(K - i * K_L1, K_L1)
                        T.load_nd2nz(A[bx, i * K_L1], A_BUF, [remain_M, remain_K])
                        T.load_nd2nz(B[i * K_L1, by], B_BUF, [remain_K, remain_N])

                        if i == 0:
                            T.gemm(A_BUF, B_BUF, C_BUF, initC=True, b_transpose=False,
                                size=[remain_M, remain_K, remain_N])
                        else:
                            T.gemm(A_BUF, B_BUF, C_BUF, initC=False, b_transpose=False,
                                size=[remain_M, remain_K, remain_N])

                        T.store_fixpipe(C_BUF, C[bx, by],
                            size=[remain_M, remain_N], enable_nz2nd=True)

        return main
    
    print("正在编译...")
    kernel = matmul_dynamic()
    
    print(f"Kernel 编译完成")
    print(f"  - symbolic: {kernel.symbolic}")
    print(f"  - param_info: {kernel.param_info}")
    print(f"  - out_idx: {kernel.out_idx}")
    
    print("\n测试运行...")
    test_cases = [
        (1024, 512, 2048),
        (512, 1024, 512),
    ]
    
    for M, N, K in test_cases:
        print(f"  测试 shape: M={M}, N={N}, K={K}")
        a = torch.randn(M, K, dtype=torch.float16, device="npu")
        b = torch.randn(K, N, dtype=torch.float16, device="npu")
        c = torch.randn(M, N, dtype=torch.float16, device="npu")
        
        kernel(a, b, c)
        
        ref = a @ b
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print(f"    ✓ 验证通过")
    
    print("✓ 动态 GEMM 内核测试通过")
    return kernel


def _convert_symbolic_to_pure_python(symbolic):
    """将 symbolic 中的 TVM tir.Var 转换为纯字符串 key"""
    result = {}
    for key, value in symbolic.items():
        if hasattr(key, 'name'):
            result[key.name] = value
        else:
            result[str(key)] = value
    return result


def _convert_shape_to_pure_python(shape):
    """将 shape 中的 TVM tir.Var 转换为字符串或整数"""
    result = []
    for dim in shape:
        if hasattr(dim, 'name'):
            result.append(dim.name)
        elif hasattr(dim, 'value'):
            result.append(int(dim.value))
        elif isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, str):
            result.append(dim)
        else:
            try:
                result.append(int(dim))
            except (ValueError, TypeError):
                result.append(str(dim))
    return result


def _convert_param_info_to_pure_python(param_info):
    """将 param_info 中的 TVM 对象转换为纯 Python 对象"""
    result = []
    for info in param_info:
        new_info = {
            'dtype': info['dtype'],
            'shape': _convert_shape_to_pure_python(info['shape']),
            'is_output': info['is_output'],
        }
        result.append(new_info)
    return result


def save_kernel(kernel, name: str):
    """保存内核到 kernels 目录"""
    kernel_dir = KERNELS_DIR / name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存内核到: {kernel_dir}")
    
    symbolic_pure = _convert_symbolic_to_pure_python(kernel.symbolic)
    param_info_pure = _convert_param_info_to_pure_python(kernel.param_info)
    
    metadata = {
        "symbolic": symbolic_pure,
        "out_idx": kernel.out_idx,
        "param_info": param_info_pure,
        "signature": kernel.signature,
        "shared": kernel.utils_shared,
        "kernel_name": kernel.kernel_name,
        "gridfunc": kernel.gridfunc,
        "mix_mode": kernel.mix_mode,
        "name": kernel.utils_name,
        "tensor_kinds": kernel.tensor_kinds,
        "kernel_src": kernel.utils_kernel_src,
    }
    
    # 保存 metadata
    metadata_path = kernel_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        cloudpickle.dump(metadata, f)
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
    
    # 打印 metadata 内容
    print(f"\n保存的 metadata 内容:")
    print(f"  - symbolic: {metadata['symbolic']}")
    print(f"  - param_info 长度: {len(metadata['param_info'])}")
    for i, info in enumerate(metadata['param_info']):
        print(f"    [{i}] dtype={info['dtype']}, shape={info['shape']}, is_output={info['is_output']}")


def main():
    print("TileLang Ascend Operators - 预编译脚本")
    print("=" * 60)
    
    # 清理旧内核
    if KERNELS_DIR.exists():
        shutil.rmtree(KERNELS_DIR)
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 编译 Flash Attention
    kernel_fa = compile_flash_attention_kernel()
    save_kernel(kernel_fa, "flash_attention")
    
    # 编译动态 GEMM
    kernel_gemm = compile_dynamic_gemm_kernel()
    save_kernel(kernel_gemm, "gemm")
    
    print("\n" + "=" * 60)
    print("✓ 所有内核预编译完成！")
    print(f"内核目录: {KERNELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
