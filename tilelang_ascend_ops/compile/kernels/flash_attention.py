"""
Flash Attention 内核定义

固定 shape: seq_len=512, dim=128
"""

import torch
import tilelang
import tilelang.language as T


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
        def main(Q, K, V, Output):
            with T.Kernel(T.ceildiv(seq_len, block_m), block_n, is_npu=True) as (bx, by):
                Q_BUF = T.alloc_L1([block_m, dim], dtype)
                K_BUF = T.alloc_L1([block_n, dim], dtype)
                V_BUF = T.alloc_L1([block_n, dim], dtype)
                O_BUF = T.alloc_L1([block_m, dim], dtype)
                acc_s = T.alloc_L0C([block_m, block_n], accum_dtype)
                acc_o = T.alloc_L0C([block_m, dim], accum_dtype)
                
                T.load_nd2nz(Q[bx * block_m, 0], Q_BUF, [block_m, dim])
                
                for i in T.serial(T.ceildiv(seq_len, block_n)):
                    T.load_nd2nz(K[i * block_n, 0], K_BUF, [block_n, dim])
                    T.load_nd2nz(V[i * block_n, 0], V_BUF, [block_n, dim])
                    
                    T.gemm(Q_BUF, K_BUF, acc_s, initC=True, b_transpose=True)
                    
                    for j in T.serial(block_m):
                        for k in T.serial(block_n):
                            acc_s[j, k] = T.exp(acc_s[j, k] * (1.0 / (dim ** 0.5)))
                    
                    T.gemm(acc_s, V_BUF, acc_o, initC=False)
                
                T.store_fixpipe(acc_o, Output[bx * block_m, 0], enable_nz2nd=True)
        
        return main
    
    print("正在编译...")
    kernel = flash_attention_kernel()
    
    print(f"Kernel 编译完成")
    print(f"  - symbolic: {kernel.symbolic}")
    print(f"  - param_info: {kernel.param_info}")
    print(f"  - out_idx: {kernel.out_idx}")
    
    print("\n测试运行...")
    q = torch.randn(seq_len, dim, dtype=torch.float16, device="npu")
    k = torch.randn(seq_len, dim, dtype=torch.float16, device="npu")
    v = torch.randn(seq_len, dim, dtype=torch.float16, device="npu")
    
    output = kernel(q, k, v)
    
    scale = (1.0 / dim) ** 0.5
    ref = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1
    ).to(torch.float16) @ v
    
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("✓ 验证通过")
    
    return kernel
