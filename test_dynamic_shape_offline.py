"""
验证动态shape kernel的离线加载功能 - 使用GEMM

验证步骤：
1. 编译一个使用T.symbolic()的动态shape GEMM kernel
2. 保存kernel到磁盘
3. 从磁盘加载kernel
4. 用不同的shape调用，验证是否能正常工作
"""
import os
import sys
import shutil
import torch
import tilelang
import tilelang.language as T
from tilelang.jit.jit_npu import JitKernel_NPU
import cloudpickle

torch.npu.set_device(0)
tilelang.cache.clear_cache()

CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_cache")

def cleanup_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)

def compile_dynamic_gemm_kernel():
    """编译动态shape的GEMM kernel"""
    print("=" * 50)
    print("Step 1: 编译动态shape GEMM kernel")
    print("=" * 50)
    
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
    
    kernel = matmul_dynamic()
    print(f"Kernel编译完成")
    print(f"  - symbolic: {kernel.symbolic}")
    print(f"  - param_info: {kernel.param_info}")
    print(f"  - out_idx: {kernel.out_idx}")
    
    return kernel

def save_kernel_to_disk(kernel):
    """保存kernel到磁盘"""
    print("\n" + "=" * 50)
    print("Step 2: 保存kernel到磁盘")
    print("=" * 50)
    
    metadata = {
        "symbolic": kernel.symbolic,
        "params": kernel.params,
        "out_idx": kernel.out_idx,
        "param_info": kernel.param_info,
        "signature": kernel.signature,
        "primfunc": kernel.prim_func,
        "mlir_content": kernel.mlir_content,
        "shared": kernel.utils_shared,
        "kernel_name": kernel.kernel_name,
        "gridfunc": kernel.gridfunc,
        "mix_mode": kernel.mix_mode,
        "name": kernel.utils_name,
        "tensor_kinds": kernel.tensor_kinds,
        "kernel_src": kernel.utils_kernel_src,
    }
    
    metadata_path = os.path.join(CACHE_DIR, "metadata.pkl")
    with open(metadata_path, "wb") as f:
        cloudpickle.dump(metadata, f)
    print(f"metadata保存到: {metadata_path}")
    
    shutil.copy(kernel.so_launcher_path, os.path.join(CACHE_DIR, "main.so"))
    print(f"main.so保存到: {os.path.join(CACHE_DIR, 'main.so')}")
    
    shutil.copy(kernel.so_utils_path, os.path.join(CACHE_DIR, "npu_utils.so"))
    print(f"npu_utils.so保存到: {os.path.join(CACHE_DIR, 'npu_utils.so')}")
    
    print(f"\n保存的metadata内容:")
    print(f"  - symbolic: {metadata['symbolic']}")
    print(f"  - param_info 长度: {len(metadata['param_info'])}")
    for i, info in enumerate(metadata['param_info']):
        print(f"    [{i}] dtype={info['dtype']}, shape={info['shape']}, is_output={info['is_output']}")

def load_kernel_from_disk():
    """从磁盘加载kernel"""
    print("\n" + "=" * 50)
    print("Step 3: 从磁盘加载kernel")
    print("=" * 50)
    
    metadata_path = os.path.join(CACHE_DIR, "metadata.pkl")
    with open(metadata_path, "rb") as f:
        metadata = cloudpickle.load(f)
    
    print(f"加载的metadata内容:")
    print(f"  - symbolic: {metadata['symbolic']}")
    print(f"  - param_info 长度: {len(metadata['param_info'])}")
    for i, info in enumerate(metadata['param_info']):
        print(f"    [{i}] dtype={info['dtype']}, shape={info['shape']}, is_output={info['is_output']}")
    
    kernel = JitKernel_NPU.from_database(
        mod=metadata["primfunc"],
        kernel_source=metadata["kernel_src"],
        kernel_launcher_path=os.path.join(CACHE_DIR, "main.so"),
        kernel_utils_path=os.path.join(CACHE_DIR, "npu_utils.so"),
        metadata=metadata,
        out_idx=metadata["out_idx"],
    )
    
    print(f"\nKernel加载完成")
    print(f"  - symbolic: {kernel.symbolic}")
    print(f"  - param_info 长度: {len(kernel.param_info)}")
    
    return kernel

def test_dynamic_shapes(kernel):
    """测试不同shape的调用"""
    print("\n" + "=" * 50)
    print("Step 4: 测试不同shape的调用")
    print("=" * 50)
    
    test_cases = [
        (1024, 512, 2048),
        (512, 1024, 512),
        (512, 512, 1264),
    ]
    
    for M, N, K in test_cases:
        print(f"\n测试 shape: M={M}, N={N}, K={K}")
        
        a = torch.randn(M, K, dtype=torch.float16, device="npu")
        b = torch.randn(K, N, dtype=torch.float16, device="npu")
        c = torch.randn(M, N, dtype=torch.float16, device="npu")
        
        kernel(a, b, c)
        
        ref_c = a @ b
        
        try:
            torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
            print(f"  ✓ 验证通过")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
            return False
    
    return True

def main():
    print("动态shape GEMM kernel离线加载验证")
    print("=" * 50)
    
    cleanup_cache()
    
    try:
        kernel = compile_dynamic_gemm_kernel()
        save_kernel_to_disk(kernel)
        loaded_kernel = load_kernel_from_disk()
        success = test_dynamic_shapes(loaded_kernel)
        
        if success:
            print("\n" + "=" * 50)
            print("✓ 所有验证通过！动态shape离线加载功能正常")
            print("=" * 50)
        else:
            print("\n" + "=" * 50)
            print("✗ 验证失败")
            print("=" * 50)
            return 1
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        pass
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
