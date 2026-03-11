"""
验证动态shape kernel的离线加载功能

验证步骤：
1. 编译一个使用T.symbolic()的动态shape kernel
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

CACHE_DIR = os.path.join(os.path.dirname(__file__), "test_cache")

def cleanup_cache():
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    os.makedirs(CACHE_DIR)

def compile_dynamic_kernel():
    """编译动态shape的kernel"""
    print("=" * 50)
    print("Step 1: 编译动态shape kernel")
    print("=" * 50)
    
    @tilelang.jit(target="npuir")
    def vec_add_dynamic(block_M=128, block_N=128, dtype="float16"):
        M = T.symbolic("M")
        N = T.symbolic("N")
        
        @T.prim_func
        def main(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype)
        ):
            with T.Kernel(T.ceildiv(M, block_M) * T.ceildiv(N, block_N), is_npu=True) as (cid, _):
                A_VEC = T.alloc_ub((block_M, block_N), dtype)
                B_VEC = T.alloc_ub((block_M, block_N), dtype)
                C_VEC = T.alloc_ub((block_M, block_N), dtype)
                
                m_num = T.ceildiv(M, block_M)
                n_num = T.ceildiv(N, block_N)
                
                start_block_id = cid * T.ceildiv(m_num * n_num, 1)
                for i in T.serial(T.ceildiv(m_num * n_num, 1)):
                    block_id = start_block_id + i
                    if block_id < m_num * n_num:
                        block_id_m = block_id // n_num
                        block_id_n = block_id % n_num
                        bx = block_id_m * block_M
                        by = block_id_n * block_N
                        remain_M = T.min(M - bx, block_M)
                        remain_N = T.min(N - by, block_N)
                        T.copy(A[bx:bx+remain_M, by:by+remain_N], A_VEC)
                        T.copy(B[bx:bx+remain_M, by:by+remain_N], B_VEC)
                        T.vadd(A_VEC, B_VEC, C_VEC)
                        T.copy(C_VEC, C[bx:bx+remain_M, by:by+remain_N])
        
        return main
    
    kernel = vec_add_dynamic()
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
    print(f"  - param_info: {metadata['param_info']}")

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
    print(f"  - param_info: {metadata['param_info']}")
    
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
    print(f"  - param_info: {kernel.param_info}")
    
    return kernel

def test_dynamic_shapes(kernel):
    """测试不同shape的调用"""
    print("\n" + "=" * 50)
    print("Step 4: 测试不同shape的调用")
    print("=" * 50)
    
    test_cases = [
        (256, 256),
        (512, 512),
        (1024, 256),
        (256, 1024),
    ]
    
    for M, N in test_cases:
        print(f"\n测试 shape: ({M}, {N})")
        
        a = torch.randn(M, N, dtype=torch.float16, device="npu")
        b = torch.randn(M, N, dtype=torch.float16, device="npu")
        
        result = kernel(a, b)
        
        expected = a + b
        
        try:
            torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)
            print(f"  ✓ 验证通过")
        except Exception as e:
            print(f"  ✗ 验证失败: {e}")
            return False
    
    return True

def main():
    print("动态shape kernel离线加载验证")
    print("=" * 50)
    
    cleanup_cache()
    
    try:
        kernel = compile_dynamic_kernel()
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
