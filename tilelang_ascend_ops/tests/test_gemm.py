"""
动态shape GEMM 算子测试
"""

import torch
import torch_npu
import tl_ascend_ops


def test_gemm():
    """测试动态shape GEMM 算子"""
    print("=" * 60)
    print("测试动态shape GEMM 算子")
    print("=" * 60)
    
    test_cases = [
        (1024, 512, 2048),
        (512, 1024, 512),
        (256, 256, 256),
    ]
    
    for M, N, K in test_cases:
        print(f"\n测试 shape: M={M}, N={N}, K={K}")
        
        a = torch.randn(M, K, dtype=torch.float16, device="npu:0")
        b = torch.randn(K, N, dtype=torch.float16, device="npu:0")
        ref = a @ b
        
        # 方式 1: 通过包调用
        c = tl_ascend_ops.gemm(a, b)
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ tl_ascend_ops.gemm 验证通过")
        
        # 方式 2: 通过 torch_npu 调用
        c2 = torch_npu.gemm(a, b)
        torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ torch_npu.gemm 验证通过")
        
        # 方式 3: 通过 torch.ops 调用
        c3 = torch.zeros(M, N, dtype=torch.float16, device="npu:0")
        torch.ops.tl_ascend_ops.gemm(a, b, c3)
        torch.testing.assert_close(c3, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ torch.ops.tl_ascend_ops.gemm 验证通过")


if __name__ == "__main__":
    test_gemm()
