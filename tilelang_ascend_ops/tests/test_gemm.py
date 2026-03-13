"""
动态shape GEMM 算子测试
"""

import torch
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
        
        c = tl_ascend_ops.gemm(a, b)
        
        ref = a @ b
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ 函数接口验证通过")
        
        c2 = torch.zeros(M, N, dtype=torch.float16, device="npu:0")
        torch.ops.tilelang_ascend.gemm(a, b, c2)
        torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ torch.ops 接口验证通过")


if __name__ == "__main__":
    test_gemm()
