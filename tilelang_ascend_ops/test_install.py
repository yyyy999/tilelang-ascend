#!/usr/bin/env python3
"""
测试 tilelang_ascend_ops 安装

Usage:
    python test_install.py
"""

import torch
import tilelang_ascend_ops


def test_flash_attention():
    """测试 Flash Attention 算子"""
    print("=" * 60)
    print("测试 Flash Attention 算子")
    print("=" * 60)
    
    seq_len = 512
    dim = 128
    
    print(f"\n测试 shape: seq_len={seq_len}, dim={dim}")
    
    q = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
    k = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
    v = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
    
    scale = (1.0 / dim) ** 0.5
    
    output = tilelang_ascend_ops.flash_attention(q, k, v, scale)
    
    ref = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1
    ).to(torch.float16) @ v
    
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ 函数接口验证通过")
    
    output2 = torch.ops.tilelang_ascend.flash_attention(q, k, v, scale)
    torch.testing.assert_close(output2, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ torch.ops 接口验证通过")


def test_gemm():
    """测试动态shape GEMM 算子"""
    print("\n" + "=" * 60)
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
        
        c = tilelang_ascend_ops.gemm(a, b)
        
        ref = a @ b
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ 函数接口验证通过")
        
        c2 = torch.zeros(M, N, dtype=torch.float16, device="npu:0")
        torch.ops.tilelang_ascend.gemm(a, b, c2)
        torch.testing.assert_close(c2, ref, rtol=1e-2, atol=1e-2)
        print("  ✓ torch.ops 接口验证通过")


if __name__ == "__main__":
    test_flash_attention()
    test_gemm()
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)
