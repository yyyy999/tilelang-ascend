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
    
    # 测试不同 shape
    test_cases = [
        (512, 128),
        (1024, 128),
        (512, 256),
    ]
    
    for seq_len, dim in test_cases:
        print(f"\n测试 shape: ({seq_len}, {dim})")
        
        q = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
        k = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
        v = torch.randn(seq_len, dim, dtype=torch.float16, device="npu:0")
        
        # 使用 torch.ops 调用
        scale = 1.0 / (dim ** 0.5)
        output = torch.ops.tilelang_ascend.flash_attention(q, k, v, scale)
        
        # 验证结果
        ref = torch.nn.functional.softmax(
            (q @ k.T).to(torch.float32) * scale, dim=-1
        ).to(torch.float16) @ v
        
        torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
        print(f"  ✓ 验证通过")
    
    print("\n" + "=" * 60)
    print("✓ 所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_flash_attention()
