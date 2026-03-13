"""
Flash Attention 算子测试
"""

import torch
import tl_ascend_ops


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
    
    output = tl_ascend_ops.flash_attention(q, k, v, scale)
    
    ref = torch.nn.functional.softmax(
        (q @ k.T).to(torch.float32) * scale, dim=-1
    ).to(torch.float16) @ v
    
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ 函数接口验证通过")
    
    output2 = torch.ops.tilelang_ascend.flash_attention(q, k, v, scale)
    torch.testing.assert_close(output2, ref, rtol=1e-2, atol=1e-2)
    print("  ✓ torch.ops 接口验证通过")


if __name__ == "__main__":
    test_flash_attention()
