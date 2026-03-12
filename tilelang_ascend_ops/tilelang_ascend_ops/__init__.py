"""
TileLang Ascend Operators Package

提供 PyTorch 算子接口，支持离线安装即用。

依赖：
- torch >= 2.0.0
- torch_npu
- cloudpickle
"""

import torch
from .loader import KernelRegistry

__version__ = "0.1.0"
__all__ = ["flash_attention", "KernelRegistry"]


_flash_attention_kernel = None


def _get_flash_attention_kernel():
    """获取 Flash Attention 内核"""
    global _flash_attention_kernel
    if _flash_attention_kernel is None:
        _flash_attention_kernel = KernelRegistry.get_kernel("flash_attention")
    return _flash_attention_kernel


def _flash_attention_impl(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float) -> torch.Tensor:
    """Flash Attention 算子实现"""
    kernel = _get_flash_attention_kernel()
    output = kernel(Q, K, V)
    return output


lib = torch.library.Library("tilelang_ascend", "DEF")
lib.define("flash_attention(Tensor Q, Tensor K, Tensor V, float scale) -> Tensor")

lib_impl = torch.library.Library("tilelang_ascend", "IMPL")
lib_impl.impl("flash_attention", _flash_attention_impl, "PrivateUse1")


def flash_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float = None,
) -> torch.Tensor:
    """
    Flash Attention 算子
    
    Args:
        Q: Query 张量 [seq_len, dim], NPU tensor
        K: Key 张量 [seq_len, dim], NPU tensor
        V: Value 张量 [seq_len, dim], NPU tensor
        scale: 缩放因子 (默认: 1/sqrt(dim))
    
    Returns:
        Output 张量 [seq_len, dim]
    
    Example:
        >>> import torch
        >>> import tilelang_ascend_ops
        >>>
        >>> q = torch.randn(512, 128, dtype=torch.float16, device="npu")
        >>> k = torch.randn(512, 128, dtype=torch.float16, device="npu")
        >>> v = torch.randn(512, 128, dtype=torch.float16, device="npu")
        >>>
        >>> output = tilelang_ascend_ops.flash_attention(q, k, v)
    """
    if Q.device.type != "npu":
        raise ValueError("Q must be an NPU tensor")
    if K.device.type != "npu":
        raise ValueError("K must be an NPU tensor")
    if V.device.type != "npu":
        raise ValueError("V must be an NPU tensor")
    
    seq_len, dim = Q.shape
    if K.shape != (seq_len, dim) or V.shape != (seq_len, dim):
        raise ValueError(f"Shape mismatch: Q={Q.shape}, K={K.shape}, V={V.shape}")
    
    if scale is None:
        scale = (1.0 / dim) ** 0.5
    
    return torch.ops.tilelang_ascend.flash_attention(Q, K, V, scale)
