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
__all__ = ["flash_attention", "gemm", "KernelRegistry"]


# ============== Flash Attention ==============

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


# ============== GEMM ==============

_gemm_kernel = None


def _get_gemm_kernel():
    """获取 GEMM 内核"""
    global _gemm_kernel
    if _gemm_kernel is None:
        _gemm_kernel = KernelRegistry.get_kernel("gemm")
    return _gemm_kernel


def _gemm_impl(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """GEMM 算子实现"""
    kernel = _get_gemm_kernel()
    output = kernel(A, B)
    return output


# ============== 算子注册 ==============

lib = torch.library.Library("tilelang_ascend", "DEF")
lib.define("flash_attention(Tensor Q, Tensor K, Tensor V, float scale) -> Tensor")
lib.define("gemm(Tensor A, Tensor B) -> Tensor")

lib_impl = torch.library.Library("tilelang_ascend", "IMPL")
lib_impl.impl("flash_attention", _flash_attention_impl, "PrivateUse1")
lib_impl.impl("gemm", _gemm_impl, "PrivateUse1")


# ============== Python API ==============

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


def gemm(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    动态shape GEMM 算子 (C = A @ B)
    
    Args:
        A: 输入矩阵 [M, K], NPU tensor, float16
        B: 输入矩阵 [K, N], NPU tensor, float16
    
    Returns:
        输出矩阵 [M, N], float16
    
    Example:
        >>> import torch
        >>> import tilelang_ascend_ops
        >>>
        >>> a = torch.randn(1024, 512, dtype=torch.float16, device="npu")
        >>> b = torch.randn(512, 2048, dtype=torch.float16, device="npu")
        >>>
        >>> c = tilelang_ascend_ops.gemm(a, b)
    """
    if A.device.type != "npu":
        raise ValueError("A must be an NPU tensor")
    if B.device.type != "npu":
        raise ValueError("B must be an NPU tensor")
    
    M, K = A.shape
    K2, N = B.shape
    if K != K2:
        raise ValueError(f"Matrix dimension mismatch: A.shape[1]={K} != B.shape[0]={K2}")
    
    return torch.ops.tilelang_ascend.gemm(A, B)
