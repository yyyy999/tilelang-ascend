"""
TileLang Ascend Operators Package

提供 PyTorch 算子接口，支持离线安装即用。

依赖：
- torch >= 2.0.0
- torch_npu
- cloudpickle

使用方式：
    import tilelang_ascend_ops
    
    # Flash Attention
    output = tilelang_ascend_ops.flash_attention(q, k, v)
    
    # GEMM
    c = tilelang_ascend_ops.gemm(a, b)
    
    # 或使用 torch.ops 接口
    output = torch.ops.tilelang_ascend.flash_attention(q, k, v, scale)
    torch.ops.tilelang_ascend.gemm(a, b, c)
"""

from .registry import OpRegistry, register_all_ops
from .loader import KernelRegistry

__version__ = "0.1.0"

_registry = register_all_ops()

from .ops.flash_attention import flash_attention_op
from .ops.gemm import gemm_op

flash_attention = flash_attention_op.python_api
gemm = gemm_op.python_api

__all__ = [
    "flash_attention",
    "gemm",
    "OpRegistry",
    "KernelRegistry",
    "flash_attention_op",
    "gemm_op",
]
