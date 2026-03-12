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

import torch
from .loader import KernelRegistry

__version__ = "0.1.0"

# 全局内核注册中心
_kernel_registry = KernelRegistry

# 定义 PyTorch 算子库
_lib_def = torch.library.Library("tilelang_ascend", "DEF")
_lib_impl = torch.library.Library("tilelang_ascend", "IMPL")


# ============== Flash Attention ==============

from .ops.flash_attention import flash_attention_op

_lib_def.define(flash_attention_op.signature)

def _flash_attention_impl(Q, K, V, scale):
    return flash_attention_op.impl(Q, K, V, scale, registry=_kernel_registry)

_lib_impl.impl("flash_attention", _flash_attention_impl, "PrivateUse1")

flash_attention = flash_attention_op.python_api


# ============== GEMM ==============

from .ops.gemm import gemm_op

_lib_def.define(gemm_op.signature)

def _gemm_impl(A, B, C):
    return gemm_op.impl(A, B, C, registry=_kernel_registry)

_lib_impl.impl("gemm", _gemm_impl, "PrivateUse1")

gemm = gemm_op.python_api


# ============== 导出 ==============

__all__ = [
    "flash_attention",
    "gemm",
    "KernelRegistry",
    "flash_attention_op",
    "gemm_op",
]

print(f"✓ tilelang_ascend_ops 加载完成，已注册算子: flash_attention, gemm")
