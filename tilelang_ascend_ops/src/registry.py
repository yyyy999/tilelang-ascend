"""
TileLang Ascend Operators - 算子注册中心

管理所有算子的 PyTorch 注册。
"""

import torch
from typing import Dict, List
from .loader import KernelRegistry
from .ops.base import BaseOp

_kernel_registry = KernelRegistry

_lib_def = None
_lib_impl = None


def _ensure_lib_initialized():
    """确保 PyTorch 算子库已初始化"""
    global _lib_def, _lib_impl
    
    if _lib_def is None:
        _lib_def = torch.library.Library("tl_ascend_ops", "DEF")
        _lib_impl = torch.library.Library("tl_ascend_ops", "IMPL")


def _register_op(op: BaseOp):
    """注册单个算子到 PyTorch"""
    _ensure_lib_initialized()
    
    _lib_def.define(op.signature)
    
    def _impl_wrapper(*args, **kwargs):
        return op.impl(*args, **kwargs, registry=_kernel_registry)
    
    _lib_impl.impl(op.name, _impl_wrapper, "PrivateUse1")


def register_all_ops() -> Dict[str, BaseOp]:
    """注册所有算子"""
    from .ops.flash_attention import flash_attention_op
    from .ops.gemm import gemm_op
    
    ops = {
        "flash_attention": flash_attention_op,
        "gemm": gemm_op,
    }
    
    for name, op in ops.items():
        _register_op(op)
    
    print(f"✓ 已注册 {len(ops)} 个算子: {', '.join(ops.keys())}")
    return ops


def get_kernel_registry():
    """获取内核注册中心"""
    return _kernel_registry
