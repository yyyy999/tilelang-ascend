"""
简化的算子注册逻辑

直接注册所有算子，避免复杂的单例和包装类。
"""

import torch
from typing import List, Dict
from .loader import KernelRegistry
from .ops.base import BaseOp

# 全局内核注册中心
_kernel_registry = KernelRegistry

# 定义 PyTorch 算子库
_lib_def = None
_lib_impl = None


def _register_op(op: BaseOp):
    """注册单个算子到 PyTorch"""
    global _lib_def, _lib_impl
    
    if _lib_def is None:
        _lib_def = torch.library.Library("tilelang_ascend", "DEF")
        _lib_impl = torch.library.Library("tilelang_ascend", "IMPL")
    
    # 定义算子签名
    _lib_def.define(op.signature)
    
    # 实现算子调用
    def _impl_wrapper(*args, **kwargs):
        return op.impl(*args, **kwargs, registry=_kernel_registry)
    
    # 注册实现
    _lib_impl.impl(op.name, _impl_wrapper, "PrivateUse1")


def register_all_ops():
    """注册所有算子"""
    from .ops.flash_attention import flash_attention_op
    from .ops.gemm import gemm_op
    
    # 注册每个算子
    _register_op(flash_attention_op)
    _register_op(gemm_op)
    
    print(f"✓ 已注册 {2} 个算子: flash_attention, gemm")
    return {"flash_attention": flash_attention_op, "gemm": gemm_op}
