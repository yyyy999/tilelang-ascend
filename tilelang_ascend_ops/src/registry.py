"""
算子注册中心

管理所有算子的注册和内核加载。
"""

from typing import Dict, List, Type
import torch
from .loader import KernelRegistry
from .ops.base import BaseOp


class OpRegistry:
    """算子注册中心"""
    
    _instance = None
    _ops: Dict[str, BaseOp] = {}
    _lib_def = None
    _lib_impl = None
    _kernel_registry = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register_op(cls, op: BaseOp):
        """注册算子"""
        cls._ops[op.name] = op
    
    @classmethod
    def get_op(cls, name: str) -> BaseOp:
        """获取算子"""
        if name not in cls._ops:
            raise KeyError(f"Operator '{name}' not found. Available: {list(cls._ops.keys())}")
        return cls._ops[name]
    
    @classmethod
    def list_ops(cls) -> List[str]:
        """列出所有已注册的算子"""
        return list(cls._ops.keys())
    
    @classmethod
    def initialize(cls):
        """初始化注册中心，注册所有算子到 PyTorch"""
        if cls._lib_def is not None:
            return
        
        cls._lib_def = torch.library.Library("tilelang_ascend", "DEF")
        cls._lib_impl = torch.library.Library("tilelang_ascend", "IMPL")
        cls._kernel_registry = KernelRegistry
        
        for name, op in cls._ops.items():
            def make_impl(op_instance):
                def impl(*args, **kwargs):
                    return op_instance.impl(*args, **kwargs, registry=cls._kernel_registry)
                return impl
            
            cls._lib_def.define(op.signature)
            cls._lib_impl.impl(op.name, make_impl(op), "PrivateUse1")
    
    @classmethod
    def get_kernel_registry(cls):
        """获取内核注册中心"""
        return cls._kernel_registry


def register_all_ops():
    """注册所有算子"""
    from .ops import FlashAttentionOp, GemmOp
    
    registry = OpRegistry()
    registry.register_op(FlashAttentionOp())
    registry.register_op(GemmOp())
    registry.initialize()
    
    return registry
