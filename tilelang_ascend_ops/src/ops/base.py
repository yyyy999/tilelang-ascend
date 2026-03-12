"""
算子基类定义

所有算子都需要继承此基类并实现相应方法。
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch


class BaseOp(ABC):
    """算子基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """算子名称，用于内核目录和注册"""
        pass
    
    @property
    @abstractmethod
    def signature(self) -> str:
        """PyTorch 算子签名，如 'flash_attention(Tensor Q, Tensor K, Tensor V, float scale) -> Tensor'"""
        pass
    
    @abstractmethod
    def get_kernel(self, registry) -> Any:
        """从注册中心获取内核"""
        pass
    
    @abstractmethod
    def impl(self, *args, **kwargs) -> torch.Tensor:
        """算子实现，调用内核执行计算"""
        pass
    
    @abstractmethod
    def python_api(self, *args, **kwargs) -> torch.Tensor:
        """Python API 接口，包含参数校验和默认值处理"""
        pass
    
    def register(self, lib_def, lib_impl, registry, dispatch_key: str = "PrivateUse1"):
        """注册算子到 PyTorch"""
        lib_def.define(self.signature)
        lib_impl.impl(self.name, self.impl, dispatch_key)
