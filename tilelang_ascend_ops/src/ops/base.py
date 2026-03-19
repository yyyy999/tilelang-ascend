"""
Operator Base Class Definition

All operators need to inherit from this base class and implement the required methods.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch


class BaseOp(ABC):
    """Operator base class"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Operator name, used for kernel directory and registration"""
        pass
    
    @property
    @abstractmethod
    def signature(self) -> str:
        """PyTorch operator signature, e.g. 'flash_attention(Tensor Q, Tensor K, Tensor V, float scale) -> Tensor'"""
        pass
    
    @abstractmethod
    def get_kernel(self, registry) -> Any:
        """Get kernel from registry"""
        pass
    
    @abstractmethod
    def impl(self, *args, **kwargs) -> torch.Tensor:
        """Operator implementation, calls kernel to execute computation"""
        pass
    
    @abstractmethod
    def python_api(self, *args, **kwargs) -> torch.Tensor:
        """Python API interface, includes parameter validation and default value handling"""
        pass
    
    def register(self, lib_def, lib_impl, registry, dispatch_key: str = "PrivateUse1"):
        """Register operator to PyTorch"""
        lib_def.define(self.signature)
        lib_impl.impl(self.name, self.impl, dispatch_key)
