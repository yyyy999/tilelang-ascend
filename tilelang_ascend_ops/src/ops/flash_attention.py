"""
Flash Attention 算子定义

固定 shape: seq_len=512, dim=128
"""

from typing import Optional
import torch
from .base import BaseOp


class FlashAttentionOp(BaseOp):
    """Flash Attention 算子"""
    
    _kernel = None
    
    @property
    def name(self) -> str:
        return "flash_attention"
    
    @property
    def signature(self) -> str:
        return "flash_attention(Tensor Q, Tensor K, Tensor V, float scale) -> Tensor"
    
    def get_kernel(self, registry):
        if FlashAttentionOp._kernel is None:
            FlashAttentionOp._kernel = registry.get_kernel(self.name)
        return FlashAttentionOp._kernel
    
    def impl(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, scale: float, registry=None) -> torch.Tensor:
        kernel = self.get_kernel(registry)
        output = kernel(Q, K, V)
        return output
    
    def python_api(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Flash Attention 算子
        
        Args:
            Q: Query 张量 [seq_len, dim], NPU tensor, float16
            K: Key 张量 [seq_len, dim], NPU tensor, float16
            V: Value 张量 [seq_len, dim], NPU tensor, float16
            scale: 缩放因子 (默认: 1/sqrt(dim))
        
        Returns:
            Output 张量 [seq_len, dim]
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


flash_attention_op = FlashAttentionOp()
