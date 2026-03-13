"""
动态shape GEMM 算子定义

支持任意 M, N, K 维度的矩阵乘法
"""

from typing import Optional
import torch
from .base import BaseOp


class GemmOp(BaseOp):
    """动态shape GEMM 算子"""
    
    _kernel = None
    
    @property
    def name(self) -> str:
        return "gemm"
    
    @property
    def signature(self) -> str:
        return "gemm(Tensor A, Tensor B, Tensor C) -> Tensor"
    
    def get_kernel(self, registry):
        if GemmOp._kernel is None:
            GemmOp._kernel = registry.get_kernel(self.name)
        return GemmOp._kernel
    
    def impl(self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, registry=None) -> torch.Tensor:
        kernel = self.get_kernel(registry)
        kernel(A, B, C)
        return C
    
    def python_api(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        C: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        动态shape GEMM 算子 (C = A @ B)
        
        Args:
            A: 输入矩阵 [M, K], NPU tensor, float16
            B: 输入矩阵 [K, N], NPU tensor, float16
            C: 输出矩阵 [M, N], NPU tensor, float16 (可选，不传则自动分配)
        
        Returns:
            输出矩阵 [M, N], float16
        """
        if A.device.type != "npu":
            raise ValueError("A must be an NPU tensor")
        if B.device.type != "npu":
            raise ValueError("B must be an NPU tensor")
        
        M, K = A.shape
        K2, N = B.shape
        if K != K2:
            raise ValueError(f"Matrix dimension mismatch: A.shape[1]={K} != B.shape[0]={K2}")
        
        if C is None:
            C = torch.randn(M, N, dtype=torch.float16, device=A.device)
        elif C.device.type != "npu":
            raise ValueError("C must be an NPU tensor")
        elif C.shape != (M, N):
            raise ValueError(f"C shape mismatch: expected ({M}, {N}), got {C.shape}")
        
        return torch.ops.tl_ascend_ops.gemm(A, B, C)


gemm_op = GemmOp()
