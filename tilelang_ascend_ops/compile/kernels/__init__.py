"""
TileLang 内核定义目录

每个内核定义在单独的文件中，便于维护和扩展。
"""

from .flash_attention import compile_flash_attention_kernel
from .gemm import compile_gemm_kernel

KERNEL_REGISTRY = {
    "flash_attention": compile_flash_attention_kernel,
    "gemm": compile_gemm_kernel,
}

__all__ = [
    "compile_flash_attention_kernel",
    "compile_gemm_kernel",
    "KERNEL_REGISTRY",
]
