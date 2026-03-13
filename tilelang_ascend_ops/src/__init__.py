"""
TileLang Ascend Operators Package

提供 PyTorch 算子接口，支持离线安装即用。

依赖：
- torch >= 2.0.0
- torch_npu

使用方式：
    import tilelang_ascend_ops
    
    # 方式 1: 通过包调用
    output = tilelang_ascend_ops.flash_attention(q, k, v)
    c = tilelang_ascend_ops.gemm(a, b)
    
    # 方式 2: 通过 torch_npu 调用
    import torch_npu
    output = torch_npu.flash_attention(q, k, v)
    c = torch_npu.gemm(a, b)
    
    # 方式 3: 通过 torch.ops 调用
    output = torch.ops.tilelang_ascend.flash_attention(q, k, v, scale)
    torch.ops.tilelang_ascend.gemm(a, b, c)
"""

from .loader import KernelRegistry
from .registry import register_all_ops, get_kernel_registry

__version__ = "0.1.0"

# 注册所有算子
_registered_ops = register_all_ops()

# 导入算子 Python API
from .ops.flash_attention import flash_attention_op
from .ops.gemm import gemm_op

flash_attention = flash_attention_op.python_api
gemm = gemm_op.python_api

# 注入到 torch_npu 模块
def _inject_to_torch_npu():
    """将算子接口注入到 torch_npu 模块"""
    try:
        import torch_npu
        
        # 动态注入所有已注册的算子
        injected_ops = []
        for op_name, op in _registered_ops.items():
            op_func = getattr(op, 'python_api', None)
            if op_func is not None:
                setattr(torch_npu, op_name, op_func)
                injected_ops.append(op_name)
        
        torch_npu.KernelRegistry = KernelRegistry
        print(f"✓ 已注入算子到 torch_npu: {', '.join(injected_ops)}")
    except ImportError:
        print(f"⚠ torch_npu 未安装，跳过注入")

_inject_to_torch_npu()

__all__ = [
    "flash_attention",
    "gemm",
    "KernelRegistry",
    "flash_attention_op",
    "gemm_op",
    "register_all_ops",
    "get_kernel_registry",
]

print(f"✓ tilelang_ascend_ops 加载完成")
