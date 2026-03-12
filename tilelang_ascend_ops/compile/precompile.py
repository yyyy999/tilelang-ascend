"""
TileLang Ascend Operators - 预编译脚本

编译所有算子内核并保存到 kernels 目录。

Usage:
    python compile/precompile.py              # 编译所有内核
    python compile/precompile.py flash_attention  # 只编译指定内核
"""

import os
import sys
import shutil
from pathlib import Path

import torch
import cloudpickle

torch.npu.set_device(0)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = PROJECT_DIR / "src"
KERNELS_DIR = PACKAGE_DIR / "kernels"

sys.path.insert(0, str(SCRIPT_DIR))
from kernels import KERNEL_REGISTRY


def _convert_symbolic_to_pure_python(symbolic):
    """将 symbolic 中的 TVM tir.Var 转换为纯字符串 key"""
    result = {}
    for key, value in symbolic.items():
        if hasattr(key, 'name'):
            result[key.name] = value
        else:
            result[str(key)] = value
    return result


def _convert_shape_to_pure_python(shape):
    """将 shape 中的 TVM tir.Var 转换为字符串或整数"""
    result = []
    for dim in shape:
        if hasattr(dim, 'name'):
            result.append(dim.name)
        elif hasattr(dim, 'value'):
            result.append(int(dim.value))
        elif isinstance(dim, int):
            result.append(dim)
        elif isinstance(dim, str):
            result.append(dim)
        else:
            try:
                result.append(int(dim))
            except (ValueError, TypeError):
                result.append(str(dim))
    return result


def _convert_param_info_to_pure_python(param_info):
    """将 param_info 中的 TVM 对象转换为纯 Python 对象"""
    result = []
    for info in param_info:
        new_info = {
            'dtype': info['dtype'],
            'shape': _convert_shape_to_pure_python(info['shape']),
            'is_output': info['is_output'],
        }
        result.append(new_info)
    return result


def save_kernel(kernel, name: str):
    """保存内核到 kernels 目录"""
    kernel_dir = KERNELS_DIR / name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存内核到: {kernel_dir}")
    
    symbolic_pure = _convert_symbolic_to_pure_python(kernel.symbolic)
    param_info_pure = _convert_param_info_to_pure_python(kernel.param_info)
    
    metadata = {
        "symbolic": symbolic_pure,
        "out_idx": kernel.out_idx,
        "param_info": param_info_pure,
        "signature": kernel.signature,
        "shared": kernel.utils_shared,
        "kernel_name": kernel.kernel_name,
        "gridfunc": kernel.gridfunc,
        "mix_mode": kernel.mix_mode,
        "name": kernel.utils_name,
        "tensor_kinds": kernel.tensor_kinds,
        "kernel_src": kernel.utils_kernel_src,
    }
    
    metadata_path = kernel_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        cloudpickle.dump(metadata, f)
    print(f"  ✓ 保存 metadata.pkl")
    
    if not hasattr(kernel, 'mod_path') or kernel.mod_path is None:
        print(f"  ⚠ 警告: kernel.mod_path 不存在，跳过 .so 文件复制")
        return
    
    cache_dir = Path(kernel.mod_path).parent
    
    main_so = cache_dir / "main.so"
    if main_so.exists():
        shutil.copy(main_so, kernel_dir / "main.so")
        print(f"  ✓ 保存 main.so")
    else:
        print(f"  ⚠ 警告: {main_so} 不存在")
    
    npu_utils_so = cache_dir / "npu_utils.so"
    if npu_utils_so.exists():
        shutil.copy(npu_utils_so, kernel_dir / "npu_utils.so")
        print(f"  ✓ 保存 npu_utils.so")
    else:
        print(f"  ⚠ 警告: {npu_utils_so} 不存在")


def main():
    print("TileLang Ascend Operators - 预编译脚本")
    print("=" * 60)
    print(f"可用内核: {list(KERNEL_REGISTRY.keys())}")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        kernels_to_compile = sys.argv[1:]
    else:
        kernels_to_compile = list(KERNEL_REGISTRY.keys())
    
    print(f"将编译: {kernels_to_compile}")
    
    if KERNELS_DIR.exists():
        shutil.rmtree(KERNELS_DIR)
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    
    for name in kernels_to_compile:
        if name not in KERNEL_REGISTRY:
            print(f"警告: 未知内核 '{name}'，跳过")
            continue
        
        try:
            compile_func = KERNEL_REGISTRY[name]
            kernel = compile_func()
            save_kernel(kernel, name)
        except Exception as e:
            print(f"错误: 编译 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✓ 预编译完成！")
    print(f"内核目录: {KERNELS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
