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
import pickle
from pathlib import Path
from copy import deepcopy

import torch

torch.npu.set_device(0)

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
PACKAGE_DIR = PROJECT_DIR / "src"
KERNELS_DIR = PACKAGE_DIR / "kernels"

sys.path.insert(0, str(SCRIPT_DIR))
from kernels import KERNEL_REGISTRY


def _to_pure_python(obj):
    """
    递归转换 TVM 类型为纯 Python 类型
    
    保留不涉及 TVM 依赖的类型（如 torch.dtype）
    """
    if obj is None:
        return None
    
    # 基本类型直接返回
    if isinstance(obj, (bool, int, float, str)):
        return obj
    
    # bytes 直接返回
    if isinstance(obj, bytes):
        return obj
    
    # torch.dtype 直接返回（不涉及 TVM 依赖）
    if isinstance(obj, torch.dtype):
        return obj
    
    # TVM IntImm -> int
    if hasattr(obj, 'value') and not isinstance(obj, (bool, int, float, str, torch.dtype)):
        try:
            return int(obj.value)
        except:
            pass
    
    # TVM tir.Var -> str (变量名)
    if hasattr(obj, 'name') and not isinstance(obj, (bool, int, float, str, torch.dtype)):
        try:
            return str(obj.name)
        except:
            pass
    
    # 列表/元组 -> 递归转换
    if isinstance(obj, (list, tuple)):
        return [_to_pure_python(item) for item in obj]
    
    # 字典 -> 递归转换
    if isinstance(obj, dict):
        return {
            _to_pure_python(k): _to_pure_python(v)
            for k, v in obj.items()
        }
    
    # 其他类型尝试转字符串
    try:
        return str(obj)
    except:
        return None


def save_kernel(kernel, name: str):
    """保存内核到 kernels 目录"""
    kernel_dir = KERNELS_DIR / name
    kernel_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存内核到: {kernel_dir}")
    
    # 深度转换所有字段为纯 Python 类型
    metadata = {
        "symbolic": _to_pure_python(kernel.symbolic),
        "out_idx": _to_pure_python(kernel.out_idx),
        "param_info": _to_pure_python(kernel.param_info),
        "signature": _to_pure_python(kernel.signature),
        "shared": _to_pure_python(kernel.utils_shared),
        "kernel_name": _to_pure_python(kernel.kernel_name),
        "gridfunc": _to_pure_python(kernel.gridfunc),
        "mix_mode": _to_pure_python(kernel.mix_mode),
        "name": _to_pure_python(kernel.utils_name),
        "tensor_kinds": _to_pure_python(kernel.tensor_kinds),
        "kernel_src": _to_pure_python(kernel.utils_kernel_src),
    }
    
    # 打印转换后的内容用于调试
    print(f"  symbolic: {metadata['symbolic']}")
    print(f"  out_idx: {metadata['out_idx']}")
    print(f"  param_info: {metadata['param_info']}")
    
    metadata_path = kernel_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"  ✓ 保存 metadata.pkl (使用标准 pickle)")
    
    # .so 文件在当前工作目录
    cwd = Path(os.getcwd())
    
    # 复制 main.so (启动器)
    launcher_so_path = cwd / kernel.so_launcher_path
    
    if launcher_so_path.exists():
        shutil.copy(launcher_so_path, kernel_dir / "main.so")
        print(f"  ✓ 保存 main.so (from {launcher_so_path})")
    else:
        print(f"  ✗ 错误: 找不到 {launcher_so_path}")
        print(f"    当前目录 .so 文件: {list(cwd.glob('*.so'))}")
    
    # 复制 npu_utils.so (工具库)
    utils_so_path = cwd / kernel.so_utils_path
    
    if utils_so_path.exists():
        shutil.copy(utils_so_path, kernel_dir / "npu_utils.so")
        print(f"  ✓ 保存 npu_utils.so (from {utils_so_path})")
    else:
        print(f"  ✗ 错误: 找不到 {utils_so_path}")
        print(f"    当前目录文件: {list(cwd.glob('*.so'))}")


def main():
    print("TileLang Ascend Operators - 预编译脚本")
    print("=" * 60)
    print(f"可用内核: {list(KERNEL_REGISTRY.keys())}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"内核输出目录: {KERNELS_DIR}")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        kernels_to_compile = sys.argv[1:]
    else:
        kernels_to_compile = list(KERNEL_REGISTRY.keys())
    
    print(f"将编译: {kernels_to_compile}")
    
    if KERNELS_DIR.exists():
        shutil.rmtree(KERNELS_DIR)
    KERNELS_DIR.mkdir(parents=True, exist_ok=True)
    
    success_kernels = []
    for name in kernels_to_compile:
        if name not in KERNEL_REGISTRY:
            print(f"警告: 未知内核 '{name}'，跳过")
            continue
        
        try:
            compile_func = KERNEL_REGISTRY[name]
            kernel = compile_func()
            save_kernel(kernel, name)
            success_kernels.append(name)
        except Exception as e:
            print(f"错误: 编译 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"✓ 预编译完成！成功: {success_kernels}")
    print(f"内核目录: {KERNELS_DIR}")
    
    # 列出生成的文件
    for name in success_kernels:
        kernel_dir = KERNELS_DIR / name
        files = list(kernel_dir.glob("*"))
        print(f"  {name}/: {[f.name for f in files]}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
