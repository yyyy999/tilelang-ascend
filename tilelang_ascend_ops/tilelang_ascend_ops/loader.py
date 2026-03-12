"""
TileLang Ascend Operators - 独立内核加载器

不依赖 tilelang 源码，只依赖：
- torch
- torch_npu
- cloudpickle
"""

import os
import math
import torch
import cloudpickle
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def replace_by_longest_key(calculate_str: str, replace_dict: dict) -> str:
    """替换字符串中的变量名"""
    sorted_keys = sorted(replace_dict.keys(), key=lambda x: (-len(x), x))
    result = calculate_str
    for key in sorted_keys:
        result = result.replace(key, str(replace_dict[key]))
    return result


class NPUKernelLoader:
    """
    独立的 NPU 内核加载器
    
    从预编译的文件加载内核：
    - metadata.pkl: 内核元数据
    - main.so: 启动器
    - npu_utils.so: 工具库
    - kernel.o: 内核二进制（可选，嵌入在 metadata 中）
    """
    
    def __init__(self, kernel_dir: str):
        self.kernel_dir = Path(kernel_dir)
        
        # 加载 metadata
        metadata_path = self.kernel_dir / "metadata.pkl"
        with open(metadata_path, "rb") as f:
            self.metadata = cloudpickle.load(f)
        
        # 提取元数据字段
        self.signature = self.metadata.get("signature", {})
        self.out_idx = self.metadata.get("out_idx", [-1])
        if isinstance(self.out_idx, int):
            self.out_idx = [self.out_idx]
        self.param_info = self.metadata.get("param_info", [])
        self.symbolic = self.metadata.get("symbolic", {})
        self.gridfunc = self.metadata.get("gridfunc", "")
        self.kernel_src = self.metadata.get("kernel_src", b"")
        self.kernel_name = self.metadata.get("name", "kernel")
        self.tensor_kinds = self.metadata.get("tensor_kinds", [])
        self.shared = self.metadata.get("shared", 1)
        self.mix_mode = self.metadata.get("mix_mode", False)
        
        # 设备信息
        self.device = torch.npu.current_device()
        self.stream = torch.npu.current_stream(self.device).npu_stream
        
        # 加载 main.so (启动器)
        self._load_launcher()
        
        # 加载 npu_utils.so (工具库)
        self._load_npu_utils()
        
        # 加载内核二进制
        self._load_kernel_binary()
    
    def _load_launcher(self):
        """加载 main.so 启动器"""
        launcher_path = self.kernel_dir / "main.so"
        spec = importlib.util.spec_from_file_location(
            "__tilelang_launcher", str(launcher_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.launch = getattr(mod, "launch")
    
    def _load_npu_utils(self):
        """加载 npu_utils.so 工具库"""
        utils_path = self.kernel_dir / "npu_utils.so"
        spec = importlib.util.spec_from_file_location(
            "npu_utils", str(utils_path)
        )
        self.npu_utils = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.npu_utils)
    
    def _load_kernel_binary(self):
        """加载内核二进制到设备"""
        kernel_mode = "aicore" if not self.mix_mode else "aiv"
        
        result = self.npu_utils.load_kernel_binary(
            self.kernel_name,
            self.kernel_src,
            len(self.kernel_src),
            0,  # shared (unused)
            self.device,
            kernel_mode,
        )
        
        self.t_module, self.t_function, self.t_n_regs, self.t_n_spills = result
    
    def _calc_grid(self, orig_to_input: dict, *args):
        """计算 grid 维度和动态值"""
        dynamic_val = {}
        extra_args = []
        
        for key, pos in self.symbolic.items():
            if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                tensor_idx, dim_idx = pos[0], pos[1]
                if tensor_idx in orig_to_input:
                    arg_pos = orig_to_input[tensor_idx]
                    arg = args[arg_pos]
                    if isinstance(arg, torch.Tensor) and dim_idx < len(arg.shape):
                        value = arg.shape[dim_idx]
                        dynamic_val[str(key)] = value
                        extra_args.append(value)
                    else:
                        raise ValueError(f"Cannot resolve symbolic {key}")
                else:
                    raise ValueError(f"Symbolic {key} depends on output")
        
        self.extra_args = extra_args
        
        # 计算 grid
        result = replace_by_longest_key(self.gridfunc, dynamic_val)
        
        try:
            if isinstance(result, (int, float)):
                grid_value = result
            elif isinstance(result, str):
                grid_value = eval(
                    result,
                    {"__builtins__": {}},
                    {"math": math, **dynamic_val},
                )
            else:
                grid_value = result
            
            if hasattr(grid_value, "__iter__"):
                self.grid = [int(x) for x in grid_value]
            else:
                self.grid = [int(grid_value), 1, 1]
        except Exception as e:
            raise ValueError(f"Failed to evaluate grid expression '{result}': {e}")
        
        return dynamic_val
    
    def __call__(self, *args) -> Any:
        """执行内核"""
        total_params = len(self.param_info)
        num_inputs = total_params - len(self.out_idx)
        
        if len(args) != num_inputs:
            raise ValueError(f"Expected {num_inputs} inputs, got {len(args)}")
        
        # 构建输入位置映射
        orig_to_input = {}
        input_pos = 0
        for i, info in enumerate(self.param_info):
            if not info["is_output"]:
                orig_to_input[i] = input_pos
                input_pos += 1
        
        # 计算 grid 和动态值
        dynamic_val = self._calc_grid(orig_to_input, *args)
        
        # 构建完整参数列表
        full_args = [None] * total_params
        input_ptr = 0
        
        for i, info in enumerate(self.param_info):
            if info["is_output"]:
                # 输出参数：创建空张量
                dtype = info["dtype"]
                shape = []
                for dim in info["shape"]:
                    if isinstance(dim, str):
                        val = dynamic_val.get(dim)
                        if val is None:
                            raise ValueError(f"Missing value for {dim}")
                        shape.append(val)
                    else:
                        shape.append(int(dim))
                
                device = args[0].device if args else torch.device("npu")
                full_args[i] = torch.empty(shape, dtype=dtype, device=device)
            else:
                # 输入参数
                full_args[i] = args[input_ptr]
                input_ptr += 1
        
        # 添加额外参数
        full_args.extend(self.extra_args)
        
        # 执行内核
        self.launch(
            self.grid[0],
            self.grid[1],
            self.grid[2],
            self.stream,
            self.t_function,
            {"kernel_name": self.kernel_name, "tensor_kinds": self.tensor_kinds},
            {},
            None,  # enter_hook
            None,  # exit_hook
            *full_args
        )
        
        # 返回结果
        if len(self.out_idx) == 1:
            return full_args[self.out_idx[0]]
        else:
            return [full_args[i] for i in self.out_idx]


class KernelRegistry:
    """内核注册表"""
    
    _kernels: Dict[str, NPUKernelLoader] = {}
    _kernel_dir: Optional[Path] = None
    
    @classmethod
    def set_kernel_dir(cls, kernel_dir: str):
        """设置内核目录"""
        cls._kernel_dir = Path(kernel_dir)
    
    @classmethod
    def get_kernel(cls, name: str) -> NPUKernelLoader:
        """获取内核（带缓存）"""
        if name not in cls._kernels:
            if cls._kernel_dir is None:
                # 默认使用包内的 kernels 目录
                cls._kernel_dir = Path(__file__).parent / "kernels"
            
            kernel_path = cls._kernel_dir / name
            if not kernel_path.exists():
                raise ValueError(f"Kernel '{name}' not found at {kernel_path}")
            
            cls._kernels[name] = NPUKernelLoader(str(kernel_path))
        
        return cls._kernels[name]
