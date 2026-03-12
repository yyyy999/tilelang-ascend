# TileLang Ascend Operators

将 TileLang Ascend 算子注册到 PyTorch，支持 **离线安装即用**。

## 特点

- ✅ **离线安装即用** - 无需编译器，安装后直接使用
- ✅ **独立运行时** - 不依赖 tilelang 源码
- ✅ **动态 Shape** - 使用 `T.symbolic()` 支持多种 shape
- ✅ **PyTorch 集成** - 支持 `torch.ops.tilelang_ascend.xxx` 调用

## 安装

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- torch_npu
- cloudpickle

### 安装步骤

```bash
cd tilelang_ascend_ops
pip install .
```

## 使用方法

### 方式一：直接调用函数

```python
import torch
import tilelang_ascend_ops

q = torch.randn(512, 128, dtype=torch.float16).npu()
k = torch.randn(512, 128, dtype=torch.float16).npu()
v = torch.randn(512, 128, dtype=torch.float16).npu()

output = tilelang_ascend_ops.flash_attention(q, k, v)
```

### 方式二：使用 torch.ops 接口

```python
import torch
import tilelang_ascend_ops

q = torch.randn(512, 128, dtype=torch.float16).npu()
k = torch.randn(512, 128, dtype=torch.float16).npu()
v = torch.randn(512, 128, dtype=torch.float16).npu()

scale = 1.0 / (128 ** 0.5)
output = torch.ops.tilelang_ascend.flash_attention(q, k, v, scale)
```

## 开发指南

### 预编译内核

在开发机上运行预编译脚本：

```bash
cd tilelang_ascend_ops
python scripts/precompile.py
```

预编译产物会保存到 `tilelang_ascend_ops/kernels/` 目录。

### 打包发布

```bash
python -m build --wheel
```

### 目录结构

```
tilelang_ascend_ops/
├── tilelang_ascend_ops/
│   ├── __init__.py       # 算子注册
│   ├── loader.py         # 独立内核加载器
│   └── kernels/          # 预编译内核
│       └── flash_attention/
│           ├── metadata.pkl    # 内核元数据
│           ├── main.so         # 启动器
│           └── npu_utils.so    # 工具库
├── scripts/
│   └── precompile.py     # 预编译脚本
├── setup.py
├── test_install.py
└── README.md
```

## 技术实现

### 独立加载器

`loader.py` 实现了独立的内核加载器，不依赖 tilelang 源码：

```python
from tilelang_ascend_ops.loader import KernelRegistry

# 加载预编译内核
kernel = KernelRegistry.get_kernel("flash_attention")

# 执行内核
output = kernel(q, k, v)
```

### 内核文件

| 文件 | 说明 |
|------|------|
| `metadata.pkl` | 内核元数据（参数信息、shape、grid 等） |
| `main.so` | 启动器，包含 `launch` 函数 |
| `npu_utils.so` | 工具库，包含内核加载函数 |

### 依赖关系

| 依赖 | 编译时 | 运行时 |
|------|--------|--------|
| tilelang | ✅ | ❌ |
| Bisheng 编译器 | ✅ | ❌ |
| g++ | ✅ | ❌ |
| torch | ✅ | ✅ |
| torch_npu | ✅ | ✅ |
| cloudpickle | ✅ | ✅ |

## License

MIT License
