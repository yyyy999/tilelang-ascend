"""
TileLang Ascend Operators - 安装配置

纯 Python 包，依赖：
- torch >= 2.0.0
- torch_npu
- cloudpickle
"""

from setuptools import setup, find_packages
from pathlib import Path

PACKAGE_NAME = "tilelang_ascend_ops"
VERSION = "0.1.0"

# 读取 README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="TileLang Ascend Team",
    author_email="tilelang@example.com",
    description="PyTorch operators for TileLang Ascend kernels - 离线安装即用",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tilelang/tilelang-ascend",
    packages=find_packages(),
    package_dir={"": "."},
    package_data={
        "tilelang_ascend_ops": [
            "kernels/**/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "cloudpickle",
    ],
    extras_require={
        "npu": ["torch_npu"],
    },
)
