"""
TileLang Ascend Operators - Installation Configuration

Pure Python package, dependencies:
- torch >= 2.0.0
- torch_npu
"""

from setuptools import setup, find_packages
from pathlib import Path

PACKAGE_NAME = "tl_ascend_ops"
VERSION = "0.1.0"

readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author="TileLang Ascend Team",
    author_email="tilelang@example.com",
    description="PyTorch operators for TileLang Ascend kernels - offline installation ready",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tilelang/tilelang-ascend",
    packages=[
        "tl_ascend_ops",
        "tl_ascend_ops.ops",
        "tl_ascend_ops.kernels",
    ],
    package_dir={"tl_ascend_ops": "src"},
    package_data={
        "tl_ascend_ops": ["kernels/**/*", "*.so", "*.pkl"],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "npu": ["torch_npu"],
    },
)
