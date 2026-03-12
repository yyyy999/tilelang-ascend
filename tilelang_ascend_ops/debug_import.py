#!/usr/bin/env python3
"""
调试导入问题
"""

import sys
import os

print("=" * 60)
print("调试 tilelang_ascend_ops 导入问题")
print("=" * 60)

print(f"\n当前工作目录: {os.getcwd()}")
print(f"Python 路径:")
for p in sys.path[:5]:
    print(f"  - {p}")

print(f"\n尝试导入 tilelang_ascend_ops...")

try:
    import tilelang_ascend_ops
    
    print(f"✓ 导入成功")
    print(f"模块路径: {tilelang_ascend_ops.__file__}")
    print(f"模块属性: {dir(tilelang_ascend_ops)}")
    
    print(f"\n检查关键属性:")
    print(f"  - __all__: {getattr(tilelang_ascend_ops, '__all__', '不存在')}")
    print(f"  - flash_attention: {hasattr(tilelang_ascend_ops, 'flash_attention')}")
    print(f"  - gemm: {hasattr(tilelang_ascend_ops, 'gemm')}")
    print(f"  - __version__: {getattr(tilelang_ascend_ops, '__version__', '不存在')}")
    
    if hasattr(tilelang_ascend_ops, 'flash_attention'):
        print(f"\n✓ flash_attention 存在")
        print(f"  类型: {type(tilelang_ascend_ops.flash_attention)}")
    else:
        print(f"\n✗ flash_attention 不存在")
        
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
