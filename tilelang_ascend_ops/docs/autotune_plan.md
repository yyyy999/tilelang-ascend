# Autotune Integration Plan

## Overview

Integrate tilelang's autotune functionality into `tilelang_ascend_ops` to improve kernel performance while maintaining offline deployment capability.

## Key Insights

### 1. Autotune is Explicitly Called

Autotune is NOT automatic - it requires explicit invocation:

```python
# Method 1: Using @autotune decorator (requires @jit)
@tilelang.jit
@tilelang.autotune(configs=my_configs)
def my_kernel(...):
    ...

# Method 2: Explicit AutoTuner call (our approach)
autotuner = AutoTuner.from_kernel(kernel=gemm_kernel, configs=gemm_configs())
result = autotuner.run()  # Explicit call
kernel = result.kernel
```

### 2. Dynamic Shape Handling

TileLang uses **symbolic variables** for dynamic shapes:

- One compiled kernel works for all shapes (M, N, K are symbolic)
- Autotune searches for optimal **tiling configuration**, not shape-specific code
- No need to re-autotune for different shapes

### 3. Offline Deployment Preserved

```
Development Phase (with network, build environment):
  kernel source + config_generator
         ↓
  AutoTuner.run() ← searches optimal config (minutes)
         ↓
  best_kernel → save_kernel() → main.so, npu_utils.so

Deployment Phase (offline):
  load_kernel() → direct execution, no autotune needed
```

---

## Implementation Plan

### Phase 1: Basic Autotune Integration

**Goal**: Add autotune to compilation process, single optimal config for all shapes.

**Files to Modify**:

| File | Changes |
|------|---------|
| `compile/kernels/gemm.py` | Add config generator, refactor to autotune mode |
| `compile/kernels/flash_attention.py` | Add config generator, refactor to autotune mode |
| `compile/precompile.py` | No changes needed |

#### 3.1 `compile/kernels/gemm.py`

**Current Structure**:
```python
def compile_gemm_kernel():
    @tilelang.jit(target="npuir")
    def matmul(block_M=128, block_N=256, K_L1=16, ...):
        @T.prim_func
        def main(...):
            ...
        return main
    
    kernel = matmul()  # Uses default config
    return kernel
```

**New Structure**:
```python
def gemm_configs():
    """Generate configuration search space for GEMM"""
    configs = []
    for block_M in [64, 128]:
        for block_N in [128, 256]:
            for K_L1 in [16, 32]:
                configs.append({
                    "block_M": block_M,
                    "block_N": block_N,
                    "K_L1": K_L1,
                })
    return configs

def gemm_kernel(block_M=128, block_N=256, K_L1=16, dtype="float16", accum_dtype="float32"):
    """GEMM kernel definition with tunable parameters"""
    M = T.symbolic("M")
    N = T.symbolic("N")
    K = T.symbolic("K")
    
    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor((K, N), dtype), C: T.Tensor((M, N), dtype)):
        # ... kernel logic unchanged ...
    return main

def compile_gemm_kernel():
    """Compile GEMM kernel with autotune"""
    from tilelang.autotuner import AutoTuner
    
    autotuner = AutoTuner.from_kernel(
        kernel=gemm_kernel,
        configs=gemm_configs(),
    ).set_compile_args(
        target="npuir",
    ).set_profile_args(
        warmup=10,
        rep=50,
        skip_check=False,  # Verify correctness
    )
    
    result = autotuner.run()
    return result.kernel  # Return best kernel
```

#### 3.2 `compile/kernels/flash_attention.py`

**Current Structure**:
```python
def compile_flash_attention_kernel():
    @tilelang.jit(out_idx=[-1], target="npuir")
    def online_flash_attention(block_M=64, block_N=64, block_K=32, ...):
        @T.prim_func
        def flash_attention(...):
            ...
        return flash_attention
    
    kernel = online_flash_attention()
    return kernel
```

**New Structure**:
```python
def flash_attention_configs():
    """Generate configuration search space for Flash Attention"""
    configs = []
    for block_M in [32, 64]:
        for block_N in [32, 64]:
            for block_K in [16, 32]:
                configs.append({
                    "block_M": block_M,
                    "block_N": block_N,
                    "block_K": block_K,
                })
    return configs

def flash_attention_kernel(block_M=64, block_N=64, block_K=32, dtype="float16", accum_dtype="float32"):
    """Flash Attention kernel definition with tunable parameters"""
    shape_q = [seq_len, dim]
    # ... other definitions ...
    
    @T.prim_func
    def flash_attention(Q, K, V, Output):
        # ... kernel logic unchanged ...
    return flash_attention

def compile_flash_attention_kernel():
    """Compile Flash Attention kernel with autotune"""
    from tilelang.autotuner import AutoTuner
    
    autotuner = AutoTuner.from_kernel(
        kernel=flash_attention_kernel,
        configs=flash_attention_configs(),
    ).set_compile_args(
        out_idx=[-1],
        target="npuir",
    ).set_profile_args(
        warmup=10,
        rep=50,
        skip_check=False,
    )
    
    result = autotuner.run()
    return result.kernel
```

### Configuration Search Space

#### GEMM Config Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `block_M` | [64, 128] | Block size in M dimension |
| `block_N` | [128, 256] | Block size in N dimension |
| `K_L1` | [16, 32] | K dimension L1 cache block |

**Total configs**: 2 × 2 × 2 = 8

#### Flash Attention Config Space

| Parameter | Values | Description |
|-----------|--------|-------------|
| `block_M` | [32, 64] | Query block size |
| `block_N` | [32, 64] | Key/Value block size |
| `block_K` | [16, 32] | K dimension block size |

**Total configs**: 2 × 2 × 2 = 8

---

### Phase 2: Bucketing Strategy (Optional, Future)

**Goal**: Different optimal configs for different shape ranges.

**When to implement**: If Phase 1 testing shows significant performance variance across shapes.

#### Bucketing Approach

```
Compile Phase:
  small shape  (M,N,K <= 512)   → autotune → kernel_small
  medium shape (512 < M,N,K <= 2048) → autotune → kernel_medium  
  large shape  (M,N,K > 2048)  → autotune → kernel_large

Runtime Phase:
  Select kernel based on input shape
```

#### Files to Modify

| File | Changes |
|------|---------|
| `compile/kernels/gemm.py` | Generate configs for each bucket, separate autotune |
| `compile/precompile.py` | Save multiple kernels (gemm_small, gemm_medium, gemm_large) |
| `src/loader.py` | Support loading different kernels by shape |
| `src/ops/gemm.py` | Select kernel based on shape |

#### Implementation Sketch

```python
# compile/kernels/gemm.py

GEMM_BUCKETS = {
    "small": {"max_dim": 512},
    "medium": {"max_dim": 2048},
    "large": {"max_dim": float('inf')},
}

def compile_gemm_kernel():
    """Compile GEMM kernels for all buckets"""
    from tilelang.autotuner import AutoTuner
    
    kernels = {}
    for bucket_name, bucket_config in GEMM_BUCKETS.items():
        # Generate test shapes for this bucket
        test_shapes = get_test_shapes_for_bucket(bucket_name)
        
        autotuner = AutoTuner.from_kernel(
            kernel=gemm_kernel,
            configs=gemm_configs(),
        ).set_profile_args(
            supply_prog=lambda _: generate_inputs(test_shapes),
        )
        
        result = autotuner.run()
        kernels[bucket_name] = result.kernel
    
    return kernels  # Return dict of kernels
```

```python
# src/ops/gemm.py

class GemmOp(BaseOp):
    _kernels = {}  # bucket_name -> kernel
    
    def __call__(self, a, b):
        M, N, K = a.shape[0], b.shape[1], a.shape[1]
        
        # Select bucket based on shape
        bucket = self._get_bucket(M, N, K)
        
        if bucket not in self._kernels:
            self._kernels[bucket] = self._load_kernel(bucket)
        
        c = torch.empty(M, N, dtype=a.dtype, device=a.device)
        self._kernels[bucket](a, b, c)
        return c
    
    def _get_bucket(self, M, N, K):
        max_dim = max(M, N, K)
        if max_dim <= 512:
            return "small"
        elif max_dim <= 2048:
            return "medium"
        else:
            return "large"
```

---

## Expected Results

| Metric | Current | Phase 1 (Autotune) | Phase 2 (Bucketing) |
|--------|---------|-------------------|---------------------|
| Compile time | ~1 min | ~5-10 min | ~15-30 min |
| Offline deploy | Yes | Yes | Yes |
| Performance | Baseline | +10-30% | +20-50% |
| Runtime overhead | None | None | Minimal (bucket selection) |

---

## Risks and Mitigation

| Risk | Mitigation |
|------|------------|
| Long compile time | Reduce config count, use parallel compilation |
| Some configs fail to compile | Autotuner automatically skips failed configs |
| Modest performance gain | Expand search space, or implement Phase 2 bucketing |

---

## Reference

- Autotuner source: `tilelang/autotuner/tuner.py`
- Autotuner cache: `tilelang/cache/tuner_cache.py`
- Example usage: `testing/python/autotune/test_tilelang_autotune.py`
