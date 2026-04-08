# docc-qant

A plugin for [DOCC](https://github.com/daisytuner/docc) (Daisytuner Optimizing Compiler Collection) that adds support for the Q.ANT native computing toolkit, enabling efficient execution of numerical computations with bfloat16 precision on Q.ANT hardware.

## Installation

```bash
pip install docc-qant
```

For PyTorch / MLIR support (Python 3.11 and 3.12 only):

```bash
pip install docc-qant[ai]
pip install torch==2.10.0+cpu torchvision==0.25.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-mlir==20260309.746 -f https://github.com/llvm/torch-mlir-release/releases/expanded_assets/dev-wheels
```

### System Requirements

- Linux (x86_64 or aarch64)
- Python 3.11, 3.12, 3.13, or 3.14
- Q.ANT Native Computing Toolkit installed on the system

## Quick Start

```python
import numpy as np
import ml_dtypes
from docc.python import native
from docc.qant import register_docc_plugin

# Register the Q.ANT target
register_docc_plugin()

# Decorate functions to run on Q.ANT hardware
@native(target="qant")
def matmul_bf16(a, b):
    return a @ b

# Use standard numpy arrays with bfloat16 precision
M, K, N = 32, 48, 64
A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)

# Automatically compiled and executed on Q.ANT hardware
C = matmul_bf16(A, B)
```

## Supported Operations

### Matrix Multiplication

```python
@native(target="qant")
def matmul(a, b):
    return a @ b
```

Supports 2D and batched (3D) matrix multiplication.

### Einsum

```python
@native(target="qant")
def batched_matmul(a, b):
    return np.einsum("bik,bkj->bij", a, b)
```

### Chained Operations

```python
@native(target="qant")
def chained_matmul(a, b, c):
    return a @ b @ c
```

## Links

- [Homepage](https://daisytuner.com)
- [Documentation](https://docs.daisytuner.com)
- [Repository](https://github.com/daisytuner/docc)
- [Issues](https://github.com/daisytuner/docc/issues)
