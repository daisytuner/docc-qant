# docc-qant

The repository implements the qant plugin for docc.

```python
import numpy as np
import ml_dtypes
from docc.python import native
from docc.qant import register_docc_plugin

# Registers qant target
register_docc_plugin()

@native(target="qant")
def matmul_bf16(a, b):
    return a @ b

M, K, N = 32, 48, 64
A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)

C = matmul_bf16(A, B)
```
