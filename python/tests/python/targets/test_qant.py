import pytest
import numpy as np
import ml_dtypes

from docc.python import native
from docc.qant import register_docc_plugin

register_docc_plugin()


def test_matmul_bf16():
    @native(target="qant")
    def matmul_bf16(a, b):
        return a @ b

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
    B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)

    C = matmul_bf16(A, B)
    C_ref = np.matmul(A, B).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(C, C_ref, rtol=2e-2, atol=0)
