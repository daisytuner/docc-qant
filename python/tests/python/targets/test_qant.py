import pytest
import numpy as np

from docc.python import native
from docc.qant import register_docc_plugin

register_docc_plugin()


def test_qant_matmul():
    @native(target="qant", category="server")
    def qant_matmul(a, b):
        return a @ b

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(np.float64)
    B = np.random.rand(K, N).astype(np.float64)

    C = qant_matmul(A, B)
    assert np.allclose(C, A @ B)
