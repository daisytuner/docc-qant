import pytest
import numpy as np
import ml_dtypes

from docc.python import native
from docc.qant import register_docc_plugin

register_docc_plugin()


@native(target="qant")
def matmul_bf16(a, b):
    return a @ b

def test_matmul_bf16():

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
    B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)

    C = matmul_bf16(A, B)
    C_ref = np.matmul(A, B).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(C, C_ref, rtol=2e-2, atol=0)


def test_matmul_scaled_bf16():

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(ml_dtypes.bfloat16) * 10
    B = np.random.rand(K, N).astype(ml_dtypes.bfloat16) * 10

    C = matmul_bf16(A, B)
    C_ref = np.matmul(A, B).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(C, C_ref, rtol=3e-2, atol=0)


def test_two_matmul_bf16():

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
    B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)
    C = np.random.rand(N, M).astype(ml_dtypes.bfloat16)
    temp1 = matmul_bf16(A, B)

    np.testing.assert_allclose(
        temp1, np.matmul(A, B).astype(ml_dtypes.bfloat16), rtol=2e-2, atol=0
    )

    D = matmul_bf16(temp1, C)
    D_ref = (np.matmul(np.matmul(A, B), C)).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(D, D_ref, rtol=2e-2, atol=0)


def test_chained_matmul_bf16():
    @native(target="qant")
    def matmul_bf16_chained(a, b, c):
        return a @ b @ c

    M, K, N = 32, 48, 64
    A = np.random.rand(M, K).astype(ml_dtypes.bfloat16)
    B = np.random.rand(K, N).astype(ml_dtypes.bfloat16)
    C = np.random.rand(N, M).astype(ml_dtypes.bfloat16)
    D = matmul_bf16_chained(A, B, C)
    D_ref = (np.matmul(np.matmul(A, B), C)).astype(ml_dtypes.bfloat16)
    np.testing.assert_allclose(D, D_ref, rtol=2e-2, atol=0)
