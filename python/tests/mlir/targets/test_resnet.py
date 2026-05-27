import pytest
import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()


@pytest.mark.skipif(not os.environ.get("SLOW_TESTS", ""), reason="slow test")
def test_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()

    model_ref = models.resnet18(weights="IMAGENET1K_V1")
    model_ref
    model_ref.eval()

    print("\n--- Compile Times ---")
    start = time.perf_counter()
    program_ref = torch.compile(model_ref)
    torch_compile_time = time.perf_counter() - start
    print(f"Torch compile: {torch_compile_time:.4f} s")

    example_input = torch.randn(1, 3, 224, 224)

    start = time.perf_counter()
    program = torch.compile(
        model, backend="docc", options={"target": "qant", "category": "server"}
    )
    docc_compile_time = time.perf_counter() - start
    print(f"Docc compile: {docc_compile_time:.4f} s")

    # Force dynamo (inference) backend
    with torch.no_grad():
        print("--- Execution Times ---")
        start = time.perf_counter()
        res = program(example_input)
        docc_exec_time = time.perf_counter() - start
        print(f"Docc first execution: {docc_exec_time:.4f} s")

        start = time.perf_counter()
        ref = program_ref(example_input)
        torch_exec_time = time.perf_counter() - start
        print(f"Torch first execution: {torch_exec_time:.4f} s")

        start = time.perf_counter()
        res = program(example_input)
        docc_exec_time = time.perf_counter() - start
        print(f"Docc 2nd execution: {docc_exec_time:.4f} s")

        start = time.perf_counter()
        ref = program_ref(example_input)
        torch_exec_time = time.perf_counter() - start
        print(f"Torch 2nd execution: {torch_exec_time:.4f} s")

    assert torch.allclose(res, ref, rtol=3e-1, atol=1e-1)
