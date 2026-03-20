import pytest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()

docc.torch.set_backend_options(target="qant", category="server")


@pytest.mark.skip(reason="Cannot yet map fp32 to qant operations")
def test_inference_fp32():
    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = LinearNet()
    model.eval()
    model_ref = LinearNet()
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(2, 4)

    # Force dynamo (inference) backend
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert res.shape == (2, 2)
    assert torch.allclose(res, ref, rtol=1e-1)


@pytest.mark.skip(reason="torch-mlir adds intermediary types that are unsupported")
def test_inference_bf16():
    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear = nn.Linear(
                in_features, out_features, bias=False, dtype=torch.bfloat16
            )

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    model = LinearNet().to(torch.bfloat16)
    model.eval()
    model_ref = LinearNet().to(torch.bfloat16)
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(2, 4, dtype=torch.bfloat16)

    # Force dynamo (inference) backend
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert res.shape == (2, 2)
    assert torch.allclose(res, ref, rtol=1e-1)
