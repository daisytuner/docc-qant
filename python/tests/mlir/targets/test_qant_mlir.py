import pytest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()

docc.torch.set_backend_options(target="qant", category="server")


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
    assert torch.allclose(res, ref, atol=2e-2)


def test_chained_linear_fp32():
    class LinearNet(nn.Module):
        def __init__(self, in_features=4, out_features=2):
            super().__init__()
            self.linear1 = nn.Linear(in_features, in_features, bias=False)
            self.linear2 = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor):
            l1 = self.linear1(x)
            return self.linear2(l1)

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


def test_single_nobias_compile():
    class SingleConv2dNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, kernel_size=3, bias=False)

        def forward(self, x: torch.Tensor):
            return self.conv(x)

    model = SingleConv2dNet()
    model_ref = SingleConv2dNet()
    model_ref.load_state_dict(model.state_dict())
    example_input = torch.randn(1, 3, 32, 32)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, atol=2e-2)


def test_maxpool2d_compile():
    class MaxPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    model = MaxPoolNet()
    model.eval()
    model_ref = MaxPoolNet()
    model_ref.eval()
    example_input = torch.randn(1, 1, 4, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, atol=1e-2)


def test_maxpool2d_batched_compile():
    class MaxPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    model = MaxPoolNet()
    model.eval()
    model_ref = MaxPoolNet()
    model_ref.eval()
    example_input = torch.randn(4, 2, 6, 6)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)
    assert res.shape == (4, 2, 3, 3)
    assert torch.allclose(res, res_ref, atol=1e-2)


def test_avgpool2d_compile():
    class AvgPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    model = AvgPoolNet()
    model.eval()
    model_ref = AvgPoolNet()
    model_ref.eval()
    example_input = torch.randn(1, 1, 4, 4)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)
    assert torch.allclose(res, res_ref, atol=1e-2)


def test_avgpool2d_batched_compile():
    class AvgPoolNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor):
            return self.pool(x)

    model = AvgPoolNet()
    model.eval()
    model_ref = AvgPoolNet()
    model_ref.eval()
    example_input = torch.randn(4, 2, 6, 6)

    program = torch.compile(model, backend="docc")
    with torch.no_grad():
        res = program(example_input)
        res_ref = model_ref(example_input)
    assert res.shape == (4, 2, 3, 3)
    assert torch.allclose(res, res_ref, atol=1e-2)
    assert torch.allclose(res, res_ref, atol=1e-2)
