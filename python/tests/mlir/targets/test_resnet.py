import pytest
import os
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()

docc.torch.set_backend_options(target="qant", category="server")


@pytest.mark.skipif(not os.environ.get("SLOW_TESTS", ""), reason="slow test")
def test_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.eval()

    model_ref = models.resnet18(weights="IMAGENET1K_V1")
    model_ref.eval()

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(1, 3, 224, 224)

    # Force dynamo (inference) backend
    with torch.no_grad():
        res = program(example_input)
        ref = model_ref(example_input)

    assert torch.allclose(res, ref, rtol=1e-3, atol=1e-5)
