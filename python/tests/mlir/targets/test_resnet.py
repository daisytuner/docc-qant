import pytest
import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()

docc.torch.set_backend_options(target="qant", category="server")


@pytest.mark.skip(reason="Does not map all mlir components yet")
def test_resnet18():
    model = models.resnet18(weights="IMAGENET1K_V1")
    model = model

    model_ref = models.resnet18(weights="IMAGENET1K_V1")
    model_ref = model_ref
    model_ref.eval()
    model_ref.load_state_dict(model.state_dict())

    program = torch.compile(model, backend="docc")

    example_input = torch.randn(1, 3, 224, 224)

    # Force dynamo (inference) backend
    res = program(example_input)
    ref = model_ref(example_input)

    print(res)
