import pytest

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import docc.torch
from docc.qant import register_docc_plugin

register_docc_plugin()


def test_linear_regression_mse():
    """Verify gradient descent learns a known linear function."""
    # Target: learn the 2x2 identity matrix
    target_weights = torch.eye(2)

    class LinearNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 2, bias=False)

        def forward(self, x: torch.Tensor):
            return self.linear(x)

    torch.manual_seed(42)
    model = LinearNet()

    program = torch.compile(
        model, backend="docc", options={"target": "qant", "category": "server"}
    )
    optimizer = torch.optim.SGD(program.parameters(), lr=0.5)
    criterion = nn.MSELoss()

    # Train on random inputs, target = input (identity function)
    for _ in range(20):
        x = torch.randn(32, 2)
        target = x  # Identity: output should equal input

        optimizer.zero_grad()
        res = program(x)
        loss = criterion(res, target)
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item():.4f}")
        print("Current weights:")
        print(model.linear.weight)

    # Verify learned weights converged to identity matrix
    learned_weights = model.linear.weight.detach()
    print(learned_weights)
    assert torch.allclose(
        learned_weights, target_weights, atol=0.05
    ), f"Expected identity matrix, got:\n{learned_weights}"
