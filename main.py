# Copyright (c) The FlowTorch Team
"""
Script to test development.

"""
import torch
from src.bijectors.affine import Affine
from src.lazy import Lazy
from src.parameters.tensor import Tensor

if __name__ == "__main__":
    x = Affine(torch.Size((16,)), params_fn=Tensor)

    y = torch.randn(
        16,
    )
    print(x._forward(y))
