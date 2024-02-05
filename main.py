# Copyright (c) The FlowTorch Team
"""
Script to test development.

"""
import torch
import flowtorch
from flowtorch.bijectors.affine import Affine
from flowtorch.lazy import Lazy
from flowtorch.parameters.tensor import Tensor

if __name__ == "__main__":
    x = Affine(torch.Size((16,)), params_fn=Tensor)

    y = torch.randn(
        16,
    )
    print(x._forward(y))
