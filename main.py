# Copyright (c) The FlowTorch Team
"""
Script to test development.

"""
import torch
import flowtorch
from flowtorch.bijectors.affine import Affine
from flowtorch.lazy import LazyWithArgs
from flowtorch.parameters.tensor import Tensor
from flowtorch.distributions.flow import Flow

if __name__ == "__main__":
    x = LazyWithArgs(Affine, params_fn=Tensor)

    y = torch.randn(
        16,
    )
    # print(x._forward(y))

    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )

    d = Flow(base_dist, x)

    y = d.sample()
    y2 = d.log_prob(y)
    print(y, y2)
