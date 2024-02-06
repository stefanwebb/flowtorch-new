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

# Random thought: Should splitting a flow into e.g. 2 flows be handled by
# introducing additional dimensions, or by returning tuples?

if __name__ == "__main__":
    x = LazyWithArgs(Affine, params_fn=Tensor)

    y = torch.randn(
        16,
    )
    # print(x._forward(y))

    base_dist = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(2), torch.ones(2)), 1
    )

    # base_dist2 = torch.distributions.Normal(torch.zeros(2), torch.ones(2))
    # print("base distribution")
    # print("batch shape", base_dist.batch_shape)
    # print("event shape", base_dist.event_shape)
    # print("")
    # print("batch shape", base_dist2.batch_shape)
    # print("event shape", base_dist2.event_shape)

    d = Flow(base_dist, x)
    y = d.sample()
    # y = base_dist.sample()

    print(y)

    log_p = d.log_prob(y)
    print(log_p)

    y2 = d.sample()

    log_p = d.log_prob(y)
    print(log_p)
