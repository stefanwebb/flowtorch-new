# Copyright (c) The FlowTorch Team

# from tensordict.prototype import tensorclass
from dataclasses import dataclass

import torch


# @tensorclass
# NOTE: Until functorch supports Python 3.12, we'll use dataclass instead of
# tensorclass
@dataclass
class BijectiveTensor:
    """
    This class represents a container for the output of a bijector and takes
    care of caching the input and related values, which is useful for other
    options performed on a normalizing flow.

    """

    # Caching of input and output of bijection
    x: torch.Tensor
    y: torch.Tensor
    log_abs_det_jacobian: torch.Tensor

    # Is there a need to store context here?
    # context: torch.Tensor

    # Caching a hash for evaluation of parameters so can link
    # to a specific bijector
    hash: int

    # TODO: Do I need separate ones of these for x and y?
    batch_shape: torch.Size
    event_shape: torch.Size

    # TODO: Following as property
    # def sample_shape(...):
    # Can infer
