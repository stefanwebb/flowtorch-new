# Copyright (c) The FlowTorch Team

from typing import Optional

import torch
from flowtorch.bijectors.elementwise import Elementwise
from flowtorch.bijectors.ops.affine import Affine as AffineOp
from flowtorch.lazy import Lazy
from flowtorch.parameters.base import Parameters


# TODO: How is initialization handled when there is multiple inheritance?
# TODO: How about initialization of Bijector when both parents inherit from it
# TODO: In light of this, how can we pass options to initializer of AffineOp?
# Current solution is to set them in the initializer of Affine
class Affine(AffineOp, Elementwise):
    r"""
    Elementwise bijector via the affine mapping :math:`\mathbf{y} = \mu +
    \sigma \otimes \mathbf{x}` where $\mu$ and $\sigma$ are learnable parameters.
    """

    def __init__(
        self,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        params_fn: Optional[Lazy[Parameters]] = None,
        *,
        log_scale_min_clip: float = -5.0,
        log_scale_max_clip: float = 3.0,
        sigmoid_bias: float = 2.0,
    ) -> None:
        super().__init__(shape=shape, context_shape=context_shape, params_fn=params_fn)
        self.log_scale_min_clip = log_scale_min_clip
        self.log_scale_max_clip = log_scale_max_clip
        self.sigmoid_bias = sigmoid_bias
