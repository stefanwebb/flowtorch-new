# Copyright (c) The FlowTorch Team

from typing import Optional, Sequence

import torch


class Parameters(torch.nn.Module):
    """
    Parameters of a flowtorch.Bijector object.
    """

    def __init__(
        self,
        param_shapes: Sequence[torch.Size],
        input_shape: torch.Size,
        context_shape: Optional[torch.Size],
    ) -> None:
        super().__init__()
        self.param_shapes = param_shapes
        self.input_shape = input_shape
        self.context_shape = context_shape

    def forward(
        self,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> Optional[Sequence[torch.Tensor]]:
        raise NotImplementedError(
            f"layer {self.__class__.__name__} does not have an `_forward` method"
        )
