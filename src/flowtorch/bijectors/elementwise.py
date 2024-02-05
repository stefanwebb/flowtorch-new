# Copyright (c) The FlowTorch Team
from typing import Any, Optional

import torch
import torch.distributions
from flowtorch.bijectors.base import Bijector
from flowtorch.lazy import Lazy
from flowtorch.parameters.tensor import Tensor
from flowtorch.parameters.base import Parameters


class Elementwise(Bijector):
    def __init__(
        self,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        params_fn: Optional[Lazy[Parameters]] = None,
        **kwargs: Any,
    ) -> None:
        if not params_fn:
            params_fn = Tensor()  # type: ignore

        assert (
            params_fn is None
            or issubclass(params_fn, Tensor)
            or (issubclass(params_fn, Lazy) and issubclass(params_fn._cls, Tensor))
        )

        super().__init__(shape=shape, context_shape=context_shape, params_fn=params_fn)
