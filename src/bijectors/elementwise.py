# Copyright (c) The FlowTorch Team
from typing import Any, Optional

import torch
import torch.distributions
from src.bijectors.base import Bijector
from src.lazy import Lazy
from src.parameters.tensor import Tensor
from src.parameters.base import Parameters


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
