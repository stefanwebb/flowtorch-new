# Copyright (c) The FlowTorch Team

from typing import Any, Dict, Optional, Union

from flowtorch.lazy import Lazy
from flowtorch.bijectors.base import Bijector
from flowtorch.bijective_tensor import BijectiveTensor
import torch
import torch.distributions as dist
from torch import Tensor
from torch.distributions.utils import _sum_rightmost


# TODO: Should the hash be handled by Flow rather than Bijector?
class Flow(torch.nn.Module, dist.Distribution):
    _default_sample_shape = torch.Size()
    arg_constraints: Dict[str, dist.constraints.Constraint] = {}

    def __init__(
        self,
        base_dist: dist.Distribution,  # Base distribution can't have learnable parameters
        bijector: Lazy[Bijector],
        validate_args: Any = None,
    ) -> None:
        torch.nn.Module.__init__(self)

        self.base_dist = base_dist
        self._context: Optional[torch.Tensor] = None
        self.bijector = bijector(shape=base_dist.event_shape)

        # TODO: Confirm that bijector was initialized and has correct type

        # TODO: Confirm that the following logic works. Shouldn't it use
        # .domain and .codomain?? Infer shape from constructed self.bijector
        # TODO: Actually, use the bijector output shape here!
        shape = (
            self.base_dist.batch_shape + self.base_dist.event_shape  # pyre-ignore[16]
        )
        event_dim = self.bijector.domain.event_dim  # type: ignore
        event_dim = max(event_dim, len(self.base_dist.event_shape))
        batch_shape = shape[: len(shape) - event_dim]
        event_shape = shape[len(shape) - event_dim :]

        dist.Distribution.__init__(
            self, batch_shape, event_shape, validate_args=validate_args
        )

    def __hash__(self):
        return hash(self.bijector)

    def sample(
        self,
        sample_shape: Union[Tensor, torch.Size] = _default_sample_shape,
        context: Optional[torch.Tensor] = None,
    ) -> BijectiveTensor:  # Tensor:
        """
        Generates a sample_shape shaped sample or sample_shape shaped batch of
        samples if the distribution parameters are batched. Samples first from
        base distribution and applies `transform()` for every transform in the
        list.
        """
        if context is None:
            context = self._context
        with torch.no_grad():
            x = self.base_dist.sample(sample_shape)
            y, log_detJ = self.bijector._forward(x, context)

            return BijectiveTensor(
                x=x,
                y=y,
                log_abs_det_jacobian=log_detJ,
                batch_shape=self.batch_shape,
                event_shape=self.event_shape,
                hash=hash(self),
            )

    def rsample(
        self,
        sample_shape: Union[Tensor, torch.Size] = _default_sample_shape,
        context: Optional[torch.Tensor] = None,
    ) -> BijectiveTensor:
        """
        Generates a sample_shape shaped reparameterized sample or sample_shape
        shaped batch of reparameterized samples if the distribution parameters
        are batched. Samples first from base distribution and applies
        `transform()` for every transform in the list.
        """
        x = self.base_dist.rsample(sample_shape)
        y, log_detJ = self.bijector._forward(x, context)

        return BijectiveTensor(
            x=x,
            y=y,
            log_abs_det_jacobian=log_detJ,
            batch_shape=self.batch_shape,
            event_shape=self.event_shape,
            hash=hash(self),
        )

    def rnormalize(
        self, y: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> BijectiveTensor:
        """
        Push a tensor through the normalizing direction of the flow where
        we can take autodiff gradients on the bijector.
        """

        x, log_detJ = self.bijector._inverse(y, None, context)  # type: ignore
        return BijectiveTensor(
            x=x,
            y=y,
            log_abs_det_jacobian=log_detJ,
            batch_shape=self.batch_shape,
            event_shape=self.event_shape,
            hash=hash(self),
        )

    def normalize(
        self, value: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> BijectiveTensor:
        """
        Push a tensor through the normalizing direction of the flow and
        block autodiff gradients on the bijector.
        """
        with torch.no_grad():
            return self.rnormalize(value, context)

    def log_prob(
        self,
        value: Union[Tensor, BijectiveTensor],
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Scores the sample by inverting the transform(s) and computing the score
        using the score of the base distribution and the log abs det jacobian.
        """
        event_dim = len(self.event_shape)  # pyre-ignore[16]

        if (
            isinstance(value, BijectiveTensor)
            and value.hash == hash(self)
            and context is None
        ):
            log_detJ = value.log_abs_det_jacobian
            x = value.x
            print("Using cached value of log_detJ")

        else:
            y = value.y if isinstance(value, BijectiveTensor) else value

            # TODO: Why does _inverse take x?
            x, log_detJ = self.bijector._inverse(y=y, x=None, context=context)  # type: ignore
            # log_detJ = self.bijector._log_abs_det_jacobian(x, value, context)

        log_prob = -_sum_rightmost(
            log_detJ,  # type: ignore
            event_dim - self.bijector.domain.event_dim,  # type: ignore
        )
        log_prob = log_prob + _sum_rightmost(
            self.base_dist.log_prob(x),
            event_dim - len(self.base_dist.event_shape),  # pyre-ignore[16]
        )

        return log_prob
