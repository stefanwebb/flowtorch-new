# Copyright (c) The FlowTorch Team
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.distributions

# from flowtorch.bijectors.bijective_tensor import BijectiveTensor, to_bijective_tensor
from src.lazy import Lazy
from src.parameters.base import Parameters
from torch.distributions import constraints


class Bijector(torch.nn.Module):
    codomain: constraints.Constraint = constraints.real
    domain: constraints.Constraint = constraints.real

    def __init__(
        self,
        shape: torch.Size,
        context_shape: Optional[torch.Size] = None,
        params_fn: Optional[Lazy[Parameters]] = None,
    ) -> None:
        super().__init__()

        # Prevent "meta bijectors" from being initialized
        # NOTE: We define a "standard bijector" as one that inherits from a
        # subclass of Bijector, hence why we need to test the length of the MRO
        if (
            self.__class__.__module__ == "flowtorch.bijectors.base"
            or len(self.__class__.__mro__) <= 3
        ):
            raise TypeError("Only standard bijectors can be initialized.")

        self._shape = shape
        self._context_shape = context_shape

        # Instantiate parameters (tensor, hypernets, etc.)
        # TODO: Should we simplify this by having a parameters.List?
        self._params_fn: Optional[Union[Parameters, torch.nn.ModuleList]] = None
        if params_fn is not None:
            param_shapes = self.param_shapes(shape)
            self._params_fn = params_fn(  # type: ignore
                param_shapes, self._shape, self._context_shape
            )

    # TODO: Re-enable bijective tensors
    # def _check_bijective_x(
    #     self, x: torch.Tensor, context: Optional[torch.Tensor]
    # ) -> bool:
    #     return (
    #         isinstance(x, BijectiveTensor)
    #         and x.from_inverse()
    #         and x.check_bijector(self)
    #         and x.check_context(context)
    #     )

    def _forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)

        # TODO: Re-enable bijective tensors
        # if self._check_bijective_x(x, context):
        #     assert isinstance(x, BijectiveTensor)
        #     return x.get_parent_from_bijector(self)

        params = self._params_fn(x, context) if self._params_fn is not None else None
        y, log_detJ = self.forward(x, params)

        # TODO: Re-enable bijective tensors
        # if (
        #     is_record_flow_graph_enabled()
        #     and not isinstance(y, BijectiveTensor)
        #     and not (isinstance(x, BijectiveTensor) and y in set(x.parents()))
        # ):
        #     # we exclude y that are bijective tensors for Compose
        #     y = to_bijective_tensor(x, y, context, self, log_detJ, mode="forward")

        return y

    def forward(
        self,
        x: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Abstract method to compute forward transformation.
        """
        raise NotImplementedError(
            f"layer {self.__class__.__name__} does not have an `_forward` method"
        )

    # TODO: Re-enable bijective tensors
    # def _check_bijective_y(
    #     self, y: torch.Tensor, context: Optional[torch.Tensor]
    # ) -> bool:
    #     return (
    #         isinstance(y, BijectiveTensor)
    #         and y.from_forward()
    #         and y.check_bijector(self)
    #         and y.check_context(context)
    #     )

    # TODO: Swap inverse and _inverse?
    def _inverse(
        self,
        y: torch.Tensor,
        x: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)

        # TODO: Re-enable bijective tensors
        # if self._check_bijective_y(y, context):
        #     assert isinstance(y, BijectiveTensor)
        #     return y.get_parent_from_bijector(self)

        params = self._params_fn(x, context) if self._params_fn is not None else None
        x, log_detJ = self._inverse(y, params)

        # TODO: Re-enable bijective tensors
        # if (
        #     is_record_flow_graph_enabled()
        #     and not isinstance(x, BijectiveTensor)
        #     and not (isinstance(y, BijectiveTensor) and x in set(y.parents()))
        # ):
        #     x = to_bijective_tensor(x, y, context, self, log_detJ, mode="inverse")

        return x

    def inverse(
        self,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Abstract method to compute inverse transformation.
        """
        raise NotImplementedError(
            f"layer {self.__class__.__name__} does not have an `_inverse` method"
        )

    # TODO: Swap log_abs_det_jacobian and _log_abs_det_jacobian?
    def _log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """
        # TODO: Allow that context can have a batch shape
        assert context is None  # or context.shape == (self._context_size,)
        ladj = None
        # if (
        #     isinstance(y, BijectiveTensor)
        #     and y.from_forward()
        #     and y.check_bijector(self)
        #     and y.check_context(context)
        # ):
        #     ladj = y.log_detJ
        # elif (
        #     isinstance(x, BijectiveTensor)
        #     and x.from_inverse()
        #     and x.check_bijector(self)
        #     and x.check_context(context)
        # ):
        #     ladj = x.log_detJ
        if ladj is None:
            params = (
                self._params_fn(x, context) if self._params_fn is not None else None
            )
            return self._log_abs_det_jacobian(x, y, params)
        return ladj

    def log_abs_det_jacobian(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        params: Optional[Sequence[torch.Tensor]],
    ) -> torch.Tensor:
        """
        Computes the log det jacobian `log |dy/dx|` given input and output.
        By default, assumes a volume preserving bijection.
        """

        raise NotImplementedError(
            f"layer {self.__class__.__name__} does not have an `_log_abs_det_jacobian` method"
        )

    def param_shapes(self, shape: torch.Size) -> Sequence[torch.Size]:
        """
        Abstract method to return shapes of parameters
        """
        # TODO: Default to empty sequence?
        raise NotImplementedError(
            f"layer {self.__class__.__name__} does not have an `param_shapes` method"
        )

    # TODO: Shouldn't this be __str__?
    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"

    def forward_shape(self, shape: torch.Size) -> torch.Size:
        """
        Infers the shape of the forward computation, given the input shape.
        Defaults to preserving shape.
        """
        return shape

    def inverse_shape(self, shape: torch.Size) -> torch.Size:
        """
        Infers the shapes of the inverse computation, given the output shape.
        Defaults to preserving shape.
        """
        return shape
