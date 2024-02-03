# Copyright (c) The FlowTorch Team
from typing import (
    Any,
)
from inspect import isclass


class LazyWithArgs[T]:
    # cls: type[T]
    # args: Optional[Iterable[Any]] = None
    # kwargs: Optional[Dict[str, Any]] = None

    def __init__(self, cls: type[T], *args: Any, **kwargs: Any):
        assert isclass(cls)

        # TODO: Confirm that args and kwargs agree with signature of T.__init__

        self._cls = cls
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs) -> T:
        """
        Instantiate the lazy object. Keyword args in this call overwrite ones
        with the same key from constructor.
        """
        return self._cls(*self._args, *args, **dict(kwargs, **self._kwargs))

    def __str__(self):
        return f"LazyObject[{self._cls.__name__}](cls={str(self._cls)}, args={str(self._args)}, kwargs={str(self._kwargs)})"


type Lazy[T] = type[T] | LazyWithArgs[T]


def lazy(cls: type, *args, **kwargs) -> LazyWithArgs[Any]:
    return LazyWithArgs(cls, *args, **kwargs)
