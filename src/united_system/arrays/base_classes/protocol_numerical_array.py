import numpy as np
from typing import TypeVar, Generic, Protocol, runtime_checkable
from ...units.base_classes.base_unit import BaseUnit
from .base_array import SCALAR_TYPE


ST = TypeVar("ST", bound=SCALAR_TYPE)
UT = TypeVar("UT", bound=BaseUnit|None)


@runtime_checkable
class ProtocolNumericalArray(Protocol, Generic[ST]):

    def sum(self) -> ST:
        ...

    def mean(self) -> ST:
        ...

    def std(self) -> ST:
        ...

    def min(self) -> ST:
        ...

    def max(self) -> ST:
        ...
