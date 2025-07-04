from typing import Protocol, runtime_checkable
from .unit import Unit
from .unit_quantity import UnitQuantity


@runtime_checkable
class United(Protocol):
    @property
    def unit(self) -> Unit:
        ...
    @property
    def unit_quantity(self) -> UnitQuantity:
        ...
