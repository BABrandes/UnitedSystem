from typing import Optional, Protocol, runtime_checkable
from abc import abstractmethod

from .unit import Unit
from .dimension import Dimension

@runtime_checkable
class United(Protocol):
    """
    A protocol for objects that have a unit.

    This protocol is used to mark objects that have a unit.
    It is used to ensure that objects that have a unit are compatible with the unit system.

    It is useful for example in type hints to indicate that an object has a unit such as isinstance(obj, United).
    """

    dimension: Dimension
    _display_unit: Optional[Unit]

    @property
    @abstractmethod
    def unit(self) -> Unit: ...