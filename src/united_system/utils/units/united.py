from typing import Generic, TypeVar, Any, Optional
from abc import abstractmethod

from .unit_group import BaseUnit
from .dimension_group import BaseDimension

DT = TypeVar("DT", bound=BaseDimension[Any, Any])
UT = TypeVar("UT", bound=BaseUnit[Any, Any])

class United(Generic[DT, UT]):

    dimension: DT
    _display_unit: Optional[UT]

    @property
    @abstractmethod
    def display_unit(self) -> Optional[UT]: ...

    @property
    @abstractmethod
    def active_unit(self) -> UT:
        """
        The active unit is the unit that is currently being used to display the value or if the display unit is not set, the canonical unit of the set dimension.
        """
        ...