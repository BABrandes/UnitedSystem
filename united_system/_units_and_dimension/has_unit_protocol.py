from typing import Optional, Protocol, runtime_checkable, TYPE_CHECKING, Union
from abc import abstractmethod

if TYPE_CHECKING:
    from .unit import Unit
    from .dimension import Dimension

@runtime_checkable
class HasUnit(Protocol):
    """
    A protocol for objects that have a unit.

    This protocol is used to mark objects that have a unit.
    It is used to ensure that objects that have a unit are compatible with the unit system.

    It is useful for example in type hints to indicate that an object has a unit such as isinstance(obj, United).
    """

    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    @property
    @abstractmethod
    def unit(self) -> "Unit": ...

    def compatible_to(self, *others: Union["Dimension", "Unit", "HasUnit"]) -> bool:
        """
        Check if the dimension is compatible with other dimensions.
        Two dimensions are compatible if they have the same subscripts
        and the same proper exponents.
        """
        return Dimension.are_compatible(self.dimension, *others)