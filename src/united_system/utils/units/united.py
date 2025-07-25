from typing import Optional, Protocol, runtime_checkable
from abc import abstractmethod

from ...unit import Unit
from ...dimension import Dimension

@runtime_checkable
class United(Protocol):

    dimension: Dimension
    _display_unit: Optional[Unit]

    @property
    @abstractmethod
    def display_unit(self) -> Optional[Unit]: ...

    @property
    @abstractmethod
    def active_unit(self) -> Unit:
        """
        The active unit is the unit that is currently being used to display the value or if the display unit is not set, the canonical unit of the set dimension.
        """
        ...