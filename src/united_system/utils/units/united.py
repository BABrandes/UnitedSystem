from typing import Optional
from abc import abstractmethod

from ...unit import Unit
from ...dimension import Dimension

class United():

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