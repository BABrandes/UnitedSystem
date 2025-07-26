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
    def unit(self) -> Unit: ...