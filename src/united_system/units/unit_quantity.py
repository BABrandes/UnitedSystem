from dataclasses import dataclass
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .unit import Unit

MASS_INDEX: int = 0
TIME_INDEX: int = 1
LENGTH_INDEX: int = 2
CURRENT_INDEX: int = 3
TEMPERATURE_INDEX: int = 4
AMOUNT_OF_SUBSTANCE_INDEX: int = 5
LUMINOUS_INTENSITY_INDEX: int = 6

@dataclass(frozen=True, slots=True)
class UnitQuantity(ABC):
    
    @abstractmethod
    def __add__(self, other: "UnitQuantity") -> "UnitQuantity":
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other: "UnitQuantity") -> "UnitQuantity":
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: "UnitQuantity") -> "UnitQuantity":
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other: "UnitQuantity") -> "UnitQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def __mul__(self, other: float) -> "UnitQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def __truediv__(self, other: float) -> "UnitQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def __eq__(self, other: "UnitQuantity") -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def __ne__(self, other: "UnitQuantity") -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def is_zero(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def to_json(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def from_json(self, json: dict) -> "UnitQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def canonical_unit(self) -> "Unit":
        raise NotImplementedError