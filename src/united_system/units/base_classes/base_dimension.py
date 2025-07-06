from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar, Generic
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .base_unit import BaseUnit

MASS_INDEX: int = 0
TIME_INDEX: int = 1
LENGTH_INDEX: int = 2
CURRENT_INDEX: int = 3
TEMPERATURE_INDEX: int = 4
AMOUNT_OF_SUBSTANCE_INDEX: int = 5
LUMINOUS_INTENSITY_INDEX: int = 6

DT = TypeVar("DT", bound="BaseDimension")
UT = TypeVar("UT", bound="BaseUnit")

@dataclass(frozen=True, slots=True)
class BaseDimension(ABC, Generic[DT,UT]):
    """
    UnitDimension is the abstract base class for all unit dimensions.
    A unit dimension represents the physical dimensions of a unit (e.g., length, mass, time).
    """
    
    @property
    @abstractmethod
    def canonical_unit(self) -> "UT":
        ...

    @abstractmethod
    def __add__(self, other: DT) -> DT:
        ...
    
    @abstractmethod
    def __radd__(self, other: DT) -> DT:
        ...
    
    @abstractmethod
    def __sub__(self, other: DT) -> DT:
        ...
    
    @abstractmethod
    def __rsub__(self, other: DT) -> DT:
        ...
    
    @abstractmethod
    def __mul__(self, other: float) -> DT:
        ...
    
    @abstractmethod
    def __truediv__(self, other: float) -> DT:
        ...
    
    @abstractmethod
    def __eq__(self, other: DT) -> bool:
        ...
    
    @abstractmethod
    def __ne__(self, other: DT) -> bool:
        ...
    
    @abstractmethod
    def __hash__(self) -> int:
        ...
    
    @abstractmethod
    def __str__(self) -> str:
        ...
    
    @abstractmethod
    def compatible_to(self, other: DT) -> bool:
        ...