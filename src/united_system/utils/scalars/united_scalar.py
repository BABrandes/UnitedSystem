from dataclasses import dataclass
from ...utils.general import JSONable, HDF5able
from ..units.united import United
from abc import ABC, abstractmethod
from ..units.base_classes.base_unit import BaseDimension
from ..units.base_classes.base_unit import BaseUnit
from typing import TypeVar, Generic, TYPE_CHECKING, Any
from ..scalars.base_scalar import BaseScalar
from ...unit import Unit

if TYPE_CHECKING:
    pass

PT = TypeVar("PT", bound=float|complex)
UST = TypeVar("UST", bound="UnitedScalar")
UT = TypeVar("UT", bound=BaseUnit[Any, Any])
UD = TypeVar("UD", bound=BaseDimension[Any, Any])

@dataclass(frozen=True, slots=True)
class UnitedScalar(BaseScalar, ABC, JSONable, HDF5able, United[UD, UT], Generic[UST, UT, UD, PT]):

    canonical_value: PT
    
    @abstractmethod
    def __add__(self, other: UST) -> UST:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    @abstractmethod
    def __radd__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __sub__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __rsub__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __mul__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __rmul__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __truediv__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __rtruediv__(self, other: UST) -> UST:
        ...
    
    @abstractmethod
    def __le__(self, other: UST) -> bool:
        ...
    
    @abstractmethod
    def __ge__(self, other: UST) -> bool:
        ...
    
    @abstractmethod
    def __lt__(self, other: UST) -> bool:
        ...

    @abstractmethod
    def __gt__(self, other: UST) -> bool:
        ...
    
    @abstractmethod
    def __eq__(self, other: object) -> bool:
        ...
    
    @abstractmethod
    def __ne__(self, other: object) -> bool:
        ...
    
    @abstractmethod
    def is_nan(self) -> bool:
        ...
    
    @abstractmethod
    def is_finite(self) -> bool:
        ...
    
    @abstractmethod
    def is_positive(self) -> bool:
        ...
    
    @abstractmethod
    def is_negative(self) -> bool:
        ...
    
    @abstractmethod
    def is_zero(self) -> bool:
        ...
    
    @abstractmethod
    def compatible_to(self, *args: UST) -> bool:
        ...

    def value_in_unit(self, unit: Unit) -> PT:
        """
        Convert the scalar to a different unit.
        """
        return unit.from_canonical_value(self.canonical_value) # type: ignore