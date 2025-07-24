from dataclasses import dataclass
from ...utils.general import JSONable, HDF5able
from ..units.united import United
from abc import ABC, abstractmethod
from ...unit import Unit
from typing import TypeVar, Generic, TYPE_CHECKING, Any
from ..scalars.base_scalar import BaseScalar
from ...unit import Unit

if TYPE_CHECKING:
    pass

PT = TypeVar("PT", bound=float|complex)
UST = TypeVar("UST", bound="UnitedScalar[Any, Any]")

@dataclass(frozen=True, slots=True)
class UnitedScalar(BaseScalar, ABC, JSONable, HDF5able, United, Generic[UST, PT]):

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
    def __abs__(self) -> UST:
        ...

    @abstractmethod
    def __pow__(self, exponent: PT) -> UST:
        ...

    @abstractmethod
    def __neg__(self) -> UST:
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
        if not unit.compatible_to(self.dimension): # type: ignore
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_value) # type: ignore
    
    def value_in_canonical_unit(self) -> PT:
        return self.value_in_unit(self.dimension.canonical_unit)
    
    def value_in_display_unit(self) -> PT:
        if self._display_unit is None:
            raise ValueError("This scalar has no display unit")
        return self.value_in_unit(self._display_unit) # type: ignore