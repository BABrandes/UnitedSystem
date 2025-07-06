from dataclasses import dataclass
from ..utils import JSONable, HDF5able
from ..units.united import United
from abc import ABC, abstractmethod
from ..units.base_classes.base_unit import BaseDimension
from ..units.base_classes.base_unit import BaseUnit
from typing import TypeAlias, TypeVar, Generic
from ..scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ..scalars.complex_united_scalar.complex_united_scalar import ComplexUnitedScalar
from ..units.base_classes.base_unit import BaseUnit

UNITED_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar

PT = TypeVar("PT", bound=float|complex)
UST = TypeVar("UST", bound=UNITED_SCALAR_TYPE)
UT = TypeVar("UT", bound=BaseUnit)
UD = TypeVar("UD", bound=BaseDimension)

@dataclass(frozen=True, slots=True)
class UnitedScalar(ABC, JSONable, HDF5able, United[UT, UD], Generic[UST, UT, UD, PT]):

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
    def __eq__(self, other: UST) -> bool:
        ...
    
    @abstractmethod
    def __ne__(self, other: UST) -> bool:
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
    def compatible_to(self, other: UST) -> bool:
        ...