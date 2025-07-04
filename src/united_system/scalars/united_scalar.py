from dataclasses import dataclass
from ..utils import JSONable, HDF5able
from ..units.utils import United
from abc import ABC
from ..units.unit import UnitQuantity

@dataclass(frozen=True, slots=True)
class UnitedScalar(ABC, JSONable, HDF5able, United):

    @property
    def unit_quantity(self) -> UnitQuantity:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __add__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __radd__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __sub__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __rsub__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __mul__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __rmul__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __truediv__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __rtruediv__(self, other: "UnitedScalar") -> "UnitedScalar":
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __le__(self, other: "UnitedScalar") -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __ge__(self, other: "UnitedScalar") -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __lt__(self, other: "UnitedScalar") -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")

    def __gt__(self, other: "UnitedScalar") -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __eq__(self, other: "UnitedScalar") -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def __ne__(self, other: "UnitedScalar") -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def is_nan(self) -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def is_finite(self) -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def is_positive(self) -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def is_negative(self) -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")
    
    def is_zero(self) -> bool:
        raise NotImplementedError("Operation on the abstract base class called!")