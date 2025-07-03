from dataclasses import dataclass, field
from .units.unit import Unit, SimpleUnit
from .units.unit_quantity import CanonicalQuantity, SimpleCanonicalQuantity
from .units.named_canonical_quantities import NamedCanonicalQuantity
import math
from .utils import str_to_float
from typing import Union, TypeAlias, cast
from abc import ABC
import h5py
import numpy as np

@dataclass(frozen=True, slots=True)
class UnitedScalar(ABC):
    
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