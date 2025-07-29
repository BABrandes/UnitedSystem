from dataclasses import dataclass, field
from .._scalars.united_scalar import UnitedScalar
from .._units_and_dimension.unit import Unit
from typing import Any, Type, Optional, Union
import h5py
from .._units_and_dimension.dimension import Dimension

@dataclass(frozen=True, slots=True, init=False)
class ComplexUnitedScalar(UnitedScalar["ComplexUnitedScalar", complex]):

    # Explicit dataclass fields
    dimension: Dimension
    _display_unit: Optional[Unit] = field(default=None, repr=False, compare=False, hash=False)
    
    def __init__(self, canonical_value: complex, dimension: Dimension, display_unit: Optional[Unit] = None):
        object.__setattr__(self, "canonical_value", canonical_value)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", display_unit)

    @classmethod
    def create(cls, value: complex, unit: Unit) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.create is not implemented")
    
    @classmethod
    def parse_string(cls, string: str) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.parse_string is not implemented")
    
    @classmethod
    def from_json(cls, data: dict[str, Any], **type_parameters: Type[Dimension]) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.from_json is not implemented")
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **type_parameters: Type[Dimension]) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.from_hdf5 is not implemented")
    
    def to_json(self) -> dict[str, Any]:
        raise NotImplementedError("ComplexUnitedScalar.to_json is not implemented")
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        raise NotImplementedError("ComplexUnitedScalar.to_hdf5 is not implemented")
    
    @property
    def unit(self) -> Unit:
        raise NotImplementedError("ComplexUnitedScalar.display_unit is not implemented")

    @property
    def active_complex(self) -> complex:
        raise NotImplementedError("ComplexUnitedScalar.active_complex is not implemented")
    
    def __add__(self, other: "ComplexUnitedScalar") -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__add__ is not implemented")
    
    def __radd__(self, other: "ComplexUnitedScalar") -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__radd__ is not implemented")
    
    def __sub__(self, other: "ComplexUnitedScalar") -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__sub__ is not implemented")
    
    def __rsub__(self, other: "ComplexUnitedScalar") -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__rsub__ is not implemented")
    
    def __mul__(self, other: Union["ComplexUnitedScalar", complex]) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__mul__ is not implemented")
    
    def __rmul__(self, other: Union["ComplexUnitedScalar", complex]) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__rmul__ is not implemented")
    
    def __truediv__(self, other: Union["ComplexUnitedScalar", complex]) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__truediv__ is not implemented")
    
    def __rtruediv__(self, other: Union["ComplexUnitedScalar", complex]) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__rtruediv__ is not implemented")
    
    def __pow__(self, exponent: complex) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__pow__ is not implemented")
    
    def __abs__(self) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__abs__ is not implemented")
    
    def __neg__(self) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.__neg__ is not implemented")
    
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.__eq__ is not implemented")
    
    def __ne__(self, other: object) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.__ne__ is not implemented")
    
    def __lt__(self, other: "ComplexUnitedScalar") -> bool:
        raise NotImplementedError("ComplexUnitedScalar.__lt__ is not implemented")
    
    def __gt__(self, other: "ComplexUnitedScalar") -> bool:
        raise NotImplementedError("ComplexUnitedScalar.__gt__ is not implemented")
    
    def __le__(self, other: "ComplexUnitedScalar") -> bool:
        raise NotImplementedError("ComplexUnitedScalar.__le__ is not implemented")
    
    def __ge__(self, other: "ComplexUnitedScalar") -> bool:
        raise NotImplementedError("ComplexUnitedScalar.__ge__ is not implemented")
    
    def __hash__(self) -> int:
        raise NotImplementedError("ComplexUnitedScalar.__hash__ is not implemented")
    
    def __str__(self) -> str:
        raise NotImplementedError("ComplexUnitedScalar.__str__ is not implemented")
    
    def __repr__(self) -> str:
        raise NotImplementedError("ComplexUnitedScalar.__repr__ is not implemented")
    
    def __format__(self, format_spec: str) -> str:
        raise NotImplementedError("ComplexUnitedScalar.__format__ is not implemented")
    
    def is_nan(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_nan is not implemented")
    
    def is_positive(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_positive is not implemented")
    
    def is_negative(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_negative is not implemented")
    
    def is_zero(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_zero is not implemented")
    
    def is_non_zero(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_non_zero is not implemented")
    
    def is_finite(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_finite is not implemented")
    
    def is_infinite(self) -> bool:
        raise NotImplementedError("ComplexUnitedScalar.is_infinite is not implemented")
    
    def compatible_to(self, *args: "ComplexUnitedScalar") -> bool:
        if len(args) == 0:
            return True
        if len(args) == 1:
            return self.dimension == args[0].dimension
        for arg in args:
            if arg.dimension != self.dimension:
                return False
        return True
    
    def scalar_in_canonical_unit(self) -> "ComplexUnitedScalar":
        return ComplexUnitedScalar(
            self.canonical_value,
            self.dimension,
            self.dimension.canonical_unit)
    
    def scalar_in_unit(self, unit: Unit) -> "ComplexUnitedScalar":
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        return ComplexUnitedScalar(
            self.canonical_value,
            self.dimension,
            unit)