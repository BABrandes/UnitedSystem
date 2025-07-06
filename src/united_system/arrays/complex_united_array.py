from dataclasses import dataclass, field
from .base_classes.base_united_array.base_united_array import BaseUnitedArray
from ..scalars.complex_united_scalar.complex_united_scalar import ComplexUnitedScalar
from ..units.simple.simple_unit import SimpleUnit
from ..units.simple.simple_dimension import SimpleDimension

@dataclass(frozen=True, slots=True)
class ComplexUnitedArray(BaseUnitedArray["ComplexUnitedArray", ComplexUnitedScalar, SimpleUnit, SimpleDimension, complex]):
    
    def get_scalar(self, index: int) -> ComplexUnitedScalar:
        canonical_value: complex = self.canonical_np_array[index]
        dimension: SimpleDimension = self.dimension
        display_unit: SimpleUnit|None = self.display_unit
        return ComplexUnitedScalar(
            canonical_value=canonical_value,
            display_unit=display_unit,
            dimension=dimension)