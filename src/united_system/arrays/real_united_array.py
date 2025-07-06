from ..units.simple.simple_unit import SimpleUnit
import numpy as np
from ..arrays.base_classes.base_united_array.base_united_array import BaseUnitedArray
from ..scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ..units.simple.simple_dimension import SimpleDimension
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class RealUnitedArray(BaseUnitedArray["RealUnitedArray", RealUnitedScalar, SimpleUnit, SimpleDimension, float]):
    """Array of real numbers with units."""

    def get_scalar(self, index: int) -> RealUnitedScalar:
        canonical_value: float = self.canonical_np_array[index]
        dimension: SimpleDimension = self.dimension
        display_unit: SimpleUnit|None = self.display_unit
        return RealUnitedScalar(
            canonical_value=canonical_value,
            display_unit=display_unit,
            dimension=dimension)
    