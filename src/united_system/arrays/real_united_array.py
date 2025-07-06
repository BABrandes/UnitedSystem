from ..units.simple.simple_unit import SimpleUnit
import numpy as np
from ..arrays.base_classes.base_united_array.base_united_array import BaseUnitedArray
from ..scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ..units.simple.simple_dimension import SimpleDimension
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class RealUnitedArray(BaseUnitedArray["RealUnitedArray", RealUnitedScalar, SimpleUnit, SimpleDimension, float]):
    """Array of real numbers with units."""

    def get_as_united_value(self, index: int) -> RealUnitedScalar:
        """Get a single element as a RealUnitedScalar."""
        canonical_value = self.canonical_np_array[index]
        return RealUnitedScalar(canonical_value=float(canonical_value), display_unit=self.display_unit, dimension=self.dimension)
    
    def _get_scalar_from_value(self, value: float) -> RealUnitedScalar:
        """Create a RealUnitedScalar from a primitive value with this array's dimension and display unit."""
        return RealUnitedScalar(canonical_value=float(value), dimension=self.dimension, display_unit=self.display_unit)
    
    def get_scalar(self, index: int) -> RealUnitedScalar:
        """Get a scalar element at the given index (required by BaseArray)."""
        canonical_value: float = self.canonical_np_array[index]
        return RealUnitedScalar(
            canonical_value=canonical_value,
            display_unit=self.display_unit,
            dimension=self.dimension)
    