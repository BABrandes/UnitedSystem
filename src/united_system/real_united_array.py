from .unit import Unit
from .utils.arrays.base_united_array import BaseUnitedArray
from .real_united_scalar import RealUnitedScalar
from .dimension import Dimension
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class RealUnitedArray(BaseUnitedArray["RealUnitedArray", RealUnitedScalar, Unit, Dimension, float]):
    """Array of real numbers with units."""

    def __init__(self, canonical_np_array: np.ndarray, dimension: Dimension, display_unit: Optional[Unit] = None):
        object.__setattr__(self, "canonical_np_array", canonical_np_array)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", display_unit)

    def get_as_scalar(self, index: int) -> RealUnitedScalar:
        """Get a single element as a RealUnitedScalar."""
        canonical_value: float = self.canonical_np_array[index]
        return RealUnitedScalar(canonical_value=canonical_value, display_unit=self._display_unit, dimension=self.dimension)

    def _get_scalar_from_value(self, value: float) -> RealUnitedScalar:
        """Create a RealUnitedScalar from a primitive value with this array's dimension and display unit."""
        canonical_value: float = self.display_unit.to_canonical_value(value)
        return RealUnitedScalar(canonical_value=canonical_value, dimension=self.dimension, display_unit=self._display_unit)
    