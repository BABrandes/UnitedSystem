from dataclasses import dataclass
from .utils.arrays.base_united_array import BaseUnitedArray
from .complex_united_scalar import ComplexUnitedScalar
from .unit import Unit
from .dimension import Dimension
import numpy as np
from typing import Optional

@dataclass(frozen=True, slots=True, init=False)
class ComplexUnitedArray(BaseUnitedArray["ComplexUnitedArray", ComplexUnitedScalar, Unit, Dimension, complex]):
    """Array of complex numbers with units."""
    
    # Required fields from United inheritance
    dimension: Dimension
    _display_unit: Optional[Unit]

    def __init__(self, canonical_np_array: np.ndarray, dimension: Dimension, display_unit: Optional[Unit] = None):
        super().__init__(canonical_np_array, dimension, display_unit)

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", display_unit)
        
    def get_as_scalar(self, index: int) -> ComplexUnitedScalar:
        """Get a single element as a ComplexUnitedScalar."""
        canonical_value = self.canonical_np_array[index]
        return ComplexUnitedScalar(canonical_value=complex(canonical_value), dimension=self.dimension, display_unit=self._display_unit)
    
    def _get_scalar_from_value(self, value: complex) -> ComplexUnitedScalar:
        """Create a ComplexUnitedScalar from a primitive value with this array's dimension and display unit."""
        return ComplexUnitedScalar(canonical_value=complex(value), dimension=self.dimension, display_unit=self._display_unit)
    
    def get_scalar(self, index: int) -> ComplexUnitedScalar:
        """Get a scalar element at the given index (required by BaseArray)."""
        canonical_value: complex = self.canonical_np_array[index]
        return ComplexUnitedScalar(
            canonical_value=canonical_value,
            dimension=self.dimension,
            display_unit=self._display_unit)