from .base_united_array import BaseUnitedArray
from .._scalars.real_united_scalar import RealUnitedScalar
from .._units_and_dimension.dimension import Dimension
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class RealUnitedArray(BaseUnitedArray["RealUnitedArray", RealUnitedScalar, float]):
    """Array of real numbers with units."""

    def get_as_scalar(self, index: int) -> RealUnitedScalar:
        """Get a single element as a RealUnitedScalar."""
        canonical_value: float = self.canonical_np_array[index]
        return RealUnitedScalar.create_from_canonical_value(canonical_value, self.dimension, self._display_unit)

    def get_scalar_from_value(self, value: float) -> RealUnitedScalar:
        """Create a RealUnitedScalar from a primitive value with this array's dimension and display unit."""
        canonical_value: float = self.unit.to_canonical_value(value) # type: ignore
        return RealUnitedScalar.create_from_canonical_value(canonical_value, self.dimension, self._display_unit)
    
    def get_numpy_array(self, slice: slice|None = None) -> np.ndarray:
        """Get the numpy array in the display unit.
        
        If slice is provided, the array is returned as a slice of the original array.
        If no slice is provided, the array is returned as a copy of the original array.
        """
        if slice is None:
            return self.unit.from_canonical_value(self.canonical_np_array) # type: ignore
        else:
            return self.unit.from_canonical_value(self.canonical_np_array[slice]) # type: ignore
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        return self.canonical_np_array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get the dtype of the array."""
        return self.canonical_np_array.dtype
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the array."""
        return self.canonical_np_array.ndim
    
    @property
    def active_dimension(self) -> Dimension:
        """Get the active dimension of the array."""
        return self.dimension
    
    @staticmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        """Check if the array has a valid real dtype."""
        return array.dtype.kind in ['i', 'f']  # Real numbers
    