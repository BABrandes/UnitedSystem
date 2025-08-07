from dataclasses import dataclass
from typing import TYPE_CHECKING, Union
from .._arrays.non_united_array import NonUnitedArray
import numpy as np

if TYPE_CHECKING:
    from .int_array import IntArray

@dataclass(frozen=True, slots=True, init=False)
class FloatArray(NonUnitedArray[float, "FloatArray"]):
    """Array of floats."""
    
    @staticmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        """Check if the array has a valid float dtype."""
        return array.dtype.kind == 'f'  # Floating point
    
    @property
    def value_type(self) -> type[float]:
        return float
    
    @classmethod
    def ones(cls, number_of_ones: int) -> "FloatArray":
        """Create an array of ones."""
        return FloatArray(np.ones(number_of_ones, dtype=float))
    
    @classmethod
    def zeros(cls, number_of_zeros: int) -> "FloatArray":
        """Create an array of zeros."""
        return FloatArray(np.zeros(number_of_zeros, dtype=float))
    
    def __add__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Add two arrays element-wise."""
        return FloatArray(self.canonical_np_array + other.canonical_np_array)
    
    def __radd__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Add two arrays element-wise."""
        return FloatArray(self.canonical_np_array + other.canonical_np_array)
    
    def __sub__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Subtract two arrays element-wise."""
        return FloatArray(self.canonical_np_array - other.canonical_np_array)
    
    def __rsub__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Subtract two arrays element-wise."""
        return FloatArray(other.canonical_np_array - self.canonical_np_array)
    
    def __mul__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Multiply two arrays element-wise."""
        return FloatArray(self.canonical_np_array * other.canonical_np_array)
    
    def __rmul__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Multiply two arrays element-wise."""
        return FloatArray(self.canonical_np_array * other.canonical_np_array)
    
    def __truediv__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Divide two arrays element-wise."""
        return FloatArray(self.canonical_np_array / other.canonical_np_array)
    
    def __rtruediv__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Divide two arrays element-wise."""
        return FloatArray(other.canonical_np_array / self.canonical_np_array)
    
    def __pow__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Raise the array to the power of another array element-wise."""
        return FloatArray(self.canonical_np_array ** other.canonical_np_array)