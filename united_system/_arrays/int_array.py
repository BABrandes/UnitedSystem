from dataclasses import dataclass
from .non_united_array import NonUnitedArray
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class IntArray(NonUnitedArray[int, "IntArray"]):
    """Array of integers."""
    
    @staticmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        """Check if the array has a valid integer dtype."""
        return array.dtype.kind == 'i'  # Signed integer
    
    @property
    def value_type(self) -> type[int]:
        return int
    
    @classmethod
    def ones(cls, number_of_ones: int) -> "IntArray":
        """Create an array of ones."""
        return IntArray(np.ones(number_of_ones, dtype=int))
    
    @classmethod
    def zeros(cls, number_of_zeros: int) -> "IntArray":
        """Create an array of zeros."""
        return IntArray(np.zeros(number_of_zeros, dtype=int))
    
    def __add__(self, other: "IntArray") -> "IntArray":
        """Add two arrays element-wise."""
        return IntArray(self.canonical_np_array + other.canonical_np_array)
    
    def __radd__(self, other: "IntArray") -> "IntArray":
        """Add two arrays element-wise."""
        return IntArray(self.canonical_np_array + other.canonical_np_array)
    
    def __sub__(self, other: "IntArray") -> "IntArray":
        """Subtract two arrays element-wise."""
        return IntArray(self.canonical_np_array - other.canonical_np_array)
    
    def __rsub__(self, other: "IntArray") -> "IntArray":
        """Subtract two arrays element-wise."""
        return IntArray(other.canonical_np_array - self.canonical_np_array)
    
    def __mul__(self, other: "IntArray") -> "IntArray":
        """Multiply two arrays element-wise."""
        return IntArray(self.canonical_np_array * other.canonical_np_array)
    
    def __rmul__(self, other: "IntArray") -> "IntArray":
        """Multiply two arrays element-wise."""
        return IntArray(self.canonical_np_array * other.canonical_np_array)