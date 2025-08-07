from dataclasses import dataclass
from typing import TYPE_CHECKING, overload, Union
from .non_united_array import NonUnitedArray
import numpy as np

if TYPE_CHECKING:
    from .real_united_array import RealUnitedArray
    from .._scalars.real_united_scalar import RealUnitedScalar

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
    
    @overload
    def __mul__(self, other: int|float|complex) -> "IntArray":
        ...
    @overload
    def __mul__(self, other: "IntArray") -> "IntArray":
        ...
    @overload
    def __mul__(self, other: "RealUnitedScalar") -> "RealUnitedArray":
        ...
    def __mul__(self, other: Union[int, float, complex, "IntArray", "RealUnitedScalar"]) -> Union["IntArray", "RealUnitedArray"]:
        """Multiply two arrays element-wise."""
        from .real_united_array import RealUnitedArray
        from .._scalars.real_united_scalar import RealUnitedScalar
        if isinstance(other, IntArray):
            return IntArray(self.canonical_np_array * other.canonical_np_array)
        elif isinstance(other, RealUnitedScalar):
            return RealUnitedArray(self.canonical_np_array * other.canonical_value, other.dimension, other.unit)
        elif isinstance(other, float) or isinstance(other, int) or isinstance(other, complex): # type: ignore
            return IntArray(self.canonical_np_array * other)
        else:
            raise ValueError(f"Invalid type: {type(other)}")
    
    @overload
    def __rmul__(self, other: int|float|complex) -> "IntArray":
        ...
    @overload
    def __rmul__(self, other: "IntArray") -> "IntArray":
        ...
    @overload
    def __rmul__(self, other: "RealUnitedScalar") -> "RealUnitedArray":
        ...
    def __rmul__(self, other: Union[int, float, complex, "IntArray", "RealUnitedScalar"]) -> Union["IntArray", "RealUnitedArray"]:
        """Multiply two arrays element-wise."""
        return self.__mul__(other)