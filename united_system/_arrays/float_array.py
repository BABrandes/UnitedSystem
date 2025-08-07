from dataclasses import dataclass
from typing import TYPE_CHECKING, overload, Union
from .._arrays.non_united_array import NonUnitedArray
import numpy as np

if TYPE_CHECKING:
    from .int_array import IntArray
    from .._scalars.real_united_scalar import RealUnitedScalar
    from .real_united_array import RealUnitedArray

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
    
    @overload
    def __mul__(self, other: int|float|complex) -> "FloatArray":
        ...
    @overload
    def __mul__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        ...
    @overload
    def __mul__(self, other: "RealUnitedScalar") -> "RealUnitedArray":
        ...
    def __mul__(self, other: Union[int, float, complex, "FloatArray", "IntArray", "RealUnitedScalar"]) -> Union["FloatArray", "RealUnitedArray"]:
        """Multiply two arrays element-wise."""
        from .int_array import IntArray
        from .real_united_array import RealUnitedArray
        from .._scalars.real_united_scalar import RealUnitedScalar
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
            return FloatArray(self.canonical_np_array * other)
        elif isinstance(other, FloatArray):
            return FloatArray(self.canonical_np_array * other.canonical_np_array)
        elif isinstance(other, IntArray):
            return FloatArray(self.canonical_np_array * other.canonical_np_array)
        elif isinstance(other, RealUnitedScalar): # type: ignore
            from .real_united_array import RealUnitedArray
            return RealUnitedArray(self.canonical_np_array * other.canonical_value, other.dimension, other.unit)
        else:
            raise ValueError(f"Invalid type: {type(other)}")
    
    @overload
    def __rmul__(self, other: int|float|complex) -> "FloatArray":
        ...
    @overload
    def __rmul__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        ...
    @overload
    def __rmul__(self, other: "RealUnitedScalar") -> "RealUnitedArray":
        ...
    def __rmul__(self, other: Union[int, float, complex, "FloatArray", "IntArray", "RealUnitedScalar"]) -> Union["FloatArray", "RealUnitedArray"]:
        """Multiply two arrays element-wise."""
        return self.__mul__(other)
    
    @overload
    def __truediv__(self, other: int|float|complex) -> "FloatArray":
        ...
    @overload
    def __truediv__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        ...
    @overload
    def __truediv__(self, other: "RealUnitedScalar") -> "RealUnitedArray":
        ...
    def __truediv__(self, other: Union[int, float, complex, "FloatArray", "IntArray", "RealUnitedScalar"]) -> Union["FloatArray", "RealUnitedArray"]:
        """Divide two arrays element-wise."""
        from .int_array import IntArray
        from .real_united_array import RealUnitedArray
        from .._scalars.real_united_scalar import RealUnitedScalar
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
            return FloatArray(self.canonical_np_array / other)
        elif isinstance(other, FloatArray):
            return FloatArray(self.canonical_np_array / other.canonical_np_array)
        elif isinstance(other, IntArray):
            return FloatArray(self.canonical_np_array / other.canonical_np_array)
        elif isinstance(other, RealUnitedScalar): # type: ignore
            from .real_united_array import RealUnitedArray
            return RealUnitedArray(self.canonical_np_array / other.canonical_value, other.dimension, other.unit)
        else:
            raise ValueError(f"Invalid type: {type(other)}")
    
    @overload
    def __rtruediv__(self, other: int|float|complex) -> "FloatArray":
        ...
    @overload
    def __rtruediv__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        ...
    @overload
    def __rtruediv__(self, other: "RealUnitedScalar") -> "RealUnitedArray":
        ...
    def __rtruediv__(self, other: Union[int, float, complex, "FloatArray", "IntArray", "RealUnitedScalar"]) -> Union["FloatArray", "RealUnitedArray"]:
        """Divide two arrays element-wise."""
        from .int_array import IntArray
        from .real_united_array import RealUnitedArray
        from .._scalars.real_united_scalar import RealUnitedScalar
        if isinstance(other, float) or isinstance(other, int) or isinstance(other, complex):
            return FloatArray(other / self.canonical_np_array)
        elif isinstance(other, FloatArray):
            return FloatArray(other.canonical_np_array / self.canonical_np_array)
        elif isinstance(other, IntArray):
            return FloatArray(other.canonical_np_array / self.canonical_np_array)
        elif isinstance(other, RealUnitedScalar): # type: ignore
            return RealUnitedArray(other.canonical_value / self.canonical_np_array, other.dimension, other.unit)
        else:
            raise ValueError(f"Invalid type: {type(other)}")
        
    def __pow__(self, other: Union["FloatArray", "IntArray"]) -> "FloatArray":
        """Raise the array to the power of another array element-wise."""
        return FloatArray(self.canonical_np_array ** other.canonical_np_array)