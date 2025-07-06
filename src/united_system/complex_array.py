from dataclasses import dataclass, field
from typing import Union, Optional, Any, Tuple
import numpy as np
import h5py
from .utils import JSONable, HDF5able
from .arrays.complex_array import ComplexArray as InternalComplexArray
from .bool_array import BoolArray
from .float_array import FloatArray

@dataclass(frozen=True, slots=True, init=False)
class ComplexArray(JSONable, HDF5able):
    """User-friendly wrapper for complex arrays with complex number operations."""

    _wrapped_array: InternalComplexArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray, list, InternalComplexArray]):
        """
        Create a ComplexArray from various input formats.
        
        Examples:
            ComplexArray([1+2j, 3+4j, 5+6j])     # List of complex numbers
            ComplexArray(np.array([1+2j, 3+4j])) # Numpy array of complex numbers
            ComplexArray(existing_array)         # Wrap existing InternalComplexArray
        """
        
        match values:
            case InternalComplexArray():
                object.__setattr__(self, "_wrapped_array", values)
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=complex)
                object.__setattr__(self, "_wrapped_array", InternalComplexArray.create(values))
            case _:
                raise ValueError(f"Invalid values type: {type(values)}")

    # Properties
    @property
    def values(self) -> np.ndarray:
        """Get the raw numpy array values."""
        return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the array."""
        return self._wrapped_array.shape
    
    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self._wrapped_array.size

    @property
    def real(self) -> "FloatArray":
        """Get the real part."""
        from .float_array import FloatArray
        result = np.real(self.values)
        return FloatArray(result)

    @property
    def imag(self) -> "FloatArray":
        """Get the imaginary part."""
        from .float_array import FloatArray
        result = np.imag(self.values)
        return FloatArray(result)

    # Array access
    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[complex, "ComplexArray"]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, (complex, np.complexfloating)):
            return complex(result)
        else:
            return ComplexArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)
    
    def __contains__(self, item: complex) -> bool:
        """Check if a complex number is in the array."""
        return item in self._wrapped_array

    # Iteration
    def __iter__(self):
        """Iterate over the array elements."""
        return iter(self._wrapped_array)

    # Arithmetic operations
    def __add__(self, other: Union["ComplexArray", complex, float, int, np.ndarray]) -> "ComplexArray":
        """Addition operation."""
        if isinstance(other, ComplexArray):
            result = self.values + other.values
        elif isinstance(other, (complex, float, int, np.ndarray)):
            result = self.values + other
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        return ComplexArray(result)

    def __radd__(self, other: Union[complex, float, int, np.ndarray]) -> "ComplexArray":
        """Reverse addition."""
        return self.__add__(other)

    def __sub__(self, other: Union["ComplexArray", complex, float, int, np.ndarray]) -> "ComplexArray":
        """Subtraction operation."""
        if isinstance(other, ComplexArray):
            result = self.values - other.values
        elif isinstance(other, (complex, float, int, np.ndarray)):
            result = self.values - other
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return ComplexArray(result)

    def __rsub__(self, other: Union[complex, float, int, np.ndarray]) -> "ComplexArray":
        """Reverse subtraction."""
        if isinstance(other, (complex, float, int, np.ndarray)):
            result = other - self.values
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return ComplexArray(result)

    def __mul__(self, other: Union["ComplexArray", complex, float, int, np.ndarray]) -> "ComplexArray":
        """Multiplication operation."""
        if isinstance(other, ComplexArray):
            result = self.values * other.values
        elif isinstance(other, (complex, float, int, np.ndarray)):
            result = self.values * other
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
        return ComplexArray(result)

    def __rmul__(self, other: Union[complex, float, int, np.ndarray]) -> "ComplexArray":
        """Reverse multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Union["ComplexArray", complex, float, int, np.ndarray]) -> "ComplexArray":
        """Division operation."""
        if isinstance(other, ComplexArray):
            result = self.values / other.values
        elif isinstance(other, (complex, float, int, np.ndarray)):
            result = self.values / other
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        return ComplexArray(result)

    def __rtruediv__(self, other: Union[complex, float, int, np.ndarray]) -> "ComplexArray":
        """Reverse division."""
        if isinstance(other, (complex, float, int, np.ndarray)):
            result = other / self.values
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        return ComplexArray(result)

    def __pow__(self, exponent: Union[complex, float, int, "ComplexArray", np.ndarray]) -> "ComplexArray":
        """Power operation."""
        if isinstance(exponent, ComplexArray):
            result = self.values ** exponent.values
        elif isinstance(exponent, (complex, float, int, np.ndarray)):
            result = self.values ** exponent
        else:
            raise TypeError(f"Unsupported type for power: {type(exponent)}")
        return ComplexArray(result)

    def __neg__(self) -> "ComplexArray":
        """Unary negation."""
        result = -self.values
        return ComplexArray(result)

    def __abs__(self) -> "FloatArray":
        """Absolute value (magnitude)."""
        from .float_array import FloatArray
        result = np.abs(self.values)
        return FloatArray(result)

    # Comparison operations
    def __eq__(self, other: Union["ComplexArray", complex, np.ndarray]) -> "BoolArray":
        """Element-wise equality comparison."""
        from .bool_array import BoolArray
        if isinstance(other, ComplexArray):
            result = self.values == other.values
        elif isinstance(other, (complex, np.ndarray)):
            result = self.values == other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __ne__(self, other: Union["ComplexArray", complex, np.ndarray]) -> "BoolArray":
        """Element-wise inequality comparison."""
        from .bool_array import BoolArray
        if isinstance(other, ComplexArray):
            result = self.values != other.values
        elif isinstance(other, (complex, np.ndarray)):
            result = self.values != other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    # Complex-specific operations
    def conjugate(self) -> "ComplexArray":
        """Complex conjugate."""
        result = np.conjugate(self.values)
        return ComplexArray(result)

    def conj(self) -> "ComplexArray":
        """Complex conjugate (alias for conjugate)."""
        return self.conjugate()

    def magnitude(self) -> "FloatArray":
        """Magnitude (absolute value)."""
        from .float_array import FloatArray
        result = np.abs(self.values)
        return FloatArray(result)

    def phase(self) -> "FloatArray":
        """Phase angle in radians."""
        from .float_array import FloatArray
        result = np.angle(self.values)
        return FloatArray(result)

    def angle(self) -> "FloatArray":
        """Phase angle in radians (alias for phase)."""
        return self.phase()

    def phase_degrees(self) -> "FloatArray":
        """Phase angle in degrees."""
        from .float_array import FloatArray
        result = np.angle(self.values, deg=True)
        return FloatArray(result)

    def to_polar(self) -> Tuple["FloatArray", "FloatArray"]:
        """Convert to polar coordinates (magnitude, phase)."""
        magnitude = self.magnitude()
        phase = self.phase()
        return magnitude, phase

    def to_rectangular(self) -> Tuple["FloatArray", "FloatArray"]:
        """Convert to rectangular coordinates (real, imaginary)."""
        real = self.real
        imag = self.imag
        return real, imag

    # Mathematical functions
    def sqrt(self) -> "ComplexArray":
        """Square root."""
        result = np.sqrt(self.values)
        return ComplexArray(result)

    def exp(self) -> "ComplexArray":
        """Exponential function."""
        result = np.exp(self.values)
        return ComplexArray(result)

    def log(self) -> "ComplexArray":
        """Natural logarithm."""
        result = np.log(self.values)
        return ComplexArray(result)

    def log10(self) -> "ComplexArray":
        """Base-10 logarithm."""
        result = np.log10(self.values)
        return ComplexArray(result)

    def sin(self) -> "ComplexArray":
        """Sine function."""
        result = np.sin(self.values)
        return ComplexArray(result)

    def cos(self) -> "ComplexArray":
        """Cosine function."""
        result = np.cos(self.values)
        return ComplexArray(result)

    def tan(self) -> "ComplexArray":
        """Tangent function."""
        result = np.tan(self.values)
        return ComplexArray(result)

    def sinh(self) -> "ComplexArray":
        """Hyperbolic sine."""
        result = np.sinh(self.values)
        return ComplexArray(result)

    def cosh(self) -> "ComplexArray":
        """Hyperbolic cosine."""
        result = np.cosh(self.values)
        return ComplexArray(result)

    def tanh(self) -> "ComplexArray":
        """Hyperbolic tangent."""
        result = np.tanh(self.values)
        return ComplexArray(result)

    # Reduction methods
    def sum(self) -> complex:
        """Sum of all elements."""
        return complex(np.sum(self.values))

    def mean(self) -> complex:
        """Mean of all elements."""
        return complex(np.mean(self.values))

    def std(self) -> complex:
        """Standard deviation."""
        return complex(np.std(self.values))

    def var(self) -> complex:
        """Variance."""
        return complex(np.var(self.values))

    def min_magnitude(self) -> complex:
        """Element with minimum magnitude."""
        idx = np.argmin(np.abs(self.values))
        return complex(self.values[idx])

    def max_magnitude(self) -> complex:
        """Element with maximum magnitude."""
        idx = np.argmax(np.abs(self.values))
        return complex(self.values[idx])

    # Utility methods
    def unique(self) -> "ComplexArray":
        """Get unique values in the array."""
        unique_values = np.unique(self.values)
        return ComplexArray(unique_values)

    def concatenate(self, other: "ComplexArray") -> "ComplexArray":
        """Concatenate with another ComplexArray."""
        concatenated = np.concatenate([self.values, other.values])
        return ComplexArray(concatenated)

    def is_real(self) -> "BoolArray":
        """Check if values are real (imaginary part is zero)."""
        from .bool_array import BoolArray
        result = np.isreal(self.values)
        return BoolArray(result)

    def is_imag(self) -> "BoolArray":
        """Check if values are purely imaginary (real part is zero)."""
        from .bool_array import BoolArray
        result = np.real(self.values) == 0
        return BoolArray(result)

    def is_finite(self) -> "BoolArray":
        """Check if values are finite."""
        from .bool_array import BoolArray
        result = np.isfinite(self.values)
        return BoolArray(result)

    def is_nan(self) -> "BoolArray":
        """Check if values are NaN."""
        from .bool_array import BoolArray
        result = np.isnan(self.values)
        return BoolArray(result)

    # Factory methods
    @classmethod
    def zeros(cls, shape: Union[int, tuple]) -> "ComplexArray":
        """Create an array of zeros."""
        values = np.zeros(shape, dtype=complex)
        return cls(values)

    @classmethod
    def ones(cls, shape: Union[int, tuple]) -> "ComplexArray":
        """Create an array of ones."""
        values = np.ones(shape, dtype=complex)
        return cls(values)

    @classmethod
    def full(cls, shape: Union[int, tuple], fill_value: complex) -> "ComplexArray":
        """Create an array filled with a value."""
        values = np.full(shape, fill_value, dtype=complex)
        return cls(values)

    @classmethod
    def from_real_imag(cls, real: Union[np.ndarray, list], imag: Union[np.ndarray, list]) -> "ComplexArray":
        """Create from real and imaginary parts."""
        real_arr = np.asarray(real)
        imag_arr = np.asarray(imag)
        values = real_arr + 1j * imag_arr
        return cls(values)

    @classmethod
    def from_polar(cls, magnitude: Union[np.ndarray, list], phase: Union[np.ndarray, list]) -> "ComplexArray":
        """Create from polar coordinates (magnitude, phase in radians)."""
        magnitude_arr = np.asarray(magnitude)
        phase_arr = np.asarray(phase)
        values = magnitude_arr * np.exp(1j * phase_arr)
        return cls(values)

    @classmethod
    def random(cls, shape: Union[int, tuple]) -> "ComplexArray":
        """Create random complex numbers."""
        real_part = np.random.random(shape)
        imag_part = np.random.random(shape)
        values = real_part + 1j * imag_part
        return cls(values)

    @classmethod
    def from_values(cls, values: Union[np.ndarray, list]) -> "ComplexArray":
        """Create from values."""
        return cls(values)

    # String representation
    def __str__(self) -> str:
        """Simple string representation."""
        return str(self.values)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"ComplexArray({self.values.tolist()!r})"

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "ComplexArray":
        """Create from JSON dictionary."""
        return cls(InternalComplexArray.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "ComplexArray":
        """Create from HDF5 group."""
        return cls(InternalComplexArray.from_hdf5(hdf5_group)) 