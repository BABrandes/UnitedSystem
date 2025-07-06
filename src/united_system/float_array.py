from dataclasses import dataclass, field
from typing import Union, Optional, Any
import numpy as np
import h5py
from .utils import JSONable, HDF5able
from .arrays.float_array import FloatArray as InternalFloatArray

@dataclass(frozen=True, slots=True, init=False)
class FloatArray(JSONable, HDF5able):
    """User-friendly wrapper for floating-point arrays with arithmetic operations and mathematical functions."""

    _wrapped_array: InternalFloatArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray, list, InternalFloatArray]):
        """
        Create a FloatArray from various input formats.
        
        Examples:
            FloatArray([1.5, 2.7, 3.14])        # List of floats
            FloatArray(np.array([1.0, 2.0]))    # Numpy array of floats
            FloatArray(existing_array)          # Wrap existing InternalFloatArray
        """
        
        match values:
            case InternalFloatArray():
                object.__setattr__(self, "_wrapped_array", values)
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=float)
                object.__setattr__(self, "_wrapped_array", InternalFloatArray.create(values))
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

    # Array access
    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[float, "FloatArray"]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, (float, np.floating)):
            return float(result)
        else:
            return FloatArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)
    
    def __contains__(self, item: float) -> bool:
        """Check if a float is in the array."""
        return item in self._wrapped_array

    # Iteration
    def __iter__(self):
        """Iterate over the array elements."""
        return iter(self._wrapped_array)

    # Arithmetic operations
    def __add__(self, other: Union["FloatArray", float, np.ndarray]) -> "FloatArray":
        """Addition operation."""
        if isinstance(other, FloatArray):
            result = self.values + other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values + other
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")
        return FloatArray(result)

    def __radd__(self, other: Union[float, np.ndarray]) -> "FloatArray":
        """Reverse addition."""
        return self.__add__(other)

    def __sub__(self, other: Union["FloatArray", float, np.ndarray]) -> "FloatArray":
        """Subtraction operation."""
        if isinstance(other, FloatArray):
            result = self.values - other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values - other
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return FloatArray(result)

    def __rsub__(self, other: Union[float, np.ndarray]) -> "FloatArray":
        """Reverse subtraction."""
        if isinstance(other, (float, int, np.ndarray)):
            result = other - self.values
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")
        return FloatArray(result)

    def __mul__(self, other: Union["FloatArray", float, np.ndarray]) -> "FloatArray":
        """Multiplication operation."""
        if isinstance(other, FloatArray):
            result = self.values * other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values * other
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")
        return FloatArray(result)

    def __rmul__(self, other: Union[float, np.ndarray]) -> "FloatArray":
        """Reverse multiplication."""
        return self.__mul__(other)

    def __truediv__(self, other: Union["FloatArray", float, np.ndarray]) -> "FloatArray":
        """Division operation."""
        if isinstance(other, FloatArray):
            result = self.values / other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values / other
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        return FloatArray(result)

    def __rtruediv__(self, other: Union[float, np.ndarray]) -> "FloatArray":
        """Reverse division."""
        if isinstance(other, (float, int, np.ndarray)):
            result = other / self.values
        else:
            raise TypeError(f"Unsupported type for division: {type(other)}")
        return FloatArray(result)

    def __pow__(self, exponent: Union[float, int, "FloatArray", np.ndarray]) -> "FloatArray":
        """Power operation."""
        if isinstance(exponent, FloatArray):
            result = self.values ** exponent.values
        elif isinstance(exponent, (float, int, np.ndarray)):
            result = self.values ** exponent
        else:
            raise TypeError(f"Unsupported type for power: {type(exponent)}")
        return FloatArray(result)

    def __neg__(self) -> "FloatArray":
        """Unary negation."""
        result = -self.values
        return FloatArray(result)

    def __abs__(self) -> "FloatArray":
        """Absolute value."""
        result = np.abs(self.values)
        return FloatArray(result)

    # Comparison operations (return boolean arrays from root BoolArray)
    def __eq__(self, other: Union["FloatArray", float, np.ndarray]) -> "BoolArray":
        """Element-wise equality comparison."""
        from .bool_array import BoolArray
        if isinstance(other, FloatArray):
            result = self.values == other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values == other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __ne__(self, other: Union["FloatArray", float, np.ndarray]) -> "BoolArray":
        """Element-wise inequality comparison."""
        from .bool_array import BoolArray
        if isinstance(other, FloatArray):
            result = self.values != other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values != other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __lt__(self, other: Union["FloatArray", float, np.ndarray]) -> "BoolArray":
        """Element-wise less than comparison."""
        from .bool_array import BoolArray
        if isinstance(other, FloatArray):
            result = self.values < other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values < other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __le__(self, other: Union["FloatArray", float, np.ndarray]) -> "BoolArray":
        """Element-wise less than or equal comparison."""
        from .bool_array import BoolArray
        if isinstance(other, FloatArray):
            result = self.values <= other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values <= other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __gt__(self, other: Union["FloatArray", float, np.ndarray]) -> "BoolArray":
        """Element-wise greater than comparison."""
        from .bool_array import BoolArray
        if isinstance(other, FloatArray):
            result = self.values > other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values > other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    def __ge__(self, other: Union["FloatArray", float, np.ndarray]) -> "BoolArray":
        """Element-wise greater than or equal comparison."""
        from .bool_array import BoolArray
        if isinstance(other, FloatArray):
            result = self.values >= other.values
        elif isinstance(other, (float, int, np.ndarray)):
            result = self.values >= other
        else:
            raise TypeError(f"Unsupported type for comparison: {type(other)}")
        return BoolArray(result)

    # Mathematical functions
    def sqrt(self) -> "FloatArray":
        """Square root."""
        result = np.sqrt(self.values)
        return FloatArray(result)

    def sin(self) -> "FloatArray":
        """Sine function."""
        result = np.sin(self.values)
        return FloatArray(result)

    def cos(self) -> "FloatArray":
        """Cosine function."""
        result = np.cos(self.values)
        return FloatArray(result)

    def tan(self) -> "FloatArray":
        """Tangent function."""
        result = np.tan(self.values)
        return FloatArray(result)

    def exp(self) -> "FloatArray":
        """Exponential function."""
        result = np.exp(self.values)
        return FloatArray(result)

    def log(self) -> "FloatArray":
        """Natural logarithm."""
        result = np.log(self.values)
        return FloatArray(result)

    def log10(self) -> "FloatArray":
        """Base-10 logarithm."""
        result = np.log10(self.values)
        return FloatArray(result)

    def floor(self) -> "FloatArray":
        """Floor function."""
        result = np.floor(self.values)
        return FloatArray(result)

    def ceil(self) -> "FloatArray":
        """Ceiling function."""
        result = np.ceil(self.values)
        return FloatArray(result)

    def round(self, decimals: int = 0) -> "FloatArray":
        """Round to given number of decimals."""
        result = np.round(self.values, decimals)
        return FloatArray(result)

    # Reduction methods
    def sum(self) -> float:
        """Sum of all elements."""
        return float(np.sum(self.values))

    def mean(self) -> float:
        """Mean of all elements."""
        return float(np.mean(self.values))

    def min(self) -> float:
        """Minimum value."""
        return float(np.min(self.values))

    def max(self) -> float:
        """Maximum value."""
        return float(np.max(self.values))

    def std(self) -> float:
        """Standard deviation."""
        return float(np.std(self.values))

    def var(self) -> float:
        """Variance."""
        return float(np.var(self.values))

    def median(self) -> float:
        """Median value."""
        return float(np.median(self.values))

    # Utility methods
    def unique(self) -> "FloatArray":
        """Get unique values in the array."""
        unique_values = np.unique(self.values)
        return FloatArray(unique_values)

    def sort(self, ascending: bool = True) -> "FloatArray":
        """Sort the array."""
        sorted_array = np.sort(self.values)
        if not ascending:
            sorted_array = sorted_array[::-1]
        return FloatArray(sorted_array)

    def concatenate(self, other: "FloatArray") -> "FloatArray":
        """Concatenate with another FloatArray."""
        concatenated = np.concatenate([self.values, other.values])
        return FloatArray(concatenated)

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "FloatArray":
        """Clip values to specified range."""
        result = np.clip(self.values, min_val, max_val)
        return FloatArray(result)

    def is_nan(self) -> "BoolArray":
        """Check for NaN values."""
        from .bool_array import BoolArray
        result = np.isnan(self.values)
        return BoolArray(result)

    def is_finite(self) -> "BoolArray":
        """Check for finite values."""
        from .bool_array import BoolArray
        result = np.isfinite(self.values)
        return BoolArray(result)

    # Factory methods
    @classmethod
    def zeros(cls, shape: Union[int, tuple]) -> "FloatArray":
        """Create an array of zeros."""
        values = np.zeros(shape, dtype=float)
        return cls(values)

    @classmethod
    def ones(cls, shape: Union[int, tuple]) -> "FloatArray":
        """Create an array of ones."""
        values = np.ones(shape, dtype=float)
        return cls(values)

    @classmethod
    def full(cls, shape: Union[int, tuple], fill_value: float) -> "FloatArray":
        """Create an array filled with a value."""
        values = np.full(shape, fill_value, dtype=float)
        return cls(values)

    @classmethod
    def linspace(cls, start: float, stop: float, num: int = 50) -> "FloatArray":
        """Create linearly spaced values."""
        values = np.linspace(start, stop, num)
        return cls(values)

    @classmethod
    def arange(cls, start: float, stop: float, step: float = 1.0) -> "FloatArray":
        """Create values in a range."""
        values = np.arange(start, stop, step)
        return cls(values)

    @classmethod
    def random(cls, shape: Union[int, tuple], low: float = 0.0, high: float = 1.0) -> "FloatArray":
        """Create random values."""
        values = np.random.uniform(low, high, shape)
        return cls(values)

    @classmethod
    def from_values(cls, values: Union[np.ndarray, list]) -> "FloatArray":
        """Create from values."""
        return cls(values)

    # String representation
    def __str__(self) -> str:
        """Simple string representation."""
        return str(self.values)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"FloatArray({self.values.tolist()!r})"

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "FloatArray":
        """Create from JSON dictionary."""
        return cls(InternalFloatArray.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "FloatArray":
        """Create from HDF5 group."""
        return cls(InternalFloatArray.from_hdf5(hdf5_group)) 