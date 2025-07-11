from dataclasses import dataclass, field
from typing import Union, Optional, Any, Sized
import numpy as np
import h5py
from ..united_system.utils.general import JSONable, HDF5able
from ..united_system.int_array import IntArray as InternalIntArray

@dataclass(frozen=True, slots=True, init=False)
class IntArray(JSONable, HDF5able, Sized):
    """User-friendly wrapper for integer arrays with convenient methods."""

    _wrapped_array: InternalIntArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray[Any, Any], list[Any], InternalIntArray]):
        """
        Create an IntArray from various input formats.
        
        Examples:
            IntArray([1, 2, 3, 4])              # List of integers
            IntArray(np.array([1, 2, 3]))       # Numpy array of integers
            IntArray(existing_array)            # Wrap existing InternalIntArray
        """
        
        match values:
            case InternalIntArray():
                object.__setattr__(self, "_wrapped_array", values)
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=int)
                if values.ndim != 1:
                    raise ValueError("IntArray must be 1-dimensional")
                object.__setattr__(self, "_wrapped_array", InternalIntArray.create(values))

    # Properties
    @property
    def values(self) -> np.ndarray:
        """Get the raw numpy array values."""
        return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self._wrapped_array.size

    # Array access
    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[int, "IntArray"]:
        """Get array element or slice."""
        result: Union[int, InternalIntArray] = self._wrapped_array[index_or_slice]
        if isinstance(result, (int, np.integer)):
            return int(result)
        else:
            return IntArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)
    
    def __contains__(self, item: int) -> bool:
        """Check if an integer is in the array."""
        return item in self._wrapped_array

    # Iteration
    def __iter__(self):
        """Iterate over the array elements."""
        return iter(self._wrapped_array)

    # Arithmetic operations (element-wise)
    def __add__(self, other: Union["IntArray", int]) -> "IntArray":
        """Add arrays or scalars."""
        if isinstance(other, IntArray):
            return IntArray(self.values + other.values)
        else:
            return IntArray(self.values + other)
    
    def __radd__(self, other: int) -> "IntArray":
        """Reverse addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union["IntArray", int]) -> "IntArray":
        """Subtract arrays or scalars."""
        if isinstance(other, IntArray):
            return IntArray(self.values - other.values)
        else:
            return IntArray(self.values - other)
    
    def __rsub__(self, other: int) -> "IntArray":
        """Reverse subtraction."""
        return IntArray(other - self.values)
    
    def __mul__(self, other: Union["IntArray", int]) -> "IntArray":
        """Multiply arrays or scalars."""
        if isinstance(other, IntArray):
            return IntArray(self.values * other.values)
        else:
            return IntArray(self.values * other)
    
    def __rmul__(self, other: int) -> "IntArray":
        """Reverse multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union["IntArray", int]) -> "IntArray":
        """Divide arrays or scalars (returns integers via floor division)."""
        if isinstance(other, IntArray):
            return IntArray(self.values // other.values)
        else:
            return IntArray(self.values // other)
    
    def __rtruediv__(self, other: int) -> "IntArray":
        """Reverse division."""
        return IntArray(other // self.values)
    
    def __pow__(self, exponent: int) -> "IntArray":
        """Raise array to a power."""
        return IntArray(self.values ** exponent)
    
    def __neg__(self) -> "IntArray":
        """Negate the array."""
        return IntArray(-self.values)
    
    def __abs__(self) -> "IntArray":
        """Get absolute value of the array."""
        return IntArray(np.abs(self.values))

    # Array reduction methods
    def sum(self) -> int:
        """Sum of all elements."""
        return int(np.sum(self.values))
    
    def mean(self) -> float:
        """Mean of all elements."""
        return float(np.mean(self.values))
    
    def max(self) -> int:
        """Maximum value."""
        return int(np.max(self.values))
    
    def min(self) -> int:
        """Minimum value."""
        return int(np.min(self.values))
    
    def std(self) -> float:
        """Standard deviation."""
        return float(np.std(self.values))
    
    def var(self) -> float:
        """Variance."""
        return float(np.var(self.values))

    # String representation
    def __str__(self) -> str:
        """Simple string representation."""
        return str(self.values)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"IntArray({self.values.tolist()!r})"

    # Factory methods
    @classmethod
    def from_values(cls, values: Union[np.ndarray[Any, Any], list[Any]]) -> "IntArray":
        """Create an IntArray from values."""
        return cls(values)
    
    @classmethod
    def zeros(cls, shape: Union[int, tuple[int, ...]]) -> "IntArray":
        """Create an array of zeros."""
        return cls(np.zeros(shape, dtype=int))
    
    @classmethod
    def ones(cls, shape: Union[int, tuple[int, ...]]) -> "IntArray":
        """Create an array of ones."""
        return cls(np.ones(shape, dtype=int))
    
    @classmethod
    def full(cls, shape: Union[int, tuple[int, ...]], fill_value: int) -> "IntArray":
        """Create an array filled with a specific value."""
        return cls(np.full(shape, fill_value, dtype=int))
    
    @classmethod
    def arange(cls, start: int, stop: Optional[int] = None, step: int = 1) -> "IntArray":
        """Create an array with evenly spaced values."""
        if stop is None:
            stop = start
            start = 0
        return cls(np.arange(start, stop, step, dtype=int))

    # Utility methods
    def count(self, value: int) -> int:
        """Count occurrences of a value."""
        return int(np.sum(self.values == value))
    
    def unique(self) -> "IntArray":
        """Get unique values."""
        return IntArray(np.unique(self.values))
    
    def sort(self) -> "IntArray":
        """Return a sorted copy of the array."""
        return IntArray(np.sort(self.values))
    
    def concatenate(self, other: "IntArray") -> "IntArray":
        """Concatenate with another IntArray."""
        combined = np.concatenate([self.values, other.values])
        return IntArray(combined)

    # Comparison methods
    def __eq__(self, other: object) -> bool:
        """Element-wise equality comparison."""
        if isinstance(other, IntArray):
            return self.values == other.values
        else:
            return self.values == other
    
    def __ne__(self, other: object) -> bool:
        """Element-wise inequality comparison."""
        return not self.__eq__(other)
    
    def __lt__(self, other: Union["IntArray", int]) -> np.ndarray:
        """Element-wise less than comparison."""
        if isinstance(other, IntArray):
            return self.values < other.values
        else:
            return self.values < other
    
    def __le__(self, other: Union["IntArray", int]) -> np.ndarray:
        """Element-wise less than or equal comparison."""
        if isinstance(other, IntArray):
            return self.values <= other.values
        else:
            return self.values <= other
    
    def __gt__(self, other: Union["IntArray", int]) -> np.ndarray:
        """Element-wise greater than comparison."""
        if isinstance(other, IntArray):
            return self.values > other.values
        else:
            return self.values > other
    
    def __ge__(self, other: Union["IntArray", int]) -> np.ndarray:
        """Element-wise greater than or equal comparison."""
        if isinstance(other, IntArray):
            return self.values >= other.values
        else:
            return self.values >= other

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict[str, Any], **_: Any) -> "IntArray":
        """Create from JSON dictionary."""
        return cls(InternalIntArray.from_json(data)) # type: ignore
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **_: Any) -> "IntArray":
        """Create from HDF5 group."""
        return cls(InternalIntArray.from_hdf5(hdf5_group)) # type: ignore


