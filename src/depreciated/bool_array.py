from dataclasses import dataclass, field
from typing import Union, Any, Sized
import numpy as np
import h5py
from ..united_system.utils.general import JSONable, HDF5able
from ..united_system.bool_array import BoolArray as InternalBoolArray

@dataclass(frozen=True, slots=True, init=False)
class BoolArray(JSONable, HDF5able, Sized):
    """User-friendly wrapper for boolean arrays with convenient boolean operations."""

    _wrapped_array: InternalBoolArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray[Any, Any], list[bool], InternalBoolArray]):
        """
        Create a BoolArray from various input formats.
        
        Examples:
            BoolArray([True, False, True])       # List of booleans
            BoolArray(np.array([True, False]))   # Numpy array of booleans
            BoolArray(existing_array)           # Wrap existing InternalBoolArray
        """
        
        match values:
            case InternalBoolArray():
                object.__setattr__(self, "_wrapped_array", values)
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=bool)
                object.__setattr__(self, "_wrapped_array", InternalBoolArray.create(values))

    # Properties
    @property
    def values(self) -> np.ndarray:
        """Get the raw numpy array values."""
        return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        return self._wrapped_array.shape
    
    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self._wrapped_array.size

    # Array access
    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[bool, "BoolArray"]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
        else:
            return BoolArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)
    
    def __contains__(self, item: bool) -> bool:
        """Check if a boolean is in the array."""
        return item in self._wrapped_array

    # Iteration
    def __iter__(self):
        """Iterate over the array elements."""
        return iter(self._wrapped_array)

    # Logical operations
    def __and__(self, other: Union["BoolArray", bool, np.ndarray]) -> "BoolArray":
        """Logical AND operation."""
        if isinstance(other, BoolArray):
            result = self.values & other.values
        elif isinstance(other, bool):
            result = self.values & other
        else:
            result = self.values & other
        return BoolArray(result)

    def __or__(self, other: Union["BoolArray", bool, np.ndarray]) -> "BoolArray":
        """Logical OR operation."""
        if isinstance(other, BoolArray):
            result = self.values | other.values
        elif isinstance(other, bool):
            result = self.values | other
        else:
            result = self.values | other
        return BoolArray(result)

    def __xor__(self, other: Union["BoolArray", bool, np.ndarray]) -> "BoolArray":
        """Logical XOR operation."""
        if isinstance(other, BoolArray):
            result = self.values ^ other.values
        elif isinstance(other, bool):
            result = self.values ^ other
        else:
            result = self.values ^ other
        return BoolArray(result)

    def __invert__(self) -> "BoolArray":
        """Logical NOT operation."""
        result = ~self.values
        return BoolArray(result)

    # Comparison operations
    def __eq__(self, other: object) -> bool:
        """Element-wise equality comparison."""
        if isinstance(other, BoolArray):
            result = self.values == other.values
        elif isinstance(other, bool):
            result = self.values == other
        else:
            result = self.values == other
        return result

    def __ne__(self, other: object) -> bool:
        """Element-wise inequality comparison."""
        if isinstance(other, BoolArray):
            result = self.values != other.values
        elif isinstance(other, bool):
            result = self.values != other
        else:
            result = self.values != other
        return result

    # Reduction methods
    def any(self) -> bool:
        """Return True if any element is True."""
        return bool(np.any(self.values))

    def all(self) -> bool:
        """Return True if all elements are True."""
        return bool(np.all(self.values))

    def count_true(self) -> int:
        """Count the number of True values."""
        return int(np.sum(self.values))

    def count_false(self) -> int:
        """Count the number of False values."""
        return int(np.sum(~self.values))

    # Utility methods
    def unique(self) -> "BoolArray":
        """Get unique values in the array."""
        unique_values = np.unique(self.values)
        return BoolArray(unique_values)

    def concatenate(self, other: "BoolArray") -> "BoolArray":
        """Concatenate with another BoolArray."""
        concatenated = np.concatenate([self.values, other.values])
        return BoolArray(concatenated)

    # Factory methods
    @classmethod
    def true(cls, shape: Union[int, tuple[int, ...]]) -> "BoolArray":
        """Create an array of True values."""
        values = np.ones(shape, dtype=bool)
        return cls(values)

    @classmethod
    def false(cls, shape: Union[int, tuple[int, ...]]) -> "BoolArray":
        """Create an array of False values."""
        values = np.zeros(shape, dtype=bool)
        return cls(values)

    @classmethod
    def random(cls, shape: Union[int, tuple[int, ...]], probability: float = 0.5) -> "BoolArray":
        """Create a random boolean array with given probability of True."""
        values = np.random.random(shape) < probability
        return cls(values)

    @classmethod
    def from_values(cls, values: Union[np.ndarray[Any, Any], list[bool]]) -> "BoolArray":
        """Create from values."""
        return cls(values)

    # String representation
    def __str__(self) -> str:
        """Simple string representation."""
        return str(self.values)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"BoolArray({self.values.tolist()!r})"

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict[str, Any], **_: Any) -> "BoolArray":
        """Create from JSON dictionary."""
        return cls(InternalBoolArray.from_json(data)) # type: ignore
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **_: Any) -> "BoolArray":
        """Create from HDF5 group."""
        return cls(InternalBoolArray.from_hdf5(hdf5_group)) # type: ignore