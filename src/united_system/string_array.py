from dataclasses import dataclass, field
from typing import Union, Optional, Any
import numpy as np
import h5py
from .utils import JSONable, HDF5able
from .arrays.string_array import StringArray as InternalStringArray

@dataclass(frozen=True, slots=True, init=False)
class StringArray(JSONable, HDF5able):
    """User-friendly wrapper for string arrays with convenient methods."""

    _wrapped_array: InternalStringArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray, list, InternalStringArray]):
        """
        Create a StringArray from various input formats.
        
        Examples:
            StringArray(['hello', 'world'])        # List of strings
            StringArray(np.array(['a', 'b']))      # Numpy array of strings
            StringArray(existing_array)            # Wrap existing InternalStringArray
        """
        
        match values:
            case InternalStringArray():
                object.__setattr__(self, "_wrapped_array", values)
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=str)
                object.__setattr__(self, "_wrapped_array", InternalStringArray.create(values))
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
    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[str, "StringArray"]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, str):
            return result
        else:
            return StringArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)
    
    def __contains__(self, item: str) -> bool:
        """Check if a string is in the array."""
        return item in self._wrapped_array

    # Iteration
    def __iter__(self):
        """Iterate over the array elements."""
        return iter(self._wrapped_array)

    # String representation
    def __str__(self) -> str:
        """Simple string representation."""
        return str(self._wrapped_array.canonical_np_array)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"StringArray({self.values.tolist()!r})"

    # Factory methods
    @classmethod
    def from_values(cls, values: Union[np.ndarray, list]) -> "StringArray":
        """Create a StringArray from values."""
        return cls(values)
    
    @classmethod
    def empty(cls, shape: Union[int, tuple]) -> "StringArray":
        """Create an empty string array."""
        return cls(np.empty(shape, dtype=str))
    
    @classmethod
    def full(cls, shape: Union[int, tuple], fill_value: str) -> "StringArray":
        """Create an array filled with a specific string."""
        return cls(np.full(shape, fill_value, dtype=str))

    # Utility methods
    def count(self, value: str) -> int:
        """Count occurrences of a value."""
        return int(np.sum(self.values == value))
    
    def unique(self) -> "StringArray":
        """Get unique values."""
        return StringArray(np.unique(self.values))
    
    def concatenate(self, other: "StringArray") -> "StringArray":
        """Concatenate with another StringArray."""
        combined = np.concatenate([self.values, other.values])
        return StringArray(combined)

    # Serialization
    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "StringArray":
        """Create from JSON dictionary."""
        return cls(InternalStringArray.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "StringArray":
        """Create from HDF5 group."""
        return cls(InternalStringArray.from_hdf5(hdf5_group))


