from typing import Any, TypeVar, Iterator, overload, Generic, Sequence
from .base_array import BaseArray
import numpy as np
from pandas._typing import Dtype
import pandas as pd
from .._utils.general import VALUE_TYPE

PT = TypeVar("PT", bound=VALUE_TYPE)
AT = TypeVar("AT", bound="NonUnitedArray[VALUE_TYPE, Any]")

class NonUnitedArray(BaseArray[PT, PT, AT], Generic[PT, AT]):

    def __post_init__(self) -> None:
        """Override BaseArray's __post_init__ to safely handle our custom construction."""
        if hasattr(self, 'canonical_np_array') and hasattr(self.canonical_np_array, 'ndim'):
            if self.canonical_np_array.ndim != 1:
                raise ValueError(f"The canonical_np_array is not a 1D array. It is a {self.canonical_np_array.ndim}D array.")
        # If canonical_np_array isn't properly set yet, skip validation (it will happen in __new__)

    def __new__(cls, values: np.ndarray|Sequence[VALUE_TYPE]) -> AT:
        # Create instance using object.__new__ to avoid inheritance issues
        instance: AT = object.__new__(cls) # type: ignore

        # Process values
        if isinstance(values, np.ndarray):
            values_array: np.ndarray = values
        elif isinstance(values, Sequence): # type: ignore
            values_array: np.ndarray = np.array(values)
        else:
            raise ValueError(f"Invalid values type: {type(values)}")
        
        # Set the canonical_np_array attribute directly BEFORE validation
        object.__setattr__(instance, "canonical_np_array", values_array)
        
        if not instance._check_numpy_type(values_array):
            raise ValueError(f"Array has wrong numpy type: {values_array.dtype}")
        
        # Do the validation inline
        if values_array.ndim != 1:
            raise ValueError(f"The canonical_np_array is not a 1D array. It is a {values_array.ndim}D array.")
        
        return instance
    
    @overload
    def __getitem__(self, key: int) -> PT:...
    @overload
    def __getitem__(self, key: slice) -> AT:...
    def __getitem__(self, key: int|slice) -> PT|AT:
        """Get array element or slice."""
        if isinstance(key, int):
            return self.get_value(key)
        else:
            sliced_array: np.ndarray = self.canonical_np_array[key]
            return type(self)(sliced_array) # type: ignore
    
    def get_numpy_array(self, slice: slice|None = None) -> np.ndarray:
        """Get the numpy array.
        
        If slice is provided, the array is returned as a slice of the original array.
        """
        if slice is None:
            return self.canonical_np_array
        else:
            return self.canonical_np_array[slice]
    
    def get_pandas_series(self, dtype: Dtype, slice: slice|None = None) -> pd.Series: # type: ignore
        """Get the array as a pandas Series.
        
        If slice is provided, the array is returned as a slice of the original array.
        """
        if slice is None:
            return pd.Series(self.canonical_np_array, dtype=dtype)
        else:
            return pd.Series(self.canonical_np_array[slice], dtype=dtype)

    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self.canonical_np_array)
    
    def __contains__(self, item: PT) -> bool:
        """Check if an item is in the array."""
        return item in self.canonical_np_array
    
    def __iter__(self) -> Iterator[PT]:
        return NonUnitedArrayIterator(self)
    
    def __next__(self) -> PT:
        raise TypeError("NonUnitedArray object is not an iterator")
    
    def get_value(self, index: int) -> PT:
        return self.canonical_np_array[index]

    def get_array(self, slice: slice) -> AT:  # type: ignore
        return type(self)(self.canonical_np_array[slice])  # type: ignore
    
    @classmethod
    def concatenate_arrays(cls, *arrays: AT) -> AT:
        """
        Concatenate multiple arrays into a single array.
        """
        return cls(canonical_np_array=np.concatenate([array.canonical_np_array for array in arrays])) # type: ignore
    
class NonUnitedArrayIterator(Iterator[PT]):
    """Iterator for NonUnitedArray that maintains separate state."""
    
    def __init__(self, array: "NonUnitedArray[VALUE_TYPE, Any]"):
        self.array = array
        self.index = 0
    
    def __iter__(self) -> Iterator[PT]:
        return self
    
    def __next__(self) -> PT:
        if self.index >= len(self.array.canonical_np_array):
            raise StopIteration
        result: PT = self.array.get_value(self.index) # type: ignore
        self.index += 1
        return result