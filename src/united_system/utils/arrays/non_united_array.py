from dataclasses import dataclass
from typing import Any, TypeVar, Iterator, overload
from .base_array import BaseArray, PT_TYPE
import numpy as np
from pandas._typing import Dtype
import pandas as pd

PT = TypeVar("PT", bound=PT_TYPE)
AT = TypeVar("AT", bound="NonUnitedArray[PT_TYPE, Any]")

@dataclass(frozen=True, slots=True)
class NonUnitedArray(BaseArray[PT, PT, AT]):
    
    @classmethod
    def create(cls, values: np.ndarray) -> "NonUnitedArray[PT_TYPE, Any]":
        return cls(values)
    
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
    
class NonUnitedArrayIterator(Iterator[PT]):
    """Iterator for NonUnitedArray that maintains separate state."""
    
    def __init__(self, array: "NonUnitedArray[PT_TYPE, Any]"):
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