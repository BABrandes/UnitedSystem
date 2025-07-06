from dataclasses import dataclass
from .base_array import BaseArray
from typing import Any, TypeVar, Iterator, Union
import numpy as np

PT = TypeVar("PT", bound=Any)
AT = TypeVar("AT", bound="NonUnitedArray")

@dataclass(frozen=True, slots=True)
class NonUnitedArray(BaseArray[PT, AT]):
    
    @classmethod
    def create(cls, values: np.ndarray) -> AT:
        return cls(values)
    
    def get_scalar(self, index: int) -> PT:
        """Get a scalar element at the given index."""
        return self.canonical_np_array[index].item()
    
    def __getitem__(self, key: Union[int, slice]) -> Union[PT, AT]:
        """Get array element or slice."""
        if isinstance(key, int):
            return self.get_scalar(key)
        elif isinstance(key, slice):
            sliced_array = self.canonical_np_array[key]
            return type(self)(sliced_array)
        else:
            raise TypeError(f"Invalid key type: {type(key)}")
    
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
    
class NonUnitedArrayIterator(Iterator[PT]):
    """Iterator for NonUnitedArray that maintains separate state."""
    
    def __init__(self, array: NonUnitedArray):
        self.array = array
        self.index = 0
    
    def __iter__(self) -> Iterator[PT]:
        return self
    
    def __next__(self) -> PT:
        if self.index >= len(self.array.canonical_np_array):
            raise StopIteration
        result = self.array.get_scalar(self.index)
        self.index += 1
        return result