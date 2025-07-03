from dataclasses import dataclass
from typing import cast
from .united_array import UnitedArray






@dataclass(frozen=True, slots=True)
class SimpleUnitedArray(UnitedArray):

    def __getitem__(self, index_key: int|slice) -> "SimpleUnitedArray":
        return cast(SimpleUnitedArray, super().__getitem__(index_key))