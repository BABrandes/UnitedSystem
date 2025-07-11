from dataclasses import dataclass
from .utils.arrays.non_united_array import NonUnitedArray
import numpy as np
import pandas as pd

@dataclass(frozen=True, slots=True, init=False)
class BoolArray(NonUnitedArray[bool, "BoolArray"]):
    """Array of booleans."""

    def __init__(self, array: list[bool]|np.ndarray|pd.Series[bool]):

        if isinstance(array, list):
            bool_array: np.ndarray = np.array(array)
        elif isinstance(array, pd.Series):
            bool_array: np.ndarray = array.to_numpy() # type: ignore

        if bool_array.dtype != np.bool_: # type: ignore
            raise ValueError("Array must be a boolean array")

        super().__init__(bool_array) # type: ignore

    def __and__(self, other: "BoolArray") -> "BoolArray":
        return BoolArray(self.canonical_np_array & other.canonical_np_array)

    def __or__(self, other: "BoolArray") -> "BoolArray":
        return BoolArray(self.canonical_np_array | other.canonical_np_array)

    def __xor__(self, other: "BoolArray") -> "BoolArray":
        return BoolArray(self.canonical_np_array ^ other.canonical_np_array)

    def __invert__(self) -> "BoolArray":
        return BoolArray(~self.canonical_np_array)