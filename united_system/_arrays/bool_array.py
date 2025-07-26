from dataclasses import dataclass
from .._arrays.non_united_array import NonUnitedArray
import numpy as np
@dataclass(frozen=True, slots=True, init=False)
class BoolArray(NonUnitedArray[bool, "BoolArray"]):
    """Array of booleans."""

    def __and__(self, other: "BoolArray") -> "BoolArray":
        return BoolArray(self.canonical_np_array & other.canonical_np_array)

    def __or__(self, other: "BoolArray") -> "BoolArray":
        return BoolArray(self.canonical_np_array | other.canonical_np_array)

    def __xor__(self, other: "BoolArray") -> "BoolArray":
        return BoolArray(self.canonical_np_array ^ other.canonical_np_array)

    def __invert__(self) -> "BoolArray":
        return BoolArray(~self.canonical_np_array)
    
    def _check_numpy_type(self, array: np.ndarray) -> bool:
        """Check if the array has a valid boolean dtype."""
        return array.dtype.kind == 'b'  # Boolean