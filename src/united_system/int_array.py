from dataclasses import dataclass
from .utils.arrays.non_united_array import NonUnitedArray

@dataclass(frozen=True, slots=True, init=False)
class IntArray(NonUnitedArray[int, "IntArray"]):
    """Array of integers."""