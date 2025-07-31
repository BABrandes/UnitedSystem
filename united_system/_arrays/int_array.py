from dataclasses import dataclass
from .non_united_array import NonUnitedArray
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class IntArray(NonUnitedArray[int, "IntArray"]):
    """Array of integers."""
    
    @staticmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        """Check if the array has a valid integer dtype."""
        return array.dtype.kind == 'i'  # Signed integer
    
    @property
    def value_type(self) -> type[int]:
        return int