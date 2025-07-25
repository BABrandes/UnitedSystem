from dataclasses import dataclass
from .utils.arrays.non_united_array import NonUnitedArray
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class StringArray(NonUnitedArray[str, "StringArray"]):
    """Array of strings."""
    
    def _check_numpy_type(self, array: np.ndarray) -> bool:
        """Check if the array has a valid string dtype."""
        return array.dtype.kind == 'U'  # Unicode strings