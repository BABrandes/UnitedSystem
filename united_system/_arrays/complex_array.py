from dataclasses import dataclass
from .._arrays.non_united_array import NonUnitedArray
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class ComplexArray(NonUnitedArray[complex, "ComplexArray"]):
    """Array of complex numbers."""
    
    @staticmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        """Check if the array has a valid complex dtype."""
        return array.dtype.kind == 'c'  # Complex numbers