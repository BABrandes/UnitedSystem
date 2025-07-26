from dataclasses import dataclass
from .._arrays.non_united_array import NonUnitedArray
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class FloatArray(NonUnitedArray[float, "FloatArray"]):
    """Array of floats."""
    
    def _check_numpy_type(self, array: np.ndarray) -> bool:
        """Check if the array has a valid float dtype."""
        return array.dtype.kind == 'f'  # Floating point