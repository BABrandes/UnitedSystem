from dataclasses import dataclass
from .non_united_array import NonUnitedArray
from pandas import Timestamp
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class TimestampArray(NonUnitedArray[Timestamp, "TimestampArray"]):
    pass
    
    @staticmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        """Check if the array has a valid timestamp dtype."""
        return array.dtype.kind == 'O' or array.dtype.kind == 'M'  # Object or datetime64
    
    @property
    def value_type(self) -> type[Timestamp]:
        return Timestamp