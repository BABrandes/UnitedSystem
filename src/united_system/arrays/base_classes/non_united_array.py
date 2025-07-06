from dataclasses import dataclass
from .base_array import BaseArray
from typing import Any, TypeVar
import numpy as np

PT = TypeVar("PT", bound=Any)
AT = TypeVar("AT", bound="NonUnitedArray")

@dataclass(frozen=True, slots=True)
class NonUnitedArray(BaseArray[PT, AT]):
    
    @classmethod
    def create(cls, values: np.ndarray) -> AT:
        return cls(values)