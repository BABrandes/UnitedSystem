from dataclasses import dataclass
from .united_array import UnitedArray
from enum import Enum
import numpy as np

@dataclass(frozen=True, slots=True)
class ComplexUnitedArray(UnitedArray):
    pass