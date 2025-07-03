from dataclasses import dataclass
from typing import cast
from .united_array import UnitedArray
from ..scalars.real_united_scalar import RealUnitedScalar
from enum import Enum
import numpy as np

@dataclass(frozen=True, slots=True)
class RealUnitedArray(UnitedArray[RealUnitedScalar]):

    def __getitem__(self, index_key: int|slice) -> "RealUnitedArray":
        return cast(RealUnitedArray, super().__getitem__(index_key))
    
    def get_float(self, index: int) -> float:
        return self.canonical_np_array[index]