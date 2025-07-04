from dataclasses import dataclass
from typing import cast
from .united_array import UnitedArray
from ..scalars.real_united_scalar import RealUnitedScalar
from ..units.unit import Unit
from ..units.unit_quantity import UnitQuantity
from enum import Enum
import numpy as np

@dataclass(frozen=True, slots=True)
class RealUnitedArray(UnitedArray[RealUnitedScalar]):

    def __getitem__(self, index_key: int|slice) -> "RealUnitedArray":
        return cast(RealUnitedArray, super().__getitem__(index_key))
    
    def get_float(self, index: int) -> float:
        return self.canonical_np_array[index]
    
    @classmethod
    def create(cls, values: np.ndarray, display_unit: Unit|None=None) -> "RealUnitedArray":
        unit_quantity: UnitQuantity = UnitQuantity.create(display_unit)
        real_united_array: RealUnitedArray = cls(values, float, unit_quantity)
        if display_unit is not None:
            real_united_array.set_display_unit(display_unit)
        return real_united_array