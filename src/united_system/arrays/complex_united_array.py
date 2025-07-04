from dataclasses import dataclass
from .united_array import UnitedArray
from ..scalars.complex_united_scalar import ComplexUnitedScalar
from ..units.unit import Unit
from ..units.unit_quantity import UnitQuantity
import numpy as np
from typing import cast

@dataclass(frozen=True, slots=True)
class ComplexUnitedArray(UnitedArray[ComplexUnitedScalar]):

    def __getitem__(self, index_key: int|slice) -> "ComplexUnitedArray":
        return cast(ComplexUnitedArray, super().__getitem__(index_key))

    def get_complex(self, index: int) -> complex:
        return self.canonical_np_array[index]

    @classmethod
    def create(cls, values: np.ndarray, display_unit: Unit|None=None) -> "ComplexUnitedArray":
        unit_quantity: UnitQuantity = UnitQuantity.create(display_unit)
        complex_united_array: ComplexUnitedArray = cls(values, complex, unit_quantity)
        if display_unit is not None:
            complex_united_array.set_display_unit(display_unit)
        return complex_united_array