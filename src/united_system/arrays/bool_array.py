from .base_classes.non_united_array import NonUnitedArray
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class BoolArray(NonUnitedArray[bool, "BoolArray"]):
    pass