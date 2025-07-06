from dataclasses import dataclass
from .base_classes.non_united_array import NonUnitedArray

@dataclass(frozen=True, slots=True)
class BoolArray(NonUnitedArray[bool, "BoolArray"]):
    pass