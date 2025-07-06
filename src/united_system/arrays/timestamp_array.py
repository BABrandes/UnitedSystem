from dataclasses import dataclass
from .base_classes.non_united_array import NonUnitedArray
from pandas import Timestamp

@dataclass(frozen=True, slots=True)
class TimestampArray(NonUnitedArray[Timestamp, "TimestampArray"]):
    pass