from dataclasses import dataclass
from .utils.arrays.non_united_array import NonUnitedArray
from pandas import Timestamp

@dataclass(frozen=True, slots=True, init=False)
class TimestampArray(NonUnitedArray[Timestamp, "TimestampArray"]):
    pass