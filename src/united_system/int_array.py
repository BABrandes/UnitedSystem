from dataclasses import dataclass, field
from typing import Union
import numpy as np
from .utils import JSONable, HDF5able
from .arrays.int_array import IntArray as InternalIntArray

@dataclass(frozen=True, slots=True)
class IntArray(JSONable, HDF5able):

    _wrapped_array: InternalIntArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray, list, InternalIntArray]):

        match values:
            case InternalIntArray():
                self._wrapped_array = values
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=int)
                self._wrapped_array = InternalIntArray.create(values)
            case _:
                raise ValueError(f"Invalid values type: {type(values)}")


