from dataclasses import dataclass, field
from typing import Union
import numpy as np
from .utils import JSONable, HDF5able
from .arrays.string_array import StringArray as InternalStringArray

@dataclass(frozen=True, slots=True)
class StringArray(JSONable, HDF5able):

    _wrapped_array: InternalStringArray = field(init=False, hash=False, repr=False, compare=False)

    def __init__(self, values: Union[np.ndarray, list, InternalStringArray]):

        match values:
            case InternalStringArray():
                self._wrapped_array = values
            case np.ndarray() | list():
                if isinstance(values, list):
                    values = np.asarray(values, dtype=np.str_)
                self._wrapped_array = InternalStringArray.create(values)
            case _:
                raise ValueError(f"Invalid values type: {type(values)}")


