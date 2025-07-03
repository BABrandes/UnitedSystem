import numpy as np
from typing import Type, Protocol, runtime_checkable
from enum import Enum
from typing import Callable


@runtime_checkable
class JSONable(Protocol):
    def to_json(self) -> dict:
        ...
    @classmethod
    def from_json(cls: Type['JSONable'], data: dict) -> 'JSONable':
        ...

class Standard_Shape_Conserving_Filters(Enum):
    value: Callable[[np.ndarray], np.ndarray]

    ONLY_POSITIVE_INCLUDING_ZERO = lambda array: np.where(array >= 0, array, np.nan)
    ONLY_NEGATIVE_INCLUDING_ZERO = lambda array: np.where(array <= 0, array, np.nan)
    ONLY_POSITIVE_EXCLUDING_ZERO = lambda array: np.where(array > 0, array, np.nan)
    ONLY_NEGATIVE_EXCLUDING_ZERO = lambda array: np.where(array < 0, array, np.nan)
    ONLY_NON_ZERO = lambda array: np.where(array != 0, array, np.nan)