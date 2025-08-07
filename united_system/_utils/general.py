import numpy as np
from typing import Protocol, runtime_checkable, Any, Generic, TypeVar
from enum import Enum
from typing import Callable
import h5py

T = TypeVar("T", covariant=True)

@runtime_checkable
class JSONable(Protocol, Generic[T]):
    """
    Protocol for JSONable objects.
    """

    def to_json(self) -> dict[str, Any]:
        ...
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> T:
        ...

@runtime_checkable
class HDF5able(Protocol, Generic[T]):
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        ...
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> T:
        ...

class Standard_Shape_Conserving_Filters(Enum):
    value: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] # type: ignore

    ONLY_POSITIVE_INCLUDING_ZERO = lambda array: np.where(array >= 0, array, np.nan) # type: ignore
    ONLY_NEGATIVE_INCLUDING_ZERO = lambda array: np.where(array <= 0, array, np.nan) # type: ignore
    ONLY_POSITIVE_EXCLUDING_ZERO = lambda array: np.where(array > 0, array, np.nan) # type: ignore
    ONLY_NEGATIVE_EXCLUDING_ZERO = lambda array: np.where(array < 0, array, np.nan) # type: ignore
    ONLY_NON_ZERO = lambda array: np.where(array != 0, array, np.nan) # type: ignore

def segment_numpy_arrays_by_key_array(key_array: np.ndarray, *data_arrays: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    """
    Splits multiple data arrays into segments, based on contiguous blocks of the same value in key_array.

    Parameters:
    - *data_arrays: multiple numpy arrays of the same length as key_array.
    - key_array: a 1D numpy array of labels, whose contiguous values define segment boundaries.

    Returns:
    - A list of tuples, each containing one segment from each data array, including the key array as the first element (in the same order).
    """
    if not all(len(arr) == len(key_array) for arr in data_arrays):
        raise ValueError("All input arrays must have the same length as the key_array")

    # Find start and end indices of each segment
    change_indices = np.flatnonzero(np.diff(key_array)) + 1
    segment_indices = np.concatenate(([0], change_indices, [len(key_array)]))

    segments: list[tuple[np.ndarray, ...]] = []
    for start, end in zip(segment_indices[:-1], segment_indices[1:]):
        segment = (key_array[start:end],) + tuple(arr[start:end] for arr in data_arrays)
        segments.append(segment)

    return segments