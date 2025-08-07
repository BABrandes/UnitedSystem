from typing import Iterator, Generic, TypeVar, Any, overload, Optional
import numpy as np
from dataclasses import dataclass
from abc import ABC, abstractmethod
import h5py  
from pandas._typing import Dtype
import pandas as pd
# Import moved to TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .._arrays.base_united_array import BaseUnitedArray
from .._utils.value_type import VALUE_TYPE
from .._utils.scalar_type import SCALAR_TYPE
from .._utils.array_type import ARRAY_TYPE

PT = TypeVar("PT", bound=VALUE_TYPE)
IT = TypeVar("IT", bound=SCALAR_TYPE)
AT = TypeVar("AT", bound="BaseArray[VALUE_TYPE, SCALAR_TYPE, Any]")

@dataclass(frozen=True, slots=True)
class BaseArray(ABC, Generic[PT, IT, AT]):

    canonical_np_array: np.ndarray

    def __post_init__(self) -> None:
        if self.canonical_np_array.ndim != 1:
            raise ValueError(f"The canonical_np_array is not a 1D array. It is a {self.canonical_np_array.ndim}D array.")

    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self.canonical_np_array.size
    
    @property
    @abstractmethod
    def value_type(self) -> type[VALUE_TYPE]:
        """
        Get the value type of the array:
        Possible return types:
        - float
        - complex
        - str
        - bool
        - int
        - Timestamp
        """
        ...
    
    @property
    def numpy_value_type(self) -> type[Any]:
        """
        Get the NumPy scalar type of the array.

        Returns:
            A NumPy dtype class, such as:
            - numpy.float32, numpy.float64, numpy.float128
            - numpy.int8, numpy.int16, numpy.int32, numpy.int64
            - numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64
            - numpy.complex64, numpy.complex128, numpy.complex256
            - numpy.bool_
            - numpy.str_, numpy.bytes_
            - pandas.Timestamp (not sure if it actually acts as datetime64[ns])

        Note:
            The returned type is a NumPy or domain-specific scalar class, 
            not a native Python type like `float` or `int`.
        """
        return self.canonical_np_array.dtype.type

    @abstractmethod
    def get_value(self, index: int) -> PT:
        ...

    @abstractmethod
    def get_array(self, slice: slice) -> "BaseArray[PT, IT, AT]":
        ...

    @abstractmethod
    def get_numpy_array(self, slice: slice|None = None) -> np.ndarray:
        ...
    
    @abstractmethod
    def get_pandas_series(self, dtype: Dtype, slice: slice|None = None) -> pd.Series: # type: ignore
        """Get the array as a pandas Series."""
        ...
    
    @abstractmethod
    @overload
    def __getitem__(self, key: int) -> IT:
        ...
    @abstractmethod
    @overload
    def __getitem__(self, key: slice) -> AT:
        ...
    @abstractmethod
    def __getitem__(self, key: int|slice) -> IT|AT:
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        ...
    
    @abstractmethod
    def __iter__(self) -> Iterator[PT]:
        ...
    
    @abstractmethod
    def __next__(self) -> PT:
        ...

    @abstractmethod
    def __contains__(self, item: PT) -> bool:
        ...

    def to_json(self) -> dict[str, Any]:
        return {"canonical_np_array": self.canonical_np_array}
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> AT:
        return cls(canonical_np_array=data["canonical_np_array"]) # type: ignore
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("canonical_np_array", data=self.canonical_np_array) # type: ignore

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> AT:
        return cls(canonical_np_array=hdf5_group["canonical_np_array"][()]) # type: ignore
    
    @staticmethod
    @abstractmethod
    def _check_numpy_type(array: np.ndarray) -> bool:
        ...

    def segment_other_arrays(self, *arrays: "BaseArray[Any, Any, Any]") -> list[tuple["BaseArray[Any, Any, Any]", ...]]:
        """
        Use this array as a key array to segment other arrays.

        Args:
            *arrays: The arrays to segment.

        Returns:
            A list of tuples, each containing one segment from each array, including the key array as the first element (in the same order).
        """

        from .._utils.general import segment_numpy_arrays_by_key_array

        segments: list[tuple[np.ndarray, ...]] = segment_numpy_arrays_by_key_array(self.canonical_np_array, *[array.canonical_np_array for array in arrays])
        results_segments: list[tuple["BaseArray[Any, Any, Any]", ...]] = []
        for segment in segments:
            segment_arrays: list["BaseArray[Any, Any, Any]"] = []
            for numpy_array, array in zip(segment, arrays):
                if isinstance(array, BaseUnitedArray):
                    segment_arrays.append(array.__new__(array.__class__, numpy_array, unit=array.unit)) # type: ignore
                else:
                    segment_arrays.append(array.__new__(array.__class__, numpy_array)) # type: ignore
            results_segments.append(tuple(segment_arrays)) # type: ignore
        return results_segments
    
    @classmethod
    @abstractmethod
    def concatenate_arrays(cls, *arrays: AT) -> AT:
        """
        Concatenate multiple arrays into a single array.
        """
        ...

@overload
def create_array(item: VALUE_TYPE|SCALAR_TYPE, size: int) -> ARRAY_TYPE:
    ...
@overload
def create_array(item: VALUE_TYPE|SCALAR_TYPE, size: int, expected_type: Optional[type[AT]]) -> AT:
    ...
def create_array(item: VALUE_TYPE|SCALAR_TYPE, size: int, expected_type: Optional[type[AT]] = None) -> ARRAY_TYPE|AT:
    
    from .._scalars.real_united_scalar import RealUnitedScalar
    from .._scalars.complex_united_scalar import ComplexUnitedScalar
    from pandas import Timestamp

    array: np.ndarray = np.array([item] * size)

    if isinstance(item, bool):
        from .bool_array import BoolArray
        if expected_type is not None and expected_type is not BoolArray:
            raise ValueError(f"Expected type {expected_type} but got {BoolArray}. You can use the create_array function to create an array of a specific type.")
        return BoolArray(array)
    elif isinstance(item, float):
        from .float_array import FloatArray
        if expected_type is not None and expected_type is not FloatArray:
            raise ValueError(f"Expected type {expected_type} but got {FloatArray}. You can use the create_array function to create an array of a specific type.")
        return FloatArray(array)
    elif isinstance(item, int):
        from .int_array import IntArray
        if expected_type is not None and expected_type is not IntArray:
            raise ValueError(f"Expected type {expected_type} but got {IntArray}. You can use the create_array function to create an array of a specific type.")
        return IntArray(array)
    elif isinstance(item, complex):
        from .complex_array import ComplexArray
        if expected_type is not None and expected_type is not ComplexArray:
            raise ValueError(f"Expected type {expected_type} but got {ComplexArray}. You can use the create_array function to create an array of a specific type.")
        return ComplexArray(array)
    elif isinstance(item, str):
        from .string_array import StringArray
        if expected_type is not None and expected_type is not StringArray:
            raise ValueError(f"Expected type {expected_type} but got {StringArray}. You can use the create_array function to create an array of a specific type.")
        return StringArray(array)
    elif isinstance(item, Timestamp):
        from .timestamp_array import TimestampArray
        if expected_type is not None and expected_type is not TimestampArray:
            raise ValueError(f"Expected type {expected_type} but got {TimestampArray}. You can use the create_array function to create an array of a specific type.")
        return TimestampArray(array)
    elif isinstance(item, RealUnitedScalar):
        from .real_united_array import RealUnitedArray
        if expected_type is not None and expected_type is not RealUnitedArray:
            raise ValueError(f"Expected type {expected_type} but got {RealUnitedArray}. You can use the create_array function to create an array of a specific type.")
        return RealUnitedArray(array)
    elif isinstance(item, ComplexUnitedScalar): # type: ignore
        from .complex_united_array import ComplexUnitedArray # type: ignore
        if expected_type is not None and expected_type is not ComplexUnitedArray:
            raise ValueError(f"Expected type {expected_type} but got {ComplexUnitedArray}. You can use the create_array function to create an array of a specific type.")
        raise NotImplementedError("ComplexUnitedArray is not implemented")
    else:
        raise ValueError(f"Invalid type: {type(item)}")