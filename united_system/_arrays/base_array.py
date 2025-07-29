from typing import Iterator, Generic, TypeVar, Any, overload, TypeAlias
import numpy as np
from dataclasses import dataclass
from pandas import Timestamp
from abc import ABC, abstractmethod
import h5py
from numpy.typing import NDArray    
from pandas._typing import Dtype
import pandas as pd

PT_TYPE: TypeAlias = float|complex|str|bool|int|Timestamp

PT = TypeVar("PT", bound=PT_TYPE)
IT = TypeVar("IT")
AT = TypeVar("AT", bound="BaseArray[PT_TYPE, PT_TYPE, Any]")

@dataclass(frozen=True, slots=True)
class BaseArray(ABC, Generic[PT, IT, AT]):

    canonical_np_array: NDArray[Any]

    def __post_init__(self) -> None:
        if self.canonical_np_array.ndim != 1:
            raise ValueError(f"The canonical_np_array is not a 1D array. It is a {self.canonical_np_array.ndim}D array.")

    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self.canonical_np_array.size
    
    @property 
    def value_type(self) -> type:
        """Get the value type of the array."""
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