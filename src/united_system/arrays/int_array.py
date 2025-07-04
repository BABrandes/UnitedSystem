from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike
import numpy as np
import pandas as pd

@dataclass(frozen=True, slots=True)
class IntArray(ArrayLike[int]):
    int_array: tuple[int, ...]

    def __getitem__(self, index_key: int|slice) -> int:
        return self.int_array[index_key]
    
    def get_int(self, index: int) -> int:
        return self.int_array[index]
    
    def __len__(self) -> int:
        return len(self.int_array)
    
    def __iter__(self) -> Iterator[int]:
        return iter(self.int_array)
    
    def __next__(self) -> int:
        return next(self.int_array)
    
    def __contains__(self, item: int) -> bool:
        return item in self.int_array
    
    def to_json(self) -> dict[str, Any]:
        return {"int_array": self.int_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "IntArray":
        return IntArray(int_array=json["int_array"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("int_array", data=self.int_array)
    
    @staticmethod
    def from_hdf5(hdf5_group: h5py.Group) -> "IntArray":
        return IntArray(int_array=hdf5_group["int_array"][()])
    
    @classmethod
    def create(cls, values: np.ndarray|pd.Series) -> "IntArray":
        if isinstance(values, np.ndarray):
            return cls(int_array=tuple(values.astype(int)))
        elif isinstance(values, pd.Series):
            return cls(int_array=tuple(values.astype(int)))
        else:
            raise ValueError(f"Invalid values type: {type(values)}")