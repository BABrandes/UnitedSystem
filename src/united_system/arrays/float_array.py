from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike
import numpy as np
import pandas as pd

@dataclass(frozen=True, slots=True)
class FloatArray(ArrayLike[float]):
    float_array: tuple[float, ...]

    def __getitem__(self, index_key: int|slice) -> float:
        return self.float_array[index_key]
    
    def get_float(self, index: int) -> float:
        return self.float_array[index]
    
    def __len__(self) -> int:
        return len(self.float_array)
    
    def __iter__(self) -> Iterator[float]:
        return iter(self.float_array)
    
    def __next__(self) -> float:
        return next(self.float_array)
    
    def __contains__(self, item: float) -> bool:
        return item in self.float_array
    
    def to_json(self) -> dict[str, Any]:
        return {"float_array": self.float_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "FloatArray":
        return FloatArray(float_array=json["float_array"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("float_array", data=self.float_array)
    
    @staticmethod
    def from_hdf5(hdf5_group: h5py.Group) -> "FloatArray":
        return FloatArray(float_array=hdf5_group["float_array"][()])
    
    @classmethod
    def create(cls, values: np.ndarray|pd.Series) -> "FloatArray":
        if isinstance(values, np.ndarray):
            return cls(float_array=tuple(values.astype(float)))
        elif isinstance(values, pd.Series):
            return cls(float_array=tuple(values.astype(float)))
        else:
            raise ValueError(f"Invalid values type: {type(values)}")