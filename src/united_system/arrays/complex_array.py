from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike
import numpy as np
import pandas as pd

@dataclass(frozen=True, slots=True)
class ComplexArray(ArrayLike[complex]):
    complex_array: tuple[complex, ...]

    def __getitem__(self, index_key: int|slice) -> complex:
        return self.complex_array[index_key]
    
    def get_complex(self, index: int) -> complex:
        return self.complex_array[index]
    
    def __len__(self) -> int:
        return len(self.complex_array)
    
    def __iter__(self) -> Iterator[complex]:
        return iter(self.complex_array)
    
    def __next__(self) -> complex:
        return next(self.complex_array)
    
    def __contains__(self, item: complex) -> bool:
        return item in self.complex_array
    
    def to_json(self) -> dict[str, Any]:
        return {"complex_array": self.complex_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "ComplexArray":
        return ComplexArray(complex_array=json["complex_array"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("float_array", data=self.float_array)
    
    @staticmethod
    def from_hdf5(hdf5_group: h5py.Group) -> "ComplexArray":
        return ComplexArray(complex_array=hdf5_group["complex_array"][()])
    
    @classmethod
    def create(cls, values: np.ndarray|pd.Series) -> "ComplexArray":
        if isinstance(values, np.ndarray):
            return cls(complex_array=tuple(values.astype(complex)))
        elif isinstance(values, pd.Series):
            return cls(complex_array=tuple(values.astype(complex)))
        else:
            raise ValueError(f"Invalid values type: {type(values)}")