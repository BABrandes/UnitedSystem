from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike
from pandas import Timestamp
import pandas as pd

@dataclass(frozen=True, slots=True)
class TimestampArray(ArrayLike[Timestamp]):
    timestamp_array: tuple[Timestamp, ...]

    def __getitem__(self, index_key: int|slice) -> Timestamp:
        return self.timestamp_array[index_key]
    
    def get_timestamp(self, index: int) -> Timestamp:
        return self.timestamp_array[index]
    
    def __len__(self) -> int:
        return len(self.timestamp_array)
    
    def __iter__(self) -> Iterator[Timestamp]:
        return iter(self.timestamp_array)
    
    def __next__(self) -> Timestamp:
        return next(self.timestamp_array)
    
    def __contains__(self, item: Timestamp) -> bool:
        return item in self.timestamp_array
    
    def to_json(self) -> dict[str, Any]:
        return {"timestamp_array": self.timestamp_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "TimestampArray":
        return TimestampArray(timestamp_array=json["timestamp_array"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("timestamp_array", data=self.timestamp_array)
    
    @staticmethod
    def from_hdf5(hdf5_group: h5py.Group) -> "TimestampArray":
        return TimestampArray(timestamp_array=tuple(hdf5_group["timestamp_array"][()]))
    
    @classmethod
    def create(cls, values: pd.Series) -> "TimestampArray":
        return cls(timestamp_array=tuple(values.astype(Timestamp)))