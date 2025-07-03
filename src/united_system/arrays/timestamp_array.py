from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike
from pandas import Timestamp

@dataclass(frozen=True, slots=True)
class TimestampArray(ArrayLike[Timestamp]):
    timestamp_array: tuple[Timestamp, ...]

    def __getitem__(self, index_key: int|slice) -> Timestamp:
        return self.datetime_array[index_key]
    
    def get_timestamp(self, index: int) -> Timestamp:
        return self.timestamp_array[index]
    
    def __len__(self) -> int:
        return len(self.datetime_array)
    
    def __iter__(self) -> Iterator[Timestamp]:
        return iter(self.datetime_array)
    
    def __next__(self) -> Timestamp:
        return next(self.datetime_array)
    
    def __contains__(self, item: Timestamp) -> bool:
        return item in self.timestamp_array
    
    def to_json(self) -> dict[str, Any]:
        return {"datetime_array": self.datetime_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "TimestampArray":
        return TimestampArray(timestamp_array=json["timestamp_array"])