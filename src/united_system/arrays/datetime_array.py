from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike
from datetime import datetime

@dataclass(frozen=True, slots=True)
class DatetimeArray(ArrayLike[datetime]):
    datetime_array: tuple[datetime, ...]

    def __getitem__(self, index_key: int|slice) -> datetime:
        return self.datetime_array[index_key]
    
    def get_datetime(self, index: int) -> datetime:
        return self.datetime_array[index]
    
    def __len__(self) -> int:
        return len(self.datetime_array)
    
    def __iter__(self) -> Iterator[datetime]:
        return iter(self.datetime_array)
    
    def __next__(self) -> datetime:
        return next(self.datetime_array)
    
    def __contains__(self, item: datetime) -> bool:
        return item in self.datetime_array
    
    def to_json(self) -> dict[str, Any]:
        return {"datetime_array": self.datetime_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "DatetimeArray":
        return DatetimeArray(datetime_array=json["datetime_array"])