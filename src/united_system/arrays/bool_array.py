from typing import Any, Iterator
from .utils import ArrayLike
from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class BoolArray(ArrayLike[bool]):
    bool_array: tuple[bool, ...]

    def __getitem__(self, index_key: int|slice) -> bool:
        return self.bool_array[index_key]
    
    def get_bool(self, index: int) -> bool:
        return self.bool_array[index]
    
    def __len__(self) -> int:
        return len(self.bool_array)
    
    def __iter__(self) -> Iterator[bool]:
        return iter(self.bool_array)
    
    def __next__(self) -> bool:
        return next(self.bool_array)
    
    def __contains__(self, item: bool) -> bool:
        return item in self.bool_array
    
    def to_json(self) -> dict[str, Any]:
        return {"bool_array": self.bool_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "BoolArray":
        return BoolArray(bool_array=tuple(json["bool_array"]))