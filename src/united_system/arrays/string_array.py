from dataclasses import dataclass
from typing import Iterator, Any
import h5py
from .utils import ArrayLike

@dataclass(frozen=True, slots=True)
class StringArray(ArrayLike[str]):
    text_array: tuple[str, ...]
    
    def __getitem__(self, index_key: int|slice) -> str:
        return self.text_array[index_key]
    
    def get_text(self, index: int) -> str:
        return self.text_array[index]
    
    def __len__(self) -> int:
        return len(self.text_array)
    
    def __iter__(self) -> Iterator[str]:
        return iter(self.text_array)
    
    def __next__(self) -> str:
        return next(self.text_array)
    
    def __contains__(self, item: str) -> bool:
        return item in self.text_array
    
    def to_json(self) -> dict[str, Any]:
        return {"text_array": self.text_array}
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "StringArray":
        return StringArray(text_array=json["text_array"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("text_array", data=self.text_array)
    
    @staticmethod
    def from_hdf5(hdf5_group: h5py.Group) -> "StringArray":
        return StringArray(text_array=hdf5_group["text_array"][()])