from typing import Protocol, runtime_checkable, TypeVar

T = TypeVar('T', bound='ColumnKey')

@runtime_checkable  
class ColumnKey(Protocol):
    def to_united_dataframe_string(self) -> str:
        ...
    @classmethod
    def from_united_dataframe_string(cls: type[T], string: str) -> T:
        ...