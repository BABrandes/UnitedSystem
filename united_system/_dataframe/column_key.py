from typing import Protocol, runtime_checkable, Self

@runtime_checkable  
class ColumnKey(Protocol):
    def to_united_dataframe_string(self) -> str:
        ...
    @classmethod
    def from_united_dataframe_string(cls, string: str) -> Self:
        ...