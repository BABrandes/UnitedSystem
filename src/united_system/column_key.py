from typing import Protocol, Type, runtime_checkable

@runtime_checkable
class ColumnKey(Protocol):
    def to_string(self) -> str:
        ...
    @classmethod
    def from_string(cls: Type['ColumnKey'], string: str) -> 'ColumnKey':
        ...