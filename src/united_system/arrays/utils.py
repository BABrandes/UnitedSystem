from typing import Protocol, runtime_checkable, Type, Any, Iterator, Generic, TypeVar, Union, NamedTuple, Enum, TypeAlias
import numpy as np
from dataclasses import dataclass
from ..scalars.real_united_scalar import RealUnitedScalar
from ..scalars.complex_united_scalar import ComplexUnitedScalar
from datetime import datetime
from ..utils import JSONable, HDF5able

T = TypeVar("T", bound=RealUnitedScalar|ComplexUnitedScalar|int|float|complex|str|datetime)

@runtime_checkable
class ArrayLike(Protocol, JSONable, HDF5able, Generic[T]):
    def __getitem__(self, key: str) -> T:
        ...
    def __len__(self) -> int:
        ...
    def __iter__(self) -> Iterator[T]:
        ...
    def __next__(self) -> T:
        ...
    def __contains__(self, item: T) -> bool:
        ...