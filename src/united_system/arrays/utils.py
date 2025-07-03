from typing import Protocol, runtime_checkable, Type, Any, Iterator, Generic, TypeVar, Union, NamedTuple, Enum, TypeAlias
import numpy as np
from dataclasses import dataclass
from ..scalars.real_united_scalar import RealUnitedScalar
from ..scalars.complex_united_scalar import ComplexUnitedScalar
from datetime import datetime
from ..utils import JSONable, HDF5able

RealUnitedArrayTypes: TypeAlias = Union[np.float64, np.float32, np.float16,]
@dataclass(frozen=True, slots=True)
class RealUnitedArrayValueType_Information(NamedTuple):
    name: str
    dtype: np.dtype
    simple_united_array_types: Type[RealUnitedArrayTypes]
    precision: int|None
class SimpleUnitedArrayValueType(Enum):
    value: RealUnitedArrayValueType_Information
FLOAT64 = RealUnitedArrayValueType_Information(name="float64", dtype=np.dtype("float64"), simple_united_array_types=np.float64, precision=64)
FLOAT32 = RealUnitedArrayValueType_Information(name="float32", dtype=np.dtype("float32"), simple_united_array_types=np.float32, precision=32)
FLOAT16 = RealUnitedArrayValueType_Information(name="float16", dtype=np.dtype("float16"), simple_united_array_types=np.float16, precision=16)

ComplexUnitedArrayTypes: TypeAlias = Union[np.complex64, np.complex128,]
@dataclass(frozen=True, slots=True)
class ComplexUnitedArrayValueType_Information(NamedTuple):
    name: str
    dtype: np.dtype
    complex_united_array_types: Type[ComplexUnitedArrayTypes]
    precision: int|None
class ComplexUnitedArrayValueType(Enum):
    value: ComplexUnitedArrayValueType_Information
COMPLEX64 =  ComplexUnitedArrayValueType_Information(name="complex64",  dtype=np.dtype("complex64"),  complex_united_array_types=np.complex64,  precision=64)
COMPLEX128 = ComplexUnitedArrayValueType_Information(name="complex128", dtype=np.dtype("complex128"), complex_united_array_types=np.complex128, precision=128)

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