from typing import Iterator, Generic, TypeVar, TypeAlias, Any
import numpy as np
from dataclasses import dataclass
from ...scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ...scalars.complex_united_scalar.complex_united_scalar import ComplexUnitedScalar
from ...utils import JSONable, HDF5able
from pandas import Timestamp
from abc import ABC, abstractmethod
import h5py

SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar|str|bool|int|float|Timestamp

ST = TypeVar("ST", bound=SCALAR_TYPE)
AT = TypeVar("AT", bound="BaseArray")

@dataclass(frozen=True, slots=True)
class BaseArray(ABC, JSONable, HDF5able, Generic[ST, AT]):

    canonical_np_array: np.ndarray

    @abstractmethod
    def get_scalar(self, index: int) -> ST:
        ...
    
    @abstractmethod
    def __getitem__(self, key: str) -> ST:
        ...
    
    @abstractmethod
    def __len__(self) -> int:
        ...
    
    @abstractmethod
    def __iter__(self) -> Iterator[ST]:
        ...
    
    @abstractmethod
    def __next__(self) -> ST:
        ...
    
    @abstractmethod
    def __contains__(self, item: ST) -> bool:
        ...

    def to_json(self) -> dict[str, Any]:
        return {"canonical_np_array": self.canonical_np_array}
    
    @classmethod
    def from_json(cls, json: dict[str, Any]) -> AT:
        return cls(canonical_np_array=json["canonical_np_array"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.create_dataset("canonical_np_array", data=self.canonical_np_array)

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> AT:
        return cls(canonical_np_array=hdf5_group["canonical_np_array"][()])