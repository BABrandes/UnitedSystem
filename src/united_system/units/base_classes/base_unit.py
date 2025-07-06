from abc import ABC, abstractmethod
from typing import Final, overload, TypeVar, TYPE_CHECKING, Generic
from dataclasses import dataclass, field
from .base_dimension import BaseDimension
import numpy as np
from ...utils import JSONable, HDF5able
import h5py
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .base_dimension import BaseDimension

DT = TypeVar("DT", bound=BaseDimension)
UT = TypeVar("UT", bound="BaseUnit")

@dataclass(frozen=True, slots=True)
class BaseUnit(ABC, JSONable, HDF5able, Generic[DT, UT]):
    dimension: DT = field(init=False, hash=False, repr=False, compare=False)
    factor: float = field(init=False, hash=False, repr=False, compare=False)
    offset: float = field(init=False, hash=False, repr=False, compare=False)

    @abstractmethod
    def compatible_to(self, other: UT) -> bool:
        raise NotImplementedError("compatible_to is not implemented for this unit")

    @abstractmethod
    def reduced_unit(self) -> UT:
        raise NotImplementedError("reduced_unit is not implemented for this unit")

    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: np.float64) -> np.float64:
        """Convert a value from the canonical unit to the display unit."""
        ...   
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: np.float32) -> np.float32:
        """Convert a value from the canonical unit to the display unit."""
        ...   
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: np.float16) -> np.float16:
        """Convert a value from the canonical unit to the display unit."""
        ...   
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: float) -> float:
        """Convert a value from the canonical unit to the display unit."""
        ...   
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: int) -> int|float:
        """Convert a value from the canonical unit to the display unit."""
        ...
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: complex) -> complex:
        """Convert a value from the canonical unit to the display unit."""
        ...
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: np.ndarray) -> np.ndarray:
        """Convert a numpy array from the canonical unit to the display unit."""
        ...
    @abstractmethod
    def from_canonical_value(self, canonical_value: np.float64|np.float32|np.float16|float|int|complex|np.ndarray) -> np.float64|np.float32|np.float16|float|int|complex|np.ndarray:
        """
        Convert a value from the canonical unit to the display unit.
        
        Examples:
        - {km} 5 -> 0.005 km
        - {cm/s} 0.01 -> 1 cm/s
        - {kV} 10 -> 0.01 kV
        - {kg} 5 -> 5 kg
        
        """

        raise NotImplementedError("from_canonical_value is not implemented for this unit")
    
    @overload
    @abstractmethod
    def to_canonical_value(self, value: float) -> float:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value: int) -> float|int:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value: complex) -> complex:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value: np.ndarray) -> np.ndarray:
        """Convert a numpy array from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value: float|int|complex|np.ndarray) -> float|int|complex|np.ndarray:
        """
        Convert a value from the display unit to the canonical unit.
        
        Examples:
        - {km} 0.005 -> 5 m
        - {cm/s} 1 -> 0.01 m/s
        - {kV} 0.01 -> 10 V
        - {kg} 5 -> 5 kg
        """

        raise NotImplementedError("to_canonical_value is not implemented for this unit")
    
    @abstractmethod
    def pow(self, exponent: float) -> UT:
        raise NotImplementedError("pow is not implemented for this unit")
    
    @abstractmethod
    def format_string(self, no_fraction: bool) -> str:
        ...

    @abstractmethod
    def parse_string(self, unit_string: str) -> UT:
        ...

    def to_json(self) -> dict[str, str]:
        return {
            "unit_string": self.format_string(no_fraction=False)
        }
    
    @classmethod
    def from_json(cls, data: dict[str, str]) -> UT:
        return cls.parse_string(data["unit_string"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.attrs["unit_string"] = self.format_string(no_fraction=False)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> UT:
        return cls.parse_string(hdf5_group.attrs["unit_string"])