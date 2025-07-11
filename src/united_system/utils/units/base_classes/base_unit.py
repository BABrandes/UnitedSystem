from abc import ABC, abstractmethod
from typing import overload, TypeVar, TYPE_CHECKING, Generic, Type, Any
from dataclasses import dataclass, field
from .base_dimension import BaseDimension
import numpy as np
from ....utils.general import JSONable, HDF5able
import h5py
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .base_dimension import BaseDimension

DT = TypeVar("DT", bound=BaseDimension[Any, Any])
UT = TypeVar("UT", bound="BaseUnit")

@dataclass(frozen=True, slots=True)
class BaseUnit(ABC, JSONable, HDF5able, Generic[DT, UT]):
    dimension: DT = field(init=False, hash=False, repr=False, compare=False)
    factor: float = field(init=False, hash=False, repr=False, compare=False)
    offset: float = field(init=False, hash=False, repr=False, compare=False)

    def compatible_to(self, *args: UT|DT) -> bool:
        return self.compatible(*args)
    
    @abstractmethod
    @classmethod
    def compatible(cls, *args: UT|DT) -> bool:
        raise NotImplementedError("compatible_to is not implemented for this unit")

    @abstractmethod
    def reduced_unit(self) -> UT:
        raise NotImplementedError("reduced_unit is not implemented for this unit")

    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: int) -> int|float:
        """Convert a value from the canonical unit to the display unit."""
        ...
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
    def from_canonical_value(self, canonical_value: np.complex128) -> np.complex128:
        """Convert a value from the canonical unit to the display unit."""
        ...  
    @abstractmethod
    @overload
    def from_canonical_value(self, canonical_value: float) -> float:
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
    def from_canonical_value(self, canonical_value: np.float64|np.float32|np.float16|np.complex128|float|int|complex|np.ndarray) -> np.float64|np.float32|np.float16|np.complex128|float|int|complex|np.ndarray:
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
    def to_canonical_value(self, value_in_unit: int) -> float|int:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: np.float64) -> np.float64:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: np.float32) -> np.float32:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: np.float16) -> np.float16:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: np.complex128) -> np.complex128:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: float) -> float:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: complex) -> complex:
        """Convert a value from the display unit to the canonical unit."""
        ...
    @overload
    @abstractmethod
    def to_canonical_value(self, value_in_unit: np.ndarray) -> np.ndarray:
        """Convert a numpy array from the display unit to the canonical unit."""
        ...
    @abstractmethod
    def to_canonical_value(self, value_in_unit: int|np.float64|np.float32|np.float16|np.complex128|float|complex|np.ndarray) -> np.float64|np.float32|np.float16|np.complex128|float|int|complex|np.ndarray:
        """
        Convert a value from the display unit to the canonical unit.
        
        Examples:
        - {km} 0.005 -> 5 m
        - {cm/s} 1 -> 0.01 m/s
        - {kV} 0.01 -> 10 V
        - {kg} 5 -> 5 kg
        """

        raise NotImplementedError("to_canonical_value is not implemented for this unit")
    
    @overload
    @classmethod
    def convert(cls, value: int, from_unit: UT, to_unit: UT) -> float:
        """Convert a value from one unit to another."""
        ...
    @overload
    @classmethod
    def convert(cls, value: float, from_unit: UT, to_unit: UT) -> float:
        """Convert a value from one unit to another."""
        ...
    @overload
    @classmethod
    def convert(cls, value: complex, from_unit: UT, to_unit: UT) -> complex:
        """Convert a value from one unit to another."""
        ...
    @overload
    @classmethod
    def convert(cls, value: np.ndarray, from_unit: UT, to_unit: UT) -> np.ndarray:
        """Convert a value from one unit to another."""
        ...
    @classmethod
    def convert(cls, value: int|float|complex|np.ndarray, from_unit: UT, to_unit: UT) -> int|float|complex|np.ndarray:
        """
        Convert a value from one unit to another.
        """
        if not cls.compatible(from_unit, to_unit):
            raise ValueError(f"Units {from_unit} and {to_unit} are not compatible.")
        return to_unit.from_canonical_value(from_unit.to_canonical_value(value)) # type: ignore

    @abstractmethod
    def pow(self, exponent: float) -> UT:
        raise NotImplementedError("pow is not implemented for this unit")
    
    @abstractmethod
    def format_string(self, no_fraction: bool) -> str:
        ...

    @abstractmethod
    @classmethod
    def parse_string(cls, unit_string: str) -> UT:
        ...

    def to_json(self) -> dict[str, str]:
        return {
            "unit_string": self.format_string(no_fraction=False)
        }
    
    @classmethod
    def from_json(cls, data: dict[str, str], **type_parameters: Type[DT]) -> UT:
        return cls.parse_string(data["unit_string"])
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        hdf5_group.attrs["unit_string"] = self.format_string(no_fraction=False)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **type_parameters: Type[DT]) -> UT:
        return cls.parse_string(hdf5_group.attrs["unit_string"]) # type: ignore