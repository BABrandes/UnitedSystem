from abc import ABC, abstractmethod
from typing import Final, overload
from dataclasses import dataclass, field
from ..units.unit_quantity import UnitQuantity
import numpy as np

_PREFIX_PAIRS: Final[dict[str, float]] = {
    "Y": 10**24,
    "Z": 10**21,
    "E": 10**18,
    "P": 10**15,
    "T": 10**12,
    "G": 10**9,
    "M": 10**6,
    "k": 10**3,
    "h": 10**2,
    "da": 10**1,
    "d": 10**-1,
    "c": 10**-2,
    "m": 10**-3,
    "µ": 10**-6,  # Micro Sign (U+00B5)
    "μ": 10**-6,  # Greek Small Letter Mu (U+03BC) - same value as µ
    "n": 10**-9,
    "p": 10**-12,
    "f": 10**-15,
    "a": 10**-18,
    "z": 10**-21,
    "y": 10**-24,
}

@dataclass(frozen=True, slots=True)
class Unit(ABC):
    unit_quantity: UnitQuantity = field(init=False, hash=False, repr=False, compare=False)

    @abstractmethod
    def compatible_to(self, other: "Unit|UnitQuantity") -> bool:
        raise NotImplementedError("compatible_to is not implemented for this unit")

    @abstractmethod
    def nice_string(self) -> str:
        raise NotImplementedError("nice_string is not implemented for this unit")
    
    @abstractmethod
    def reduced_unit(self) -> "Unit":
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
    def pow(self, exponent: float) -> "Unit":
        raise NotImplementedError("pow is not implemented for this unit")

