from dataclasses import dataclass
from typing import List, Tuple, Final, cast
from abc import ABC, abstractmethod

MASS_INDEX: int = 0
TIME_INDEX: int = 1
LENGTH_INDEX: int = 2
CURRENT_INDEX: int = 3
TEMPERATURE_INDEX: int = 4
AMOUNT_OF_SUBSTANCE_INDEX: int = 5
LUMINOUS_INTENSITY_INDEX: int = 6

@dataclass(frozen=True, slots=True)
class CanonicalQuantity(ABC):
    
    @abstractmethod
    def __add__(self, other: "CanonicalQuantity") -> "CanonicalQuantity":
        raise NotImplementedError

    @abstractmethod
    def __radd__(self, other: "CanonicalQuantity") -> "CanonicalQuantity":
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: "CanonicalQuantity") -> "CanonicalQuantity":
        raise NotImplementedError

    @abstractmethod
    def __rsub__(self, other: "CanonicalQuantity") -> "CanonicalQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def __mul__(self, other: float) -> "CanonicalQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def __truediv__(self, other: float) -> "CanonicalQuantity":
        raise NotImplementedError
    
    @abstractmethod
    def __eq__(self, other: "CanonicalQuantity") -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def __ne__(self, other: "CanonicalQuantity") -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def is_zero(self) -> bool:
        raise NotImplementedError
    
    @abstractmethod
    def to_json(self) -> dict:
        raise NotImplementedError
    
    @abstractmethod
    def from_json(self, json: dict) -> "CanonicalQuantity":
        raise NotImplementedError
    
    from .unit import Unit
    @abstractmethod
    def canonical_unit(self) -> "Unit":
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class SimpleCanonicalQuantity(CanonicalQuantity):
    quantity_exponents: Final[Tuple[float, float, float, float, float, float, float]]
    pseudo_quantity_exponents: Final[Tuple[int, int]]

    @classmethod
    def create(cls, quantity_exponents: Tuple[float, float, float, float, float, float, float]|list[float], pseudo_quantity_exponents: Tuple[int, int]|list[int]) -> "SimpleCanonicalQuantity":
        if isinstance(quantity_exponents, list):
            quantity_exponents = cast(Tuple[float, float, float, float, float, float, float], tuple(quantity_exponents))
        if isinstance(pseudo_quantity_exponents, list):
            pseudo_quantity_exponents = cast(Tuple[int, int], tuple(pseudo_quantity_exponents))
        return cls(quantity_exponents, pseudo_quantity_exponents)

    def __add__(self, other: "SimpleCanonicalQuantity") -> "SimpleCanonicalQuantity":
        return SimpleCanonicalQuantity((self.quantity_exponents[0] + other.quantity_exponents[0], self.quantity_exponents[1] + other.quantity_exponents[1], self.quantity_exponents[2] + other.quantity_exponents[2], self.quantity_exponents[3] + other.quantity_exponents[3], self.quantity_exponents[4] + other.quantity_exponents[4], self.quantity_exponents[5] + other.quantity_exponents[5], self.quantity_exponents[6] + other.quantity_exponents[6]),
                                         (self.pseudo_quantity_exponents[0] + other.pseudo_quantity_exponents[0], self.pseudo_quantity_exponents[1] + other.pseudo_quantity_exponents[1]))

    def __radd__(self, other: "SimpleCanonicalQuantity") -> "SimpleCanonicalQuantity":
        return self.__add__(other)
        
    def __sub__(self, other: "SimpleCanonicalQuantity") -> "SimpleCanonicalQuantity":
        return SimpleCanonicalQuantity((self.quantity_exponents[0] - other.quantity_exponents[0], self.quantity_exponents[1] - other.quantity_exponents[1], self.quantity_exponents[2] - other.quantity_exponents[2], self.quantity_exponents[3] - other.quantity_exponents[3], self.quantity_exponents[4] - other.quantity_exponents[4], self.quantity_exponents[5] - other.quantity_exponents[5], self.quantity_exponents[6] - other.quantity_exponents[6]),
                                         (self.pseudo_quantity_exponents[0] - other.pseudo_quantity_exponents[0], self.pseudo_quantity_exponents[1] - other.pseudo_quantity_exponents[1]))

    def __rsub__(self, other: "SimpleCanonicalQuantity") -> "SimpleCanonicalQuantity":
        return SimpleCanonicalQuantity((other.quantity_exponents[0] - self.quantity_exponents[0], other.quantity_exponents[1] - self.quantity_exponents[1], other.quantity_exponents[2] - self.quantity_exponents[2], other.quantity_exponents[3] - self.quantity_exponents[3], other.quantity_exponents[4] - self.quantity_exponents[4], other.quantity_exponents[5] - self.quantity_exponents[5], other.quantity_exponents[6] - self.quantity_exponents[6]),
                                         (other.pseudo_quantity_exponents[0] - self.pseudo_quantity_exponents[0], other.pseudo_quantity_exponents[1] - self.pseudo_quantity_exponents[1]))
    
    def __mul__(self, other: float|int) -> "SimpleCanonicalQuantity":
        if self.pseudo_quantity_exponents[0] == 0 and self.pseudo_quantity_exponents[1] == 0:
            return SimpleCanonicalQuantity(
                (self.quantity_exponents[0] * other, self.quantity_exponents[1] * other, self.quantity_exponents[2] * other, self.quantity_exponents[3] * other, self.quantity_exponents[4] * other, self.quantity_exponents[5] * other, self.quantity_exponents[6] * other),
                (0, 0))
        elif isinstance(other, int) or other % 1 == 0:
            return SimpleCanonicalQuantity(
                (self.quantity_exponents[0] * other, self.quantity_exponents[1] * other, self.quantity_exponents[2] * other, self.quantity_exponents[3] * other, self.quantity_exponents[4] * other, self.quantity_exponents[5] * other, self.quantity_exponents[6] * other),
                (self.pseudo_quantity_exponents[0] * int(other), self.pseudo_quantity_exponents[1] * int(other)))
        else:
            raise ValueError(f"Cannot have a non-integer pseudo_quantity_exponents value: {self.pseudo_quantity_exponents[0]} {self.pseudo_quantity_exponents[1]}")
    
    def __truediv__(self, other: float|int) -> "SimpleCanonicalQuantity":
        if self.pseudo_quantity_exponents[0] == 0 and self.pseudo_quantity_exponents[1] == 0:
            return SimpleCanonicalQuantity(
                (self.quantity_exponents[0] / other, self.quantity_exponents[1] / other, self.quantity_exponents[2] / other, self.quantity_exponents[3] / other, self.quantity_exponents[4] / other, self.quantity_exponents[5] / other, self.quantity_exponents[6] / other),
                (0, 0))
        elif isinstance(other, int) or other % 1 == 0:
            return SimpleCanonicalQuantity(
                (self.quantity_exponents[0] / other, self.quantity_exponents[1] / other, self.quantity_exponents[2] / other, self.quantity_exponents[3] / other, self.quantity_exponents[4] / other, self.quantity_exponents[5] / other, self.quantity_exponents[6] / other),
                (int(self.pseudo_quantity_exponents[0] / other), int(self.pseudo_quantity_exponents[1] / other)))
        else:
            raise ValueError(f"Cannot have a non-integer pseudo_quantity_exponents value: {self.pseudo_quantity_exponents[0]} {self.pseudo_quantity_exponents[1]}")
    
    def __eq__(self, other: "SimpleCanonicalQuantity") -> bool:
        return self.quantity_exponents == other.quantity_exponents and self.pseudo_quantity_exponents == other.pseudo_quantity_exponents
    
    def __ne__(self, other: "SimpleCanonicalQuantity") -> bool:
        return self.quantity_exponents != other.quantity_exponents or self.pseudo_quantity_exponents != other.pseudo_quantity_exponents
    
    def is_zero(self) -> bool:
        return self.quantity_exponents[0] == 0 and self.quantity_exponents[1] == 0 and self.quantity_exponents[2] == 0 and self.quantity_exponents[3] == 0 and self.quantity_exponents[4] == 0 and self.quantity_exponents[5] == 0 and self.quantity_exponents[6] == 0 and self.pseudo_quantity_exponents[0] == 0 and self.pseudo_quantity_exponents[1] == 0
    
    def to_json(self) -> dict:
        return {"canonical_unit": self.canonical_unit().nice_string()}
    
    @classmethod
    def from_json(cls, json: dict) -> "SimpleCanonicalQuantity":
        from .unit import SimpleUnit
        return SimpleUnit.parse(json["canonical_unit"]).canonical_quantity
    
    from .unit import SimpleUnit
    def canonical_unit(self) -> "SimpleUnit":
        from .unit import SimpleUnit
        unit: SimpleUnit = SimpleUnit.suggest_unit_from_named_units(self, None)
        return unit

@dataclass(frozen=True, slots=True)
class Subscripted_Canonical_Quantity(CanonicalQuantity):
    subscript_quantity_exponents: List[Tuple[float, float, float, float, float, float, float]]
    log_lin_exp_specified: List[Tuple[int, int, int]]
    substripts: List[str]

