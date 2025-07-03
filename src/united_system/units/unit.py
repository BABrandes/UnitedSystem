from abc import ABC, abstractmethod
from typing import Final, overload, List, Set, Tuple
from dataclasses import dataclass, field
from .unit_quantity import CanonicalQuantity, SimpleCanonicalQuantity
from .unit_symbol import UnitSymbol
from .named_canonical_quantities import NamedCanonicalQuantity
from .named_units import NamedUnit
from bidict import bidict
import re
import math
import numpy as np

_PREFIX_PAIRS: Final[bidict[str, float]] = bidict({
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
    "µ": 10**-6,
    "n": 10**-9,
    "p": 10**-12,
    "f": 10**-15,
    "a": 10**-18,
})

@dataclass(frozen=True, slots=True)
class Unit(ABC):
    canonical_quantity: CanonicalQuantity = field(init=False, hash=False, repr=False, compare=False)

    @abstractmethod
    def compatible_to(self, other: "Unit|CanonicalQuantity") -> bool:
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

@dataclass(frozen=True, slots=True)
class SimpleUnit(Unit):
    canonical_quantity: SimpleCanonicalQuantity = field(init=False, hash=False, repr=False, compare=False)
    unit_prefix_and_symbol_and_exponent_pairs: Final[tuple[(tuple[str, UnitSymbol, float], ...)]] = field()

    factor: float = field(init=False, hash=False, repr=False, compare=False)
    offset: float = field(init=False, hash=False, repr=False, compare=False)

    _CACHE: Final[dict[str, "SimpleUnit"]] = {}

    def __post_init__(self):
        quantity_exponents: list[float] = [0.0] * 7
        pseudo_quantity_exponents: list[int] = [0] * 2
        factor: float = 1
        offset: float = 0
        for prefix_and_symbol_and_exponent_pair in self.unit_prefix_and_symbol_and_exponent_pairs:
            if prefix_and_symbol_and_exponent_pair[0] != "":
                unit_symbol: UnitSymbol = prefix_and_symbol_and_exponent_pair[1]
                factor *= (_PREFIX_PAIRS[prefix_and_symbol_and_exponent_pair[0]] * unit_symbol.factor) ** prefix_and_symbol_and_exponent_pair[2]
                for exponent_index, exponent in enumerate(unit_symbol.named_canonical_quantity.canonical_quantity.quantity_exponents):
                    quantity_exponents[exponent_index] += exponent
                for exponent_index, exponent in enumerate(unit_symbol.named_canonical_quantity.canonical_quantity.pseudo_quantity_exponents):
                    pseudo_quantity_exponents[exponent_index] += exponent

            if prefix_and_symbol_and_exponent_pair[1].offset != 0:
                if offset != 0:
                    raise ValueError("Cannot have two non-zero offsets in the same unit")
                offset = prefix_and_symbol_and_exponent_pair[1].offset

        object.__setattr__(self, "quantity", SimpleCanonicalQuantity.create(quantity_exponents, pseudo_quantity_exponents))
        object.__setattr__(self, "factor", factor)
        object.__setattr__(self, "offset", offset)

    @overload
    @classmethod
    def create(cls, arg: tuple[str, UnitSymbol, float], *args: tuple[str, UnitSymbol, float]) -> "SimpleUnit":
        ...
    @overload
    @classmethod
    def create(cls, arg: "SimpleUnit", *args: "SimpleUnit") -> "SimpleUnit":
        ...
    @overload
    @classmethod
    def create(cls, arg: List["SimpleUnit"]|set["SimpleUnit"]) -> "SimpleUnit":
        ...
    @classmethod
    def create(cls, arg: tuple[str, UnitSymbol, float]|"SimpleUnit"|List["SimpleUnit"]|set["SimpleUnit"], *args: None|tuple[str, UnitSymbol, float]|"SimpleUnit") -> "SimpleUnit":

        items: list[tuple[str, UnitSymbol, float]] = []
        match arg:
            case list() | set():
                for unit in arg:
                    items.extend(unit.unit_prefix_and_symbol_and_exponent_pairs)
            case SimpleUnit():
                items.extend(arg.unit_prefix_and_symbol_and_exponent_pairs)
            case tuple():
                x: Tuple[str, UnitSymbol, float] = arg
                items.append(x)
        match args:
            case tuple():
                for v in args:
                    match v:
                        case tuple():
                            items.append(v)
                        case SimpleUnit():
                            items.extend(v.unit_prefix_and_symbol_and_exponent_pairs)
                        case _:
                            raise ValueError(f"Invalid units: {x}")
            case _:
                raise ValueError(f"Invalid units: {args}")
        return cls(tuple(items))
        
    @classmethod
    def parse(cls, unit_string: str) -> "SimpleUnit":
        """
        Parse a unit string into a Simple_Unit.
        
        Examples:
        - "m" -> Simple_Unit(("", {"meter"}, 1))
        - "km" -> Simple_Unit(("k", {"meter"}, 1))
        - "m/s" -> Simple_Unit(("", {"meter"}, 1), ("s", {"second"}, -1))
        - "V*m/ns^2" -> Simple_Unit(("", {"volt"}, 1), ("meter", {"meter"}, 1), ("n", {"second"}, -2))
        """

        if unit_string in cls._CACHE:
            return cls._CACHE[unit_string]

        unit_string = unit_string.strip()
        if unit_string == "":
            return cls(())
        unit_string_parts: list[str] = re.findall(r'[^*/]+|[*/][^*/]+', unit_string)
        unit_prefix_and_symbol_and_exponent_pairs: list[tuple[str, UnitSymbol, float]] = []
        for unit_string_part in unit_string_parts:

            # Determine the initial exponent based on if the unit part is in numerator or denominator
            if unit_string_part[0] == "/":
                exponent: float = -1
                unit_string_part = unit_string_part[1:]
            elif unit_string_part[0] == "*":
                exponent: float = 1
                unit_string_part = unit_string_part[1:]
            else:
                exponent: float = 1

            # Refine the exponent based on if the unit part is raised to a power
            if unit_string_part.count("^") == 1:
                unit_string_part, exponent_str = unit_string_part.split("^")
                if exponent_str.startswith("-"):
                    exponent *= -1 * float(exponent_str[1:])
                else:
                    exponent *= float(exponent_str)
            elif unit_string_part.count("^") > 1:
                raise ValueError(f"Invalid unit string: {unit_string}")
            else:
                # find the first letter from the end of the string that is not a number or a dot or a minus sign
                for i in range(len(unit_string_part) - 1, -1, -1):
                    if not unit_string_part[i].isdigit() and unit_string_part[i] != "." and unit_string_part[i] != "-":
                        exponent *= float(unit_string_part[i:])
                        unit_string_part = unit_string_part[:i]
                        break

            retrieved_unit_symbol: UnitSymbol|None = UnitSymbol.find_unit_symbol(unit_string_part)
            if retrieved_unit_symbol is not None:
                unit_prefix_and_symbol_and_exponent_pairs.append((prefix, retrieved_unit_symbol, exponent))
            else:
                for prefix in _PREFIX_PAIRS:
                    if unit_string_part.startswith(prefix):
                        retrieved_unit_symbol: UnitSymbol|None = UnitSymbol.find_unit_symbol(unit_string_part[len(prefix):])
                        if retrieved_unit_symbol is not None:
                            unit_prefix_and_symbol_and_exponent_pairs.append((prefix, retrieved_unit_symbol, exponent))
                            break
                else:
                    raise ValueError(f"Invalid unit string: {unit_string}")
        simple_unit: SimpleUnit = cls(tuple(unit_prefix_and_symbol_and_exponent_pairs))
        cls._CACHE[unit_string] = simple_unit
        return simple_unit

    def compatible_to(self, other: "Unit|CanonicalQuantity") -> bool:
        if isinstance(other, CanonicalQuantity):
            return self.canonical_quantity == other
        elif isinstance(other, SimpleUnit):
            return self.canonical_quantity == other.canonical_quantity
        else:
            raise ValueError(f"Invalid unit: {other}")
        
    @overload
    def from_canonical_value(self, canonical_value: np.float64) -> np.float64:
        ...
    @overload
    def from_canonical_value(self, canonical_value: np.float32) -> np.float32:
        ...
    @overload
    def from_canonical_value(self, canonical_value: np.float16) -> np.float16:
        ...
    @overload
    def from_canonical_value(self, canonical_value: float) -> float:
        ...
    @overload
    def from_canonical_value(self, canonical_value: int) -> int|float:
        ...
    @overload
    def from_canonical_value(self, canonical_value: complex) -> complex:
        ...
    @overload
    def from_canonical_value(self, canonical_value: np.ndarray) -> np.ndarray:
        ...
    def from_canonical_value(self, canonical_value: np.float64|np.float32|np.float16|float|int|complex|np.ndarray) -> np.float64|np.float32|np.float16|float|int|complex|np.ndarray:

        match canonical_value:
            case int():
                new_value = (canonical_value - self.offset) / self.factor
                return int(new_value) if new_value.is_integer() else new_value
            case float():
                float_type = type(canonical_value)
                if float_type is np.float16:
                    return np.float16((canonical_value - self.offset) / self.factor)
                elif float_type is np.float32:
                    return np.float32((canonical_value - self.offset) / self.factor)
                elif float_type is np.float64:
                    return np.float64((canonical_value - self.offset) / self.factor)
                else:
                    return (canonical_value - self.offset) / self.factor
            case complex():
                return (canonical_value - self.offset) / self.factor
            case np.ndarray():
                return (canonical_value - self.offset) / self.factor
            case _:
                raise ValueError(f"Invalid canonical value type: {type(canonical_value)}")

    @overload
    def to_canonical_value(self, value_in_unit: np.float64) -> float:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.float32) -> np.float32:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.float16) -> np.float16:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: float) -> float:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: int) -> float|int:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: complex) -> complex:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.ndarray) -> np.ndarray:
        ...
    def to_canonical_value(self, value_in_unit: np.float64|np.float32|np.float16|float|int|complex|np.ndarray) -> np.float64|np.float32|np.float16|float|int|complex|np.ndarray:
        match value_in_unit:
            case int():
                new_value: float = value_in_unit * self.factor + self.offset
                if new_value.is_integer():
                    return int(new_value)
                else:
                    return new_value
            case float():
                float_type = type(value_in_unit)
                if float_type is np.float16:
                    return np.float16(value_in_unit * self.factor + self.offset)
                elif float_type is np.float32:
                    return np.float32(value_in_unit * self.factor + self.offset)
                elif float_type is np.float64:
                    return np.float64(value_in_unit * self.factor + self.offset)
                else:
                    return value_in_unit * self.factor + self.offset
            case complex():
                return value_in_unit * self.factor + self.offset
            case np.ndarray():
                return value_in_unit * self.factor + self.offset
            case _:
                raise ValueError(f"Invalid value: {value_in_unit}")

    def pow(self, exponent: float) -> "SimpleUnit":
        new_prefix_and_symbol_and_exponent_pairs: list[tuple[str, UnitSymbol, float]] = []
        for prefix_and_symbol_and_exponent_pair in self.unit_prefix_and_symbol_and_exponent_pairs:
            new_prefix_and_symbol_and_exponent_pairs.append((prefix_and_symbol_and_exponent_pair[0], prefix_and_symbol_and_exponent_pair[1], prefix_and_symbol_and_exponent_pair[2] * exponent))
        return SimpleUnit(tuple(new_prefix_and_symbol_and_exponent_pairs))
        
    def __str__(self) -> str:
        return self.nice_string()
        
    def reduced_unit(self) -> "SimpleUnit":
        return self

    def nice_string(self) -> str:
        nice_string: str = ""
        for prefix_and_symbol_and_exponent_pair in self.unit_prefix_and_symbol_and_exponent_pairs:
            prefix: str = prefix_and_symbol_and_exponent_pair[0]
            symbol: UnitSymbol = prefix_and_symbol_and_exponent_pair[1]
            exponent: float = prefix_and_symbol_and_exponent_pair[2]

            symbol_str: str = symbol.symbols[0]
            match symbol, exponent:
                case symbol, 1:
                    nice_string += f"*{prefix}{symbol_str}"
                case symbol, exponent if exponent != 1 and exponent > 0:
                    exponent_str: str = str(round(exponent, 2))
                    if exponent_str.count(".") == 1:
                        if exponent_str.endswith("0"):
                            exponent_str = exponent_str[:-1]
                        if exponent_str.endswith("0"):
                            exponent_str = exponent_str[:-1]
                        if exponent_str.endswith("."):
                            exponent_str = exponent_str[:-1]
                    nice_string += f"*{prefix}{symbol_str}^{exponent_str}"
                case symbol, exponent if exponent < 0:
                    exponent_str: str = str(round(exponent, 2))
                    if exponent_str.endswith(".0"):
                        exponent_str = exponent_str[:-2]
                    elif exponent_str.endswith(".00"):
                        exponent_str = exponent_str[:-3]
                    nice_string += f"/{prefix}{symbol_str}^{exponent_str}"
                case symbol, exponent if exponent == 0:
                    nice_string += ""
                case _, _:
                    raise ValueError(f"Invalid unit: {self}")
                
        if nice_string.startswith("*"):
            nice_string = nice_string[1:]

        return nice_string
    
    @staticmethod
    def suggest_unit_from_named_units(quantity: SimpleCanonicalQuantity, canonical_value: float|None) -> "SimpleUnit":
        """
        Suggest a unit for a given quantity and value.

        Args:
            quantity: The quantity to suggest a unit for.
            canonical_value: The value to suggest a unit for.

        Returns:
            A unit so that the value is well formatted. If the value is None, the unit is the canonical unit.
        """

        exponents_to_check: list[float] = [1, -1, 0.5, -0.5, 2, -2, -1/3, 1.3, 3, -3, -1/4, 1.4, 4, -4]

        for exponent in exponents_to_check:
            named_canonical_quantity: NamedCanonicalQuantity|None = NamedCanonicalQuantity.find_named_canonical_quantity(quantity / exponent)
            if named_canonical_quantity is not None:
                canonical_unit: SimpleUnit = named_canonical_quantity.canonical_unit.pow(exponent)
                break
        else:
            raise ValueError(f"No unit found for quantity: {quantity}")
        
        if canonical_value is not None:
            named_units: set[NamedUnit] = NamedUnit.find_named_units(canonical_unit.canonical_quantity)
            unit_unit_string_pairs: list[Tuple[SimpleUnit, str]] = []
            for named_unit in named_units:
                value_in_unit: float = named_unit.unit.from_canonical_value(canonical_value)
                string_value: str = str(value_in_unit)
                unit_unit_string_pairs.append((named_unit.unit, string_value))

            def penality_function(string_value: str) -> float:
                if not "." in string_value and len(string_value) <= 3:
                    return 0
                if not "." in string_value and len(string_value) > 3:
                    return len(string_value) - 3
                if "." in string_value:
                    return (len(string_value) - string_value.index(".") - 1) - 3
                raise ValueError(f"Invalid string value: {string_value}")

            unit_unit_string_pairs.sort(key=lambda x: penality_function(x[1]))
            return unit_unit_string_pairs[0][0]
        else:
            return canonical_unit

    @staticmethod
    def suggest_units(quantity: SimpleCanonicalQuantity, value: float|None, must_include: set["SimpleUnit"]|list["SimpleUnit"]=set(), n: int = 1000) ->  Tuple["SimpleUnit", list["SimpleUnit"]]:
        """
        Suggest units for a given quantity and value.

        Args:
            quantity: The quantity to suggest units for.
            value: The value to suggest units for.
            must_include: A set of units that must be included in the suggestions.
            n: The number of suggestions to return.

        Returns:
            A tuple containing the first suggestion and a list of n suggestions.
            The first suggestion is the unit that is the closest to the value.
        """

        unit_string: str = ""
        checked_options: int = 0
        
        quantity_exponents: list[float] = list(quantity.quantity_exponents)
        pseudo_quantity_exponents: list[float] = list(quantity.pseudo_quantity_exponents)

        units: set["SimpleUnit"] = set()
        quantity_to_get_to_zero: SimpleCanonicalQuantity = quantity
        for unit in must_include:
            units.add(unit)
            quantity_to_get_to_zero -= unit.canonical_quantity

        exponents_to_check: list[float] = [1, -1, 0.5, -0.5, 2, -2, -1/3, 1.3, 3, -3]

        for exponent in exponents_to_check:
            named_canonical_quantity: NamedCanonicalQuantity|None = NamedCanonicalQuantity.find_named_canonical_quantity(quantity_to_get_to_zero / exponent)
            if named_canonical_quantity is not None:
                unit: SimpleUnit = named_canonical_quantity.canonical_unit.pow(exponent)
                units.add(unit)
                quantity_to_get_to_zero -= unit.canonical_quantity
                break
        else:
            raise NotImplementedError("Not implemented to construct units from canonical quantities by itself. Later...")

        if quantity_to_get_to_zero.is_zero():
            first_suggestion: SimpleUnit = SimpleUnit.create(units)

            # Get every option for each unit in the set of units
            options: dict[SimpleUnit, set[SimpleUnit]] = {}
            for unit in units:
                options[unit] = set()
                named_canonical_quantity: NamedCanonicalQuantity|None = NamedCanonicalQuantity.find_named_canonical_quantity(unit.canonical_quantity)
                if named_canonical_quantity is not None:
                    for named_unit in NamedUnit.find_named_units(unit.canonical_quantity):
                        options[unit].add(named_unit.unit)

            # Get all permutations of the options
            permutations: list[tuple[SimpleUnit, ...]] = list(product(*options.values()))

            # Convert the permutations to SimpleUnits
            unit_suggestions: list[SimpleUnit] = [first_suggestion]
            for permutation in permutations:
                unit_elements: list[SimpleUnit] = []
                for x in permutation:
                    unit_elements.append(x)
                unit_suggestions.append(SimpleUnit.create(unit_elements))

            # Sort the unit_suggestions based on how different the factor is from the value
            unit_suggestions.sort(key=lambda x: abs(math.log(x.factor/first_suggestion.factor)))

            if len(unit_suggestions) > n:
                unit_suggestions = unit_suggestions[:n]

            return first_suggestion, unit_suggestions

        else:
            raise NotImplementedError("Not implemented to construct units from canonical quantities by itself. Later...")
        
        # So, now we have possible combinations of unit symbols, we first need to find the canonical base unit:
