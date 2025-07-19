from typing import Final, Tuple, overload, Union, Literal, Optional
from collections.abc import Sequence
import re
import numpy as np

from .unit_symbol import UnitSymbol
from .unit_element import UnitElement
from .dimension_group import DimensionGroup
from ...named_quantity import NamedQuantity

# Cache for parsed units (outside the dataclass to avoid slots conflict)
_SIMPLE_UNIT_CACHE__STRING_KEY: dict[str, "UnitGroup"] = {}
_SIMPLE_UNIT_CACHE__DIMENSION_KEY: dict[DimensionGroup, "UnitGroup"] = {}

EPSILON: float = 1e-10

def clear_unit_cache():
    """Clear the unit cache to force re-parsing of unit strings."""
    global _SIMPLE_UNIT_CACHE__STRING_KEY, _SIMPLE_UNIT_CACHE__DIMENSION_KEY
    _SIMPLE_UNIT_CACHE__STRING_KEY.clear()
    _SIMPLE_UNIT_CACHE__DIMENSION_KEY.clear()
    # Also clear the SimpleUnitElement cache
    from .unit_element import clear_simple_unit_element_cache
    clear_simple_unit_element_cache()

class UnitGroup:
    """
    A unit group is a collection of unit elements.

    Creation:

    - Unit groups can be created from a string.

    - Unit groups can be created from a list of unit elements.

    - Unit groups can be created from a dimension, giving the canonical unit of that dimension.

    Fields:

    - unit_elements: A tuple of unit elements.

    - dimension_group: The dimension group of the unit group.

    - factor: The factor of the unit group.

    Each unit element has a prefix, a unit symbol, and an exponent.

    Their may not be two unit elements with the same unit symbol.

    The prefix, the unit symbol and the exponent together give a factor to the canonical unit.

    The product of the factor of each unit element gives the factor of the unit.

    One unit element may have an offset.

    This offset gives the offset of the unit.

    Each unit has a dimension, computed from the unit elements.

    Unit groups are immutable.

    Operations:

    - Unit groups cannot be added or subtracted.

    - Unit groups can be multiplied and divided by other unit groups or by floats or by ints.

    - Unit groups can be raised to a power of a float or an int.

    A new UnitGroup is created when an operation is performed on a unit group.

    It must be made sure that the unit elements with the same unit symbol are combined.

    Methods:

    - reduce: The unit is reduced by combining unit elements to minimize the number of unit elements. (Not implemented yet)

    - combine_same_unit_symbols: Combine unit elements with the same unit symbol.

    """

########################################################
# Initialization
########################################################

    def __new__(
            cls,
            value: Optional[Union[Sequence[UnitElement], str, DimensionGroup]],
            log_unit_group: Optional["UnitGroup"] = None,
            angle_unit_group: Optional["UnitGroup"] = None,
            log_exponent: float = 0.0,
            angle_exponent: float = 0.0) -> "UnitGroup":
        
        if isinstance(value, str) or isinstance(value, DimensionGroup):
            return cls._get_unit_group_via_cache(value)
        else:
            return super().__new__(cls)

    @overload
    def __init__(
        self,
        value: Sequence[UnitElement],
        log_unit_group: Optional["UnitGroup"] = None,
        angle_unit_group: Optional["UnitGroup"] = None,
        log_exponent: float = 0.0,
        angle_exponent: float = 0.0) -> None:
        ...
    @overload
    def __init__(self, value: str) -> None:
        """
        Parse a unit string into a Unit.
        
        Examples:
        - "m" -> Simple_Unit(("", {"meter"}, 1))
        - "km" -> Simple_Unit(("k", {"meter"}, 1))
        - "m/s" -> Simple_Unit(("", {"meter"}, 1), ("s", {"second"}, -1))
        - "V*m/ns^2" -> Simple_Unit(("", {"volt"}, 1), ("meter", {"meter"}, 1), ("n", {"second"}, -2))
        """
        ...
    @overload
    def __init__(self, value: DimensionGroup) -> None:
        ...
    @overload
    def __init__(self, value: None) -> None:
        ...
    def __init__(
            self,
            value: Optional[Union[Sequence[UnitElement], str, DimensionGroup]],
            log_unit_group: Optional["UnitGroup"] = None,
            angle_unit_group: Optional["UnitGroup"] = None,
            log_exponent: float = 0.0,
            angle_exponent: float = 0.0):

        if isinstance(value, str) or isinstance(value, DimensionGroup) or value is None:
            return  # Prevents accidental init of cached objects
            
        self._unit_elements = tuple(value)
        self._log_unit_group = log_unit_group
        self._angle_unit_group = angle_unit_group
        self._log_exponent = log_exponent
        self._angle_exponent = angle_exponent

        proper_exponents: list[float] = [0.0] * 7
        factor: float = 1
        offset: float = 0
        for unit_element in self._unit_elements:

            # Set factor and offset
            factor *= unit_element.canonical_factor
            new_offset: float = unit_element.canonical_offset
            if offset != 0 and new_offset != 0:
                raise ValueError("Cannot have two non-zero offsets in the same unit")
            offset = new_offset
            
            # Add quantity exponents (regardless of prefix)
            unit_symbol: UnitSymbol = unit_element.unit_symbol
            exponent: float = unit_element.exponent
            for exponent_index, proper_exp in enumerate(unit_symbol.named_quantity.dimension_group.proper_exponents):
                proper_exponents[exponent_index] += float(proper_exp * exponent)

        self._dimension_group = DimensionGroup(
            proper_exponents=proper_exponents,
            log_dimension=log_unit_group.dimension_group if log_unit_group is not None else None,
            angle_dimension=angle_unit_group.dimension_group if angle_unit_group is not None else None,
            log_exponent=log_exponent,
            angle_exponent=angle_exponent
        )

        self._factor = factor
        self._offset = offset

########################################################
# Helper methods
########################################################
    
    @classmethod
    def _get_unit_group_via_cache(cls, value: Union[str, DimensionGroup, None]) -> "UnitGroup":
        if isinstance(value, str):
            if value in _SIMPLE_UNIT_CACHE__STRING_KEY:
                return _SIMPLE_UNIT_CACHE__STRING_KEY[value]
            else:
                unit_group: UnitGroup = cls._create_from_string(value)
                _SIMPLE_UNIT_CACHE__STRING_KEY[value] = unit_group
                return unit_group
        elif isinstance(value, DimensionGroup):
            if value in _SIMPLE_UNIT_CACHE__DIMENSION_KEY:
                return _SIMPLE_UNIT_CACHE__DIMENSION_KEY[value]
            else:
                unit_group: UnitGroup = cls._create_from_dimension_group(value)
                _SIMPLE_UNIT_CACHE__DIMENSION_KEY[value] = unit_group
                return unit_group
        else:
            return DIMENSIONLESS_UNIT_GROUP
        
    @classmethod
    def _create_from_string(cls, unit_string: str) -> "UnitGroup":
        """
        Parse a unit string into a tuple of SimpleUnitElements.
        
        Examples:
        - "m" -> Simple_Unit(("", {"meter"}, 1))
        - "km" -> Simple_Unit(("k", {"meter"}, 1))
        - "m/s" -> Simple_Unit(("", {"meter"}, 1), ("s", {"second"}, -1))
        - "V*m/ns^2" -> Simple_Unit(("", {"volt"}, 1), ("meter", {"meter"}, 1), ("n", {"second"}, -2))
        - "s^2/(kg*m)" -> Simple_Unit(("s", {"second"}, 2), ("kg", {"gram"}, -1), ("m", {"meter"}, -1))
        """

        # Handle special case: if string is empty, return empty tuple
        if not unit_string.strip():
            return cls(())
        
        if unit_string.count("dec(") or unit_string.count("rad(") or unit_string.count("dec") or unit_string.count("rad") or unit_string.count("°") or unit_string.count("°("):
            raise ValueError(f"Not implemented yet: {unit_string}")

        # Handle parentheses properly by parsing the structure
        unit_elements: list[UnitElement] = []
        
        # Find the main division operator (outside parentheses)
        # This regex finds / that are not inside parentheses
        main_division_match = re.search(r'/(?![^(]*\))', unit_string)
        
        if main_division_match:
            # Split at the main division
            numerator_str = unit_string[:main_division_match.start()].strip()
            denominator_str = unit_string[main_division_match.end():].strip()
            
            # Remove outer parentheses from denominator if present
            if denominator_str.startswith("(") and denominator_str.endswith(")"):
                denominator_str = denominator_str[1:-1].strip()
            
            # Parse numerator
            if numerator_str:
                unit_elements.extend(cls._parse_multiplication_expression(numerator_str, "nominator"))
            
            # Parse denominator
            if denominator_str:
                unit_elements.extend(cls._parse_multiplication_expression(denominator_str, "denominator"))
        else:
            # No main division, treat as all numerator
            unit_elements.extend(cls._parse_multiplication_expression(unit_string, "nominator"))

        return cls(tuple(unit_elements))
    
    @classmethod
    def _create_from_dimension_group(cls, dimension_group: DimensionGroup) -> "UnitGroup":
        unit_elements: list[UnitElement] = []
        # Handle main dimensions
        for i, named_quantity in zip([0, 1, 2, 3, 4, 5, 6], [NamedQuantity.MASS, NamedQuantity.TIME, NamedQuantity.LENGTH, NamedQuantity.CURRENT, NamedQuantity.TEMPERATURE, NamedQuantity.AMOUNT_OF_SUBSTANCE, NamedQuantity.LUMINOUS_INTENSITY]):
            if dimension_group.proper_exponents[i] != 0:
                unit_element: UnitElement = named_quantity.unit_group._unit_elements[0]
                unit_elements.append(UnitElement(unit_element.prefix, unit_element.unit_symbol, dimension_group.proper_exponents[i]))
        log_unit_group: Optional[UnitGroup] = None
        angle_unit_group: Optional[UnitGroup] = None
        log_exponent: float = 0.0
        angle_exponent: float = 0.0
        if dimension_group.log_dimension is not None:
            log_unit_group = cls._create_from_dimension_group(dimension_group.log_dimension)
            log_exponent = dimension_group.log_exponent
        if dimension_group.angle_dimension is not None:
            angle_unit_group = cls._create_from_dimension_group(dimension_group.angle_dimension)
            angle_exponent = dimension_group.angle_exponent

        return cls(
            value=unit_elements,
            log_unit_group=log_unit_group,
            angle_unit_group=angle_unit_group,
            log_exponent=log_exponent,
            angle_exponent=angle_exponent)
    
    @classmethod
    def dimensionless_unit_group(cls) -> "UnitGroup":
        return DIMENSIONLESS_UNIT_GROUP

########################################################
# Properties
########################################################

    @property
    def dimension_group(self) -> DimensionGroup:
        return self._dimension_group

    @property
    def factor(self) -> float:
        return self._factor

    @property
    def offset(self) -> float:
        return self._offset 
    
########################################################
# Parsing operations
########################################################

    @classmethod
    def parse_string(cls, unit_string: str) -> "UnitGroup":
        return cls._get_unit_group_via_cache(unit_string)

    @classmethod
    def _parse_multiplication_expression(cls, expression: str, position: Literal["nominator", "denominator"]) -> list[UnitElement]:
        """Parse a multiplication expression (e.g., "kg*m*s^2") into unit elements."""
        unit_elements: list[UnitElement] = []
        
        # Split by multiplication operators
        parts = re.split(r'\*', expression)
        parts = [part.strip() for part in parts if part.strip()]
        
        for part in parts:
            # Skip "1" as it represents dimensionless (no contribution)
            if part == "1":
                continue
            
            unit_elements.append(UnitElement.parse_string(part, position))
        
        return unit_elements
    
########################################################
# Arithmetic operations
########################################################

    # ----------- Simple operations -----------#

    def __mul__(self, other: "UnitGroup") -> "UnitGroup":
        """Multiply two units. E.g. m * s -> m*s"""
        unit_elements: list[UnitElement] = []
        for element in self._unit_elements:
            unit_elements.append(element)
        for element in other._unit_elements:
            unit_elements.append(element)
        combined_unit_elements: Sequence[UnitElement] = self._combine_same_unit_symbols_and_prefixes(unit_elements)
        return UnitGroup(combined_unit_elements)
    
    def __truediv__(self, other: "UnitGroup") -> "UnitGroup":
        """Divide two units. E.g. m/s -> m/s"""
        unit_elements: list[UnitElement] = []
        for element in self._unit_elements:
            unit_elements.append(element)
        for element in other._unit_elements:
            unit_elements.append(element.invert())
        combined_unit_elements: Sequence[UnitElement] = self._combine_same_unit_symbols_and_prefixes(unit_elements)
        return UnitGroup(combined_unit_elements)

    def pow(self, exponent: float|int) -> "UnitGroup":
        """Raise a unit to a power. E.g. m^2 -> m^2"""
        unit_elements: list[UnitElement] = []
        for element in self._unit_elements:
            # Multiply the exponent, preserving its sign
            unit_elements.append(UnitElement(element.prefix, element.unit_symbol, element.exponent * exponent))
        combined_unit_elements: Sequence[UnitElement] = self._combine_same_unit_symbols_and_prefixes(unit_elements)
        return UnitGroup(combined_unit_elements)
    
    def __invert__(self) -> "UnitGroup":
        """Invert a unit. E.g. ~m -> 1/m"""
        unit_elements: list[UnitElement] = []
        for element in self._unit_elements:
            unit_elements.append(element.invert())
        combined_unit_elements: Sequence[UnitElement] = self._combine_same_unit_symbols_and_prefixes(unit_elements)
        return UnitGroup(combined_unit_elements)
    
    def invert(self) -> "UnitGroup":
        """Invert a unit. E.g. ~m -> 1/m"""
        return ~self
    
    # ----------- Advanced operations -----------#

    def log(self) -> "UnitGroup":
        """Take the log of a unit. E.g. log(m) -> log(m)"""

        return UnitGroup(
            value=(),
            log_unit_group=self,
            angle_unit_group=None,
            log_exponent=1.0,
            angle_exponent=0.0
        )
    
    def exp(self) -> "UnitGroup":
        """Exponentiate a unit. E.g. exp(m) -> e^m"""
        if self.is_dimensionless:
            return self
        elif len(self._unit_elements) == 0 and self._angle_unit_group is None and self._log_unit_group is not None:
            log_exponent: float = self._log_exponent - 1
            if abs(log_exponent) < EPSILON:
                log_exponent = 0.0
            else:
                raise ValueError("Cannot exponentiate a unit with a log exponent not equal to 1")
            return self._log_unit_group
        else:
            raise ValueError("Cannot exponentiate a non-dimensionless unit if it has no log unit group")
    
    def arc(self) -> "UnitGroup":
        """Take the arcsin of a unit. E.g. arcsin(m) -> sin^-1(m)"""
        return UnitGroup(
            value=(),
            log_unit_group=self,
            angle_unit_group=None,
            log_exponent=0.0,
            angle_exponent=1.0
        )
    
    def trig(self) -> "UnitGroup":
        """Take the trigonometric functions of a unit. E.g. sin(m) -> sin(m)"""
        if self.is_dimensionless:
            return self
        elif len(self._unit_elements) == 0 and self._log_unit_group is None and self._angle_unit_group is not None:
            angle_exponent: float = self._angle_exponent - 1
            if abs(angle_exponent) < EPSILON:
                angle_exponent = 0.0
            else:
                raise ValueError("Cannot take the trigonometric functions of a unit with an angle exponent not equal to 1")
            return self._angle_unit_group
        else:
            raise ValueError("Cannot take the trigonometric functions of a non-dimensionless unit if it has no angle unit group")
    

########################################################
# Compatibility operations
########################################################

    @overload
    def compatible_to(self, other: DimensionGroup) -> bool:
        """Check if two unit groups are compatible. E.g. meter and mile are compatible, but meter and kilogram are not."""
        ...
    @overload
    def compatible_to(self, other: "UnitGroup") -> bool:
        """Check if two unit groups are compatible. E.g. meter and mile are compatible, but meter and kilogram are not."""
        ...

    def compatible_to(self, other: Union[DimensionGroup, "UnitGroup"]) -> bool:
        """Check if two unit groups are compatible. E.g. meter and mile are compatible, but meter and kilogram are not."""
        if isinstance(other, DimensionGroup):   
            return self._dimension_group.compatible_to(other)
        else:
            return self._dimension_group.compatible_to(other._dimension_group)

########################################################
# Comparison operations
########################################################

    def equal_exact(self, other: "UnitGroup") -> bool:
        """
        Check if two unit groups are equal. This is a strong comparison, not a weak one.
        This is used to check if two unit groups are exactly the same, including the order of the unit elements.

        - Have the same Dimension
        - Have the same factor and offset
        - Have the same order of unit symbols
        - Have the same prefixes and exponents for each symbol
        """
        return self._unit_elements == other._unit_elements
    
    def equal_effectively(self, other: "UnitGroup") -> bool:
        """
        Check if two unit groups are equal. This is a weak comparison, not a strong one.
        This is used to check if two unit groups are effectively the same.

        - Have the same dimension
        - Have the same factor and offset        
        """
        return self._dimension_group == other._dimension_group and abs(self._factor / other._factor - 1) < EPSILON and abs(self._offset - other._offset) < EPSILON

    def __eq__(self, other: object) -> bool:
        """Check if two units are equal. This is a strong comparison, not a weak one."""
        if not isinstance(other, UnitGroup):
            return False
        return self.equal_exact(other)
    
    def __ne__(self, other: object) -> bool:
        """Check if two units are not equal."""
        if not isinstance(other, UnitGroup):
            return True
        return not self.equal_exact(other)
    
    def __hash__(self) -> int:
        """Hash based on unit elements, factor, and offset."""
        return hash(self._unit_elements)

########################################################
# Conversions
########################################################

    @overload
    def from_canonical_value(self, canonical_value: int) -> float|int:
        ...
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
    def from_canonical_value(self, canonical_value: np.complex128) -> np.complex128:
        ...
    @overload
    def from_canonical_value(self, canonical_value: float) -> float:
        ...
    @overload
    def from_canonical_value(self, canonical_value: complex) -> complex:
        ...
    @overload
    def from_canonical_value(self, canonical_value: np.ndarray) -> np.ndarray:
        ...
    def from_canonical_value(self, canonical_value: int|np.float64|np.float32|np.float16|np.complex128|float|complex|np.ndarray) -> np.float64|np.float32|np.float16|np.complex128|float|int|complex|np.ndarray:

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
    def to_canonical_value(self, value_in_unit: int) -> float|int:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.float64) -> np.float64:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.float32) -> np.float32:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.float16) -> np.float16:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.complex128) -> np.complex128:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: float) -> float:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: complex) -> complex:
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.ndarray) -> np.ndarray:
        ...
    def to_canonical_value(self, value_in_unit: int|np.float64|np.float32|np.float16|np.complex128|float|complex|np.ndarray) -> np.float64|np.float32|np.float16|np.complex128|float|int|complex|np.ndarray:
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
        
########################################################
# String representation
########################################################

    def __str__(self) -> str:
        return self.format_string(as_fraction=True)
    
    def format_string(self, as_fraction: bool = True) -> str:
        """
        Format the unit as a string.

        Examples:
        - "m^1", TRUE/FALSE -> "m"
        - "m^1, s^-1", TRUE/FALSE -> "m/s"
        - "m^1, s^-1", FALSE/TRUE -> "m*s^-1"
        - "m^1, s^-2, kg^-1", TRUE -> "m/(s^2*kg)"
        - "m^1, s^-2, kg^-1", FALSE -> "m*kg^-1*s^-2"
        - "kg^1, m^1, s^2", TRUE -> "kg*s^2/m"
        - "kg^1, m^1, s^2", FALSE -> "kg*m^-1*s^-2"
        - "kg^1, m^2, s^-3, A^-1", TRUE -> "kg*m^2/(A*s^3)"
        - "kg^1, m^2, s^-3, A^-1", FALSE -> "kg*m^2*A^-1*s^-3"
        
        Args:
            as_fraction: Whether to format the unit as a fraction.
        
        Returns:
            A string representation of the unit.
        """
        if not self._unit_elements:
            return ""
        
        nominator_parts: list[str] = []
        denominator_parts: list[str] = []
        
        for element in self._unit_elements:
            formatted, position = element.format_string(as_fraction)
            if formatted:  # Skip empty strings (zero exponents)
                if position == "nominator":
                    nominator_parts.append(formatted)
                else:  # position == "denominator"
                    denominator_parts.append(formatted)
        
        # Build the final string
        if not nominator_parts and not denominator_parts:
            return ""
        elif not denominator_parts:
            return "*".join(nominator_parts)
        elif not nominator_parts:
            if len(denominator_parts) == 1:
                return f"1/{denominator_parts[0]}"
            else:
                return f"1/({'*'.join(denominator_parts)})"
        else:
            if len(denominator_parts) == 1:
                return f"{'*'.join(nominator_parts)}/{denominator_parts[0]}"
            else:
                return f"{'*'.join(nominator_parts)}/({'*'.join(denominator_parts)})"

########################################################
# Reducing and combining unit elements
########################################################

    def reduced_unit_group(self) -> "UnitGroup":
        return self
    
    @staticmethod
    def _combine_same_unit_symbols_and_prefixes(unit_elements: Sequence[UnitElement]) -> Sequence[UnitElement]:
        """
        Combine unit elements that have the same unit symbol AND prefix.
        Units with different prefixes are NOT combined to avoid factor ambiguity.
        """
        unit_elements_dict: dict[tuple[str, UnitSymbol], UnitElement] = {}
        for element in unit_elements:
            prefix: str = element.prefix
            unit_symbol: UnitSymbol = element.unit_symbol
            key = (prefix, unit_symbol)
            if key in unit_elements_dict:
                existing_unit_element: UnitElement = unit_elements_dict[key]
                new_exponent: float = existing_unit_element.exponent + element.exponent
                if abs(new_exponent) < 1e-12:  # treat as zero
                    unit_elements_dict.pop(key)
                else:
                    unit_elements_dict[key] = UnitElement(
                        prefix=existing_unit_element.prefix,
                        unit_symbol=unit_symbol,
                        exponent=new_exponent
                    )
            else:
                if abs(element.exponent) >= 1e-12:
                    unit_elements_dict[key] = element
        return tuple(unit_elements_dict.values())
    
    @staticmethod
    def combine_same_unit_symbols(unit_elements: Sequence[UnitElement]) -> tuple[Sequence[UnitElement], float]:
        """
        Combine unit elements that have the same unit symbol.

        Prefixes may change, so a factor is returned to account for this.
        
        Examples:
        - ({"", "m": 1}, {"", "m": 1}, {"µ", "s": 2}) -> ({"", "m": 2}, {"µ", "s": 2}), 1
        - ({"k", "m": 1}, {"", "m": 1}, {"µ", "s": 2}) -> ({"k", "m": 2}, {"µ", "s": 2}), 1
        - ({"", "m": 1}, {"k", "m": 1}, {"µ", "s": 2}) -> ({"", "m": 2}, {"µ", "s": 2}), 0.001
        
        """

        unit_elements_dict: dict[UnitSymbol, UnitElement] = {}
        factor: float = 1.0
        for element in unit_elements:
            unit_symbol: UnitSymbol = element.unit_symbol
            if unit_symbol in unit_elements_dict:
                existing_unit_element: UnitElement = unit_elements_dict[unit_symbol]
                new_exponent: float = existing_unit_element.exponent + element.exponent
                if new_exponent.is_integer():
                    new_exponent = int(new_exponent)
                else:
                    new_exponent = float(new_exponent)
                if new_exponent != 0:
                    combined_unit_element: UnitElement = UnitElement(
                        prefix=existing_unit_element.prefix,
                        unit_symbol=unit_symbol,
                        exponent=new_exponent
                    )
                    factor *= existing_unit_element.canonical_factor / element.canonical_factor
                    unit_elements_dict[unit_symbol] = combined_unit_element
                else:
                    factor *= existing_unit_element.canonical_factor / element.canonical_factor
                    unit_elements_dict.pop(unit_symbol)
            else:
                unit_elements_dict[unit_symbol] = element

        return tuple(unit_elements_dict.values()), factor

########################################################
# Suggesting units
########################################################

    @staticmethod
    def suggest_units(dimension_group: DimensionGroup, canonical_value: float|None, must_include: set[UnitElement]|list[UnitElement]=set(), n: int = 1000) -> Tuple["UnitGroup", list["UnitGroup"]]:
        """
        Suggest units for a given dimension and canonical value, optimized for readability.
        
        The method finds units that make the numerical value "nice" - preferably 1-9 digits
        before the decimal point with minimal zeros and decimal places.
        
        This method works without relying on NamedUnit by generating all possible combinations
        of unit symbols and prefixes.
        
        Args:
            quantity: The unit dimension to match
            canonical_value: The value in canonical units to optimize for
            must_include: Unit elements that must be included in suggestions
            n: Maximum number of suggestions to return
            
        Returns:
            Tuple of (best_unit, list_of_alternative_units)
        """
        from .unit_symbol import UnitSymbol
        from .utils import PREFIX_PAIRS
        from .unit_element import UnitElement
        
        if canonical_value is None:
            # If no value given, just return a basic canonical unit
            canonical_unit = dimension_group.canonical_unit
            return canonical_unit, [canonical_unit]
        
        # Generate all possible unit combinations
        compatible_units: list[UnitGroup] = []
        seen_units: set[str] = set()  # To avoid duplicates
        
        # Get all unit symbols
        all_symbols = list(UnitSymbol)
        
        # Generate single-symbol units with various prefixes
        for symbol in all_symbols:
            # Try without prefix
            try:
                unit_group = UnitGroup((UnitElement("", symbol, 1.0),))
                if unit_group.compatible_to(dimension_group):
                    unit_str = unit_group.format_string(as_fraction=False)
                    if unit_str not in seen_units:
                        compatible_units.append(unit_group)
                        seen_units.add(unit_str)
            except:
                pass
            
            # Try with different prefixes
            for prefix in PREFIX_PAIRS.keys():
                if prefix == "":  # Skip empty prefix as we already tried it
                    continue
                try:
                    unit_group = UnitGroup((UnitElement(prefix, symbol, 1.0),))
                    if unit_group.compatible_to(dimension_group):
                        unit_str = unit_group.format_string(as_fraction=False)
                        if unit_str not in seen_units:
                            compatible_units.append(unit_group)
                            seen_units.add(unit_str)
                except:
                    pass
        
        # Generate compound units for common derived dimensions
        # This is more complex but we'll do some basic combinations
        if len(compatible_units) == 0:
            # Try some basic compound units
            for symbol1 in all_symbols:
                for symbol2 in all_symbols:
                    if symbol1 == symbol2:
                        continue
                    # Try combinations like m/s, kg*m/s^2, etc.
                    for exp1 in [1, -1, 2, -2]:
                        for exp2 in [1, -1, 2, -2]:
                            try:
                                unit_group = UnitGroup((
                                    UnitElement("", symbol1, exp1),
                                    UnitElement("", symbol2, exp2)
                                ))
                                if unit_group.compatible_to(dimension_group):
                                    unit_str = unit_group.format_string(as_fraction=False)
                                    if unit_str not in seen_units:
                                        compatible_units.append(unit_group)
                                        seen_units.add(unit_str)
                                        if len(compatible_units) >= 50:  # Limit to prevent explosion
                                            break
                            except:
                                pass
                        if len(compatible_units) >= 50:
                            break
                    if len(compatible_units) >= 50:
                        break
                if len(compatible_units) >= 50:
                    break
        
        # Add must_include elements if specified
        if must_include:
            required_elements = list(must_include) if isinstance(must_include, set) else must_include
            for element in required_elements:
                try:
                    unit_group = UnitGroup((element,))
                    if unit_group.compatible_to(dimension_group):
                        unit_str = unit_group.format_string(as_fraction=False)
                        if unit_str not in seen_units:
                            compatible_units.append(unit_group)
                            seen_units.add(unit_str)
                except:
                    pass
        
        if not compatible_units:
            # Fallback to canonical unit
            canonical_unit = dimension_group.canonical_unit
            return canonical_unit, [canonical_unit]
        
        # Score each unit based on how "nice" the resulting value would be
        scored_units: list[Tuple[float, UnitGroup, str]] = []
        
        for unit_group in compatible_units:
            try:
                # Convert canonical value to this unit
                value_in_unit = unit_group.from_canonical_value(canonical_value)
                value_str = f"{value_in_unit:g}"  # Format without unnecessary trailing zeros
                unit_str = unit_group.format_string(as_fraction=False)
                
                # Calculate score (lower is better)
                score = UnitGroup._calculate_value_score(value_str, unit_str)
                scored_units.append((score, unit_group, value_str))
                
            except (ValueError, ZeroDivisionError, OverflowError):
                # Skip units that cause conversion errors
                continue
        
        # Sort by score (ascending - lower is better)
        scored_units.sort(key=lambda x: x[0])
        
        # Return top n suggestions
        suggestions = [unit for _, unit, _ in scored_units[:n]]
        
        if suggestions:
            return suggestions[0], suggestions
        else:
            # Fallback
            canonical_unit = dimension_group.canonical_unit
            return canonical_unit, [canonical_unit]
    
    @staticmethod
    def _calculate_value_score(value_str: str, unit_str: str) -> float:
        """
        Calculate a score for how "nice" a value/unit combination is.
        Lower scores are better.
        """
        import math
        import re
        
        # Parse the numeric value
        try:
            numeric_value = float(value_str)
        except ValueError:
            return float('inf')  # Invalid numbers get worst score
        
        if numeric_value == 0:
            return 0.0  # Zero is perfect
        
        score = 0.0
        
        # 1. Prefer values between 1 and 999 (1-3 digits before decimal)
        abs_value = abs(numeric_value)
        if 1 <= abs_value < 10:
            score += 0.0  # Perfect range
        elif 10 <= abs_value < 100:
            score += 1.0  # Good range
        elif 100 <= abs_value < 1000:
            score += 2.0  # Acceptable range
        elif 0.1 <= abs_value < 1:
            score += 3.0  # Small decimals
        elif 0.01 <= abs_value < 0.1:
            score += 5.0  # Very small decimals
        else:
            # Very large or very small numbers
            log_value = math.log10(abs_value)
            score += 10.0 + abs(log_value)
        
        # 2. Penalize decimal places and zeros
        if '.' in value_str:
            decimal_part = value_str.split('.')[1]
            score += len(decimal_part) * 0.5  # Each decimal place adds penalty
            score += decimal_part.count('0') * 0.3  # Extra penalty for trailing zeros
        
        # 3. Penalize leading/trailing zeros in the whole number
        if 'e' not in value_str.lower():  # Avoid scientific notation
            score += value_str.count('0') * 0.2
        
        # 4. Prefer shorter unit strings
        score += len(unit_str) * 0.1
        
        # 5. Total character count (value + unit)
        score += len(value_str) * 0.1
        
        # 6. Heavily penalize scientific notation
        if 'e' in value_str.lower():
            score += 20.0
        
        # 7. HEAVILY penalize fractional exponents in unit strings
        # Look for patterns like ^0.5, ^-0.5, ^1.5, ^-1.5, etc.
        fractional_exponent_pattern = r'\^(-?\d*\.\d+)'
        fractional_matches = re.findall(fractional_exponent_pattern, unit_str)
        
        for exponent_str in fractional_matches:
            try:
                exponent = float(exponent_str)
                # Heavy penalty for fractional exponents
                score += 50.0  # Base penalty for any fractional exponent
                
                # Extra penalty for "weird" fractions
                if abs(exponent - 0.5) < EPSILON or abs(exponent + 0.5) < EPSILON:
                    score += 10.0  # ±0.5 is less bad than other fractions
                elif abs(exponent - 1.5) < EPSILON or abs(exponent + 1.5) < EPSILON:
                    score += 20.0  # ±1.5 is worse
                else:
                    score += 30.0  # Other fractions are worst
                    
            except ValueError:
                # If we can't parse the exponent, it's definitely bad
                score += 100.0
        
        # 8. Prefer simple integer exponents
        # Look for patterns like ^2, ^3, ^-1, ^-2, etc.
        integer_exponent_pattern = r'\^(-?\d+)(?!\.\d)'  # Negative lookbehind to avoid matching 2.5 as 2
        integer_matches = re.findall(integer_exponent_pattern, unit_str)
        
        for exponent_str in integer_matches:
            try:
                exponent = int(exponent_str)
                # Small penalty for high integer exponents
                if abs(exponent) > 3:
                    score += abs(exponent) * 0.5
            except ValueError:
                pass
        
        # 9. Prefer simple compound units over complex derived units
        # Count the number of different unit symbols in the string
        # Use Unicode letter pattern to handle symbols like ν, Ω, μ, etc.
        unit_symbols = re.findall(r'[\w\u0080-\uFFFF]+', unit_str)
        # Filter to keep only letter-containing symbols (exclude pure numbers)
        unit_symbols = [s for s in unit_symbols if re.search(r'[A-Za-z\u0080-\uFFFF]', s)]
        
        if len(unit_symbols) > 3:
            score += (len(unit_symbols) - 3) * 2.0  # Penalty for too many different symbols
        
        # 10. Use UnitSymbolTag to prefer SI units over non-SI units
        from .unit_symbol import UnitSymbol, UNIT_SYMBOL_TAG
        
        for symbol in unit_symbols:
            # Remove common prefixes to get the base symbol
            base_symbol = symbol
            common_prefixes = ['k', 'M', 'G', 'T', 'c', 'm', 'μ', 'µ', 'n', 'p', 'f', 'a']
            for prefix in common_prefixes:
                if symbol.startswith(prefix) and len(symbol) > len(prefix):
                    base_symbol = symbol[len(prefix):]
                    break
            
            try:
                unit_symbol = UnitSymbol.from_symbol(base_symbol)
                tags = unit_symbol.value.symbol_tags
                
                # Apply scoring based on tags
                if UNIT_SYMBOL_TAG.SI_BASE_UNIT in tags:
                    score += 0.0  # Best score for SI base units
                elif UNIT_SYMBOL_TAG.SI_DERIVED_UNIT in tags:
                    score += 0.1  # Slightly less preferred than base units
                elif UNIT_SYMBOL_TAG.SI_BASE_UNIT_EQUIVALENT in tags:
                    score += 0.2  # Even less preferred
                elif UNIT_SYMBOL_TAG.NON_SI_SYSTEM in tags:
                    score += 2.0  # Penalty for non-SI units
                elif UNIT_SYMBOL_TAG.US_CUSTOMARY_SYSTEM in tags or UNIT_SYMBOL_TAG.IMPERIAL_SYSTEM in tags:
                    score += 3.0  # Higher penalty for non-metric systems
                else:
                    score += 1.0  # Default penalty for unclassified units
                    
            except ValueError:
                # Symbol not found in UnitSymbol enum - likely obscure
                score += 5.0  # Heavy penalty for unknown symbols
        
        return score
    
########################################################
# Other
########################################################
    
    @property
    def is_dimensionless(self) -> bool:
        return len(self._unit_elements) == 0
    
    @property
    def includes_log_level(self) -> bool:
        return self.dimension_group.includes_log_level
    
    @property
    def includes_angle(self) -> bool:
        return self.dimension_group.includes_angle
    
    @property
    def unit_elements(self) -> Sequence[UnitElement]:
        return self._unit_elements

########################################################
# Preset units
########################################################
 
DIMENSIONLESS_UNIT_GROUP: Final["UnitGroup"] = UnitGroup(())