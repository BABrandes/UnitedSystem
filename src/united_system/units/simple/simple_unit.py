from dataclasses import dataclass, field
from typing import Final, Tuple, List, overload, Union
import re
import numpy as np

from ..base_classes.base_unit import BaseUnit
from ..unit_symbol import UnitSymbol
from ..base_classes.base_dimension import BaseDimension
from .simple_dimension import SimpleDimension
from .simple_unit_element import SimpleUnitElement

# Cache for parsed units (outside the dataclass to avoid slots conflict)
_SIMPLE_UNIT_CACHE: dict[str, "SimpleUnit"] = {}

@dataclass(frozen=True, slots=True)
class SimpleUnit(BaseUnit[SimpleDimension, "SimpleUnit"]):
    unit_elements: Final[tuple[SimpleUnitElement, ...]] = field(init=False)

    def __init__(self, unit_elements: tuple[SimpleUnitElement, ...]):
        object.__setattr__(self, "unit_elements", unit_elements)
        quantity_exponents: list[float] = [0.0] * 7
        pseudo_quantity_exponents: list[int] = [0] * 2
        factor: float = 1
        offset: float = 0
        for unit_element in self.unit_elements:

            # Set factor and offset
            factor *= unit_element.canonical_factor
            new_offset: float = unit_element.canonical_offset
            if offset != 0 and new_offset != 0:
                raise ValueError("Cannot have two non-zero offsets in the same unit")
            offset = new_offset
            
            # Add quantity exponents (regardless of prefix)
            unit_symbol: UnitSymbol = unit_element.unit_symbol
            exponent: float = unit_element.exponent
            for exponent_index, qty_exp in enumerate(unit_symbol.named_simple_dimension.simple_dimension.dimension_exponents):
                quantity_exponents[exponent_index] += qty_exp * exponent
            for exponent_index, pseudo_exp in enumerate(unit_symbol.named_simple_dimension.simple_dimension.pseudo_dimension_exponents):
                pseudo_quantity_exponents[exponent_index] += pseudo_exp * exponent

            if unit_symbol.value.offset != 0:
                if offset != 0:
                    raise ValueError("Cannot have two non-zero offsets in the same unit")
                offset = unit_symbol.value.offset

        object.__setattr__(self, "dimension", SimpleDimension.create(quantity_exponents, pseudo_quantity_exponents))
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
    def create(cls, arg: Union[List["SimpleUnit"], set["SimpleUnit"]]) -> "SimpleUnit":
        ...
    @classmethod
    def create(cls, arg: Union[SimpleUnitElement, "SimpleUnit", List["SimpleUnit"], set["SimpleUnit"]], *args: Union[None, SimpleUnitElement, "SimpleUnit"]) -> "SimpleUnit":

        items: list[SimpleUnitElement] = []
        match arg:
            case list() | set():
                for unit in arg:
                    items.extend(unit.unit_elements)
            case SimpleUnit():
                items.extend(arg.unit_elements)
            case tuple():
                x: Tuple[str, UnitSymbol, float] = arg
                items.append(SimpleUnitElement(x[0], x[1], x[2]))
        match args:
            case tuple():
                for v in args:
                    match v:
                        case tuple():
                            items.append(SimpleUnitElement(v[0], v[1], v[2]))
                        case SimpleUnit():
                            items.extend(v.unit_elements)
                        case _:
                            raise ValueError(f"Invalid units: {x}")
            case _:
                raise ValueError(f"Invalid units: {args}")
        return cls(tuple(items))
    
    @classmethod
    def create_empty(cls) -> "SimpleUnit":
        return cls(())

    @classmethod
    def parse_string(cls, unit_string: str) -> "SimpleUnit":
        """
        Parse a unit string into a Simple_Unit.
        
        Examples:
        - "m" -> Simple_Unit(("", {"meter"}, 1))
        - "km" -> Simple_Unit(("k", {"meter"}, 1))
        - "m/s" -> Simple_Unit(("", {"meter"}, 1), ("s", {"second"}, -1))
        - "V*m/ns^2" -> Simple_Unit(("", {"volt"}, 1), ("meter", {"meter"}, 1), ("n", {"second"}, -2))
        """

        if unit_string in _SIMPLE_UNIT_CACHE:
            return _SIMPLE_UNIT_CACHE[unit_string]

        unit_string = unit_string.strip()
        if unit_string == "" or unit_string == "1":
            return cls(())
        unit_string_parts: list[str] = re.findall(r'[^*/]+|[*/][^*/]+', unit_string)
        unit_elements: list[SimpleUnitElement] = []
        for index, unit_string_part in enumerate(unit_string_parts):
            
            # Skip "1" as it represents dimensionless (no contribution)
            if unit_string_part == "1":
                if index == 0:
                    continue
                else:
                    raise ValueError(f"Invalid unit string: {unit_string}")

            # Determine the initial exponent based on if the unit part is in numerator or denominator
            if unit_string_part[0:2] == "1/":
                unit_string_part = unit_string_part[2:]
                unit_elements.append(SimpleUnitElement.parse_string(unit_string_part, "denominator"))
            elif unit_string_part[0] == "/":
                unit_string_part = unit_string_part[1:]
                unit_elements.append(SimpleUnitElement.parse_string(unit_string_part, "denominator"))
            elif unit_string_part[0] == "*":
                unit_string_part = unit_string_part[1:]
                unit_elements.append(SimpleUnitElement.parse_string(unit_string_part, "nominator"))
            else:
                unit_elements.append(SimpleUnitElement.parse_string(unit_string_part, "nominator"))

        # Convert tuples to SimpleUnitElement objects
        unit_elements_as_tuple: tuple[SimpleUnitElement, ...] = tuple(unit_elements)
        simple_unit: SimpleUnit = cls(unit_elements_as_tuple)
        _SIMPLE_UNIT_CACHE[unit_string] = simple_unit
        return simple_unit

    def compatible_to(self, other: Union["BaseUnit", "BaseDimension"]) -> bool:
        if isinstance(other, BaseDimension):
            return self.dimension == other
        elif isinstance(other, SimpleUnit):
            return self.dimension == other.dimension
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
        new_unit_elements: list[SimpleUnitElement] = []
        for element in self.unit_elements:
            new_unit_elements.append(SimpleUnitElement(element.prefix, element.unit_symbol, element.exponent * exponent))
        return SimpleUnit(tuple(new_unit_elements))
        
    def __str__(self) -> str:
        return self.format_string(no_fraction=False)
        
    def reduced_unit(self) -> "SimpleUnit":
        return self

    def format_string(self, no_fraction: bool) -> str:
        if not self.unit_elements:
            return ""
        
        nominator_parts: list[str] = []
        denominator_parts: list[str] = []
        
        for element in self.unit_elements:
            formatted, position = element.format_string(no_fraction)
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
            return "1/" + "*".join(denominator_parts)
        else:
            return "*".join(nominator_parts) + "/" + "*".join(denominator_parts)
        
    @staticmethod
    def suggest_units(dimension: SimpleDimension, canonical_value: float|None, must_include: set[SimpleUnitElement]|list[SimpleUnitElement]=set(), n: int = 1000) -> Tuple["SimpleUnit", list["SimpleUnit"]]:
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
        from ..unit_symbol import UnitSymbol
        from ..utils import _PREFIX_PAIRS
        from .simple_unit_element import SimpleUnitElement
        import math
        
        if canonical_value is None:
            # If no value given, just return a basic canonical unit
            canonical_unit = dimension.canonical_unit
            return canonical_unit, [canonical_unit]
        
        # Generate all possible unit combinations
        compatible_units: list[SimpleUnit] = []
        seen_units: set[str] = set()  # To avoid duplicates
        
        # Get all unit symbols
        all_symbols = list(UnitSymbol)
        
        # Generate single-symbol units with various prefixes
        for symbol in all_symbols:
            # Try without prefix
            try:
                unit = SimpleUnit((SimpleUnitElement("", symbol, 1.0),))
                if unit.compatible_to(dimension):
                    unit_str = unit.format_string(no_fraction=False)
                    if unit_str not in seen_units:
                        compatible_units.append(unit)
                        seen_units.add(unit_str)
            except:
                pass
            
            # Try with different prefixes
            for prefix in _PREFIX_PAIRS.keys():
                if prefix == "":  # Skip empty prefix as we already tried it
                    continue
                try:
                    unit = SimpleUnit((SimpleUnitElement(prefix, symbol, 1.0),))
                    if unit.compatible_to(dimension):
                        unit_str = unit.format_string(no_fraction=False)
                        if unit_str not in seen_units:
                            compatible_units.append(unit)
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
                                unit = SimpleUnit((
                                    SimpleUnitElement("", symbol1, exp1),
                                    SimpleUnitElement("", symbol2, exp2)
                                ))
                                if unit.compatible_to(dimension):
                                    unit_str = unit.format_string(no_fraction=False)
                                    if unit_str not in seen_units:
                                        compatible_units.append(unit)
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
                    unit = SimpleUnit((element,))
                    if unit.compatible_to(dimension):
                        unit_str = unit.format_string(no_fraction=False)
                        if unit_str not in seen_units:
                            compatible_units.append(unit)
                            seen_units.add(unit_str)
                except:
                    pass
        
        if not compatible_units:
            # Fallback to canonical unit
            canonical_unit = dimension.canonical_unit
            return canonical_unit, [canonical_unit]
        
        # Score each unit based on how "nice" the resulting value would be
        scored_units: list[Tuple[float, SimpleUnit, str]] = []
        
        for unit in compatible_units:
            try:
                # Convert canonical value to this unit
                value_in_unit = unit.from_canonical_value(canonical_value)
                value_str = f"{value_in_unit:g}"  # Format without unnecessary trailing zeros
                unit_str = unit.format_string(no_fraction=False)
                
                # Calculate score (lower is better)
                score = SimpleUnit._calculate_value_score(value_str, unit_str)
                scored_units.append((score, unit, value_str))
                
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
            canonical_unit = dimension.canonical_unit
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
                if abs(exponent - 0.5) < 1e-10 or abs(exponent + 0.5) < 1e-10:
                    score += 10.0  # ±0.5 is less bad than other fractions
                elif abs(exponent - 1.5) < 1e-10 or abs(exponent + 1.5) < 1e-10:
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
        from ..unit_symbol import UnitSymbol, UnitSymbolTag
        
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
                if UnitSymbolTag.SI_BASE_UNIT in tags:
                    score += 0.0  # Best score for SI base units
                elif UnitSymbolTag.SI_DERIVED_UNIT in tags:
                    score += 0.1  # Slightly less preferred than base units
                elif UnitSymbolTag.SI_BASE_UNIT_EQUIVALENT in tags:
                    score += 0.2  # Even less preferred
                elif UnitSymbolTag.NON_SI_SYSTEM in tags:
                    score += 2.0  # Penalty for non-SI units
                elif UnitSymbolTag.US_CUSTOMARY_SYSTEM in tags or UnitSymbolTag.IMPERIAL_SYSTEM in tags:
                    score += 3.0  # Higher penalty for non-metric systems
                else:
                    score += 1.0  # Default penalty for unclassified units
                    
            except ValueError:
                # Symbol not found in UnitSymbol enum - likely obscure
                score += 5.0  # Heavy penalty for unknown symbols
        
        return score