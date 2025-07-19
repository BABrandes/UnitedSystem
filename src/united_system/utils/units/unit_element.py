from dataclasses import dataclass, field
from typing import Literal, Tuple, Optional
from .utils import PREFIX_PAIRS
import re

from .dimension_group import DimensionGroup
from .unit_symbol import UnitSymbol

_CACHE__SIMPLE_UNIT_ELEMENT: dict[str, "UnitElement"] = {}

def clear_simple_unit_element_cache():
    """Clear the simple unit element cache."""
    global _CACHE__SIMPLE_UNIT_ELEMENT
    _CACHE__SIMPLE_UNIT_ELEMENT.clear()

@dataclass(frozen=True, slots=True)
class UnitElement:
    """
    A simple unit element is a unit symbol with a prefix and an exponent.

    It serves as a building block for units.

    Examples:
    - "m" -> SimpleUnitElement("", UnitSymbol.METER, 1)
    - "km" -> SimpleUnitElement("k", UnitSymbol.METER, 1)
    - "s" -> SimpleUnitElement("", UnitSymbol.SECOND, -1)
    - "nV^4" -> SimpleUnitElement("n", UnitSymbol.VOLT, 4)
    """

    prefix: str
    unit_symbol: UnitSymbol
    exponent: float
    _dimension_group: Optional[DimensionGroup] = field(default=None, init=False)

    @property
    def dimension_group(self) -> DimensionGroup:
        return self.unit_symbol.named_quantity.dimension_group
    
    @property
    def canonical_factor(self) -> float:
        if self.prefix == "":
            prefix_factor: float = 1.0
        elif self.prefix in PREFIX_PAIRS:
            prefix_factor: float = PREFIX_PAIRS[self.prefix]
        else:
            raise ValueError(f"Invalid prefix: {self.prefix}")
        return (self.unit_symbol.value.factor * prefix_factor) ** self.exponent
    
########################################################
# Properties
########################################################
    
    @property
    def canonical_offset(self) -> float:
        return self.unit_symbol.value.offset
    
    @property
    def dimension(self) -> DimensionGroup:
        if self._dimension_group is None:
            object.__setattr__(self, "_dimension_group", self.unit_symbol.named_quantity.dimension_group ** self.exponent)
        if self._dimension_group is None:
            raise AssertionError("Dimension group is not set")
        return self._dimension_group
    
########################################################
# Arithmetic
########################################################

    def invert(self) -> "UnitElement":
        return UnitElement(self.prefix, self.unit_symbol, -self.exponent)
    
    def pow(self, exponent: float|int) -> "UnitElement":
        return UnitElement(self.prefix, self.unit_symbol, self.exponent * exponent)

########################################################
# String representation
########################################################

    def format_string(self, as_fraction: bool) -> Tuple[str, Literal["nominator", "denominator"]]:
        """
        Examples:
        - SimpleUnitElement("", UnitSymbol.METER, 1), no_fraction=False -> ("m", "nominator")
        - SimpleUnitElement("k", UnitSymbol.METER, 1) no_fraction=False-> ("km", "nominator")
        - SimpleUnitElement("", UnitSymbol.SECOND, -1) no_fraction=False-> ("s", "denominator")
        - SimpleUnitElement("n", UnitSymbol.VOLT, 4) no_fraction=False-> ("nV^4", "nominator")
        - SimpleUnitElement("k", UnitSymbol.GRAM, 1) no_fraction=False-> ("kg", "nominator")
        - SimpleUnitElement("", UnitSymbol.JOULE, -3) no_fraction=False-> ("J^3", "denominator")
        - SimpleUnitElement("", UnitSymbol.JOULE, -3) no_fraction=True-> ("J^-3", "nominator")
        """
        symbol_str: str = self.unit_symbol.value.symbols[0]
        exponent_str: str = ""

        EXPONENT_MAX_DECIMALS: int = 3

        if self.exponent == 0:
            return ("", "nominator")
        elif abs(self.exponent) == 1:
            # For negative exponents with no_fraction=True, we still need to show the exponent
            if self.exponent < 0 and not as_fraction:
                exponent_str = "^-1"
            else:
                exponent_str = ""
        else:
            exponent_str = str(abs(self.exponent))
            if exponent_str.count(".") == 1:
                dot_index: int = exponent_str.rfind(".")
                # Remove all characters after the dot index + max decimals + 1
                exponent_str = exponent_str[:dot_index + EXPONENT_MAX_DECIMALS + 1]
                # Remove all trailing zeros
                while exponent_str.endswith("0"):
                    exponent_str = exponent_str[:-1]
                # Remove the dot if it is the last character
                if exponent_str.endswith("."):
                    exponent_str = exponent_str[:-1]
            if exponent_str == "0":
                return ("", "nominator")
            
            if exponent_str == "1":
                if self.exponent < 0 and not as_fraction:
                    exponent_str = "^-1"
                else:
                    exponent_str = ""
            else:
                if self.exponent < 0 and not as_fraction:
                    exponent_str = f"^-{exponent_str}"
                else:
                    exponent_str = "^" + exponent_str
                
        if self.exponent < 0 and as_fraction:
            return f"{self.prefix}{symbol_str}{exponent_str}", "denominator"
        else:
            return f"{self.prefix}{symbol_str}{exponent_str}", "nominator"

########################################################
# Parsing
########################################################

    @classmethod
    def parse_string(cls, unit_string: str, nominator_or_denominator: Literal["nominator", "denominator"]) -> "UnitElement":
        """
        Examples:
        - "m", nominator -> SimpleUnitElement("", UnitSymbol.METER, 1)
        - "km", nominator -> SimpleUnitElement("k", UnitSymbol.METER, 1)
        - "s", denominator -> SimpleUnitElement("", UnitSymbol.SECOND, -1)
        - "nV^4", nominator -> SimpleUnitElement("n", UnitSymbol.VOLT, 4)
        - "kg" -> SimpleUnitElement("k", UnitSymbol.GRAM, 1)
        - "J^3", denominator -> SimpleUnitElement("", UnitSymbol.JOULE, -3)
        - "cm2", nominator -> SimpleUnitElement("c", UnitSymbol.METER, 2)
        - "nV-1", nominator -> SimpleUnitElement("n", UnitSymbol.VOLT, -1)
        """

        # Create a cache key that includes both unit_string and nominator_or_denominator
        cache_key = f"{unit_string}:{nominator_or_denominator}"
        if cache_key in _CACHE__SIMPLE_UNIT_ELEMENT:
            return _CACHE__SIMPLE_UNIT_ELEMENT[cache_key]
        
        # Parse exponent from unit string
        if "^" in unit_string:
            parts = unit_string.split("^")
            if len(parts) != 2:
                raise ValueError(f"Invalid unit string: {unit_string}")
            unit_part = parts[0]
            exponent_part = parts[1]
            try:
                exponent = float(exponent_part)
            except ValueError:
                raise ValueError(f"Invalid exponent in unit string: {unit_string}")
        else:
            # Look for digit suffix (like "cm2" or "nV-1")
            suffix_match = re.search(r'(-?\d*\.?\d+)$', unit_string)
            if suffix_match:
                exponent_part = suffix_match.group(1)
                unit_part = unit_string[:suffix_match.start()]
                try:
                    exponent = float(exponent_part)
                except ValueError:
                    raise ValueError(f"Invalid exponent in unit string: {unit_string}")
            else:
                # No exponent found
                unit_part = unit_string
                exponent = 1.0
        
        # Apply sign based on nominator/denominator
        if nominator_or_denominator == "denominator":
            exponent = -exponent
        
        # Try to find unit symbol without prefix first
        try:
            from .unit_symbol import UnitSymbol
            unit_symbol = UnitSymbol.from_symbol(unit_part)
            simple_unit_element: UnitElement = UnitElement("", unit_symbol, exponent)
            _CACHE__SIMPLE_UNIT_ELEMENT[cache_key] = simple_unit_element
            return simple_unit_element
        except ValueError:
            pass
        
        # Try with prefixes (longest first to prioritize "da" over "d", etc.)
        for prefix in sorted(PREFIX_PAIRS.keys(), key=len, reverse=True):
            if unit_part.startswith(prefix):
                try:
                    from .unit_symbol import UnitSymbol
                    unit_symbol = UnitSymbol.from_symbol(unit_part[len(prefix):])
                    simple_unit_element: UnitElement = UnitElement(prefix, unit_symbol, exponent)
                    _CACHE__SIMPLE_UNIT_ELEMENT[cache_key] = simple_unit_element
                    return simple_unit_element
                except ValueError:
                    continue
        
        # If no match found, raise error
        raise ValueError(f"Could not parse unit string: {unit_string}")