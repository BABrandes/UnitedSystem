from dataclasses import dataclass
from typing import Literal, Tuple, TYPE_CHECKING
from .utils import PREFIX_PAIRS
import re

if TYPE_CHECKING:
    from .unit_symbol import UnitSymbol
    from ..dimension import Dimension

@dataclass(frozen=True, slots=True)
class SimpleUnitElement:
    prefix: str
    unit_symbol: "UnitSymbol"
    exponent: float

    @property
    def unit_dimension(self) -> "Dimension":
        return self.unit_symbol.value.named_simple_dimension.simple_dimension
    
    @property
    def canonical_factor(self) -> float:
        if self.prefix == "":
            prefix_factor: float = 1.0
        elif self.prefix in PREFIX_PAIRS:
            prefix_factor: float = PREFIX_PAIRS[self.prefix]
        else:
            raise ValueError(f"Invalid prefix: {self.prefix}")
        return (self.unit_symbol.value.factor * prefix_factor) ** self.exponent
    
    @property
    def canonical_offset(self) -> float:
        return self.unit_symbol.value.offset

    def format_string(self, no_fraction: bool) -> Tuple[str, Literal["nominator", "denominator"]]:
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
            if self.exponent < 0 and no_fraction:
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
                if self.exponent < 0 and no_fraction:
                    exponent_str = "^-1"
                else:
                    exponent_str = ""
            else:
                if self.exponent < 0 and no_fraction:
                    exponent_str = f"^-{exponent_str}"
                else:
                    exponent_str = "^" + exponent_str
                
        if self.exponent < 0 and not no_fraction:
            return f"{self.prefix}{symbol_str}{exponent_str}", "denominator"
        else:
            return f"{self.prefix}{symbol_str}{exponent_str}", "nominator"
    
    @classmethod
    def parse_string(cls, unit_string: str, nominator_or_denominator: Literal["nominator", "denominator"]) -> "SimpleUnitElement":
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
            return cls("", unit_symbol, exponent)
        except ValueError:
            pass
        
        # Try with prefixes (longest first to prioritize "da" over "d", etc.)
        for prefix in sorted(PREFIX_PAIRS.keys(), key=len, reverse=True):
            if unit_part.startswith(prefix):
                try:
                    from .unit_symbol import UnitSymbol
                    unit_symbol = UnitSymbol.from_symbol(unit_part[len(prefix):])
                    return cls(prefix, unit_symbol, exponent)
                except ValueError:
                    continue
        
        # If no match found, raise error
        raise ValueError(f"Could not parse unit string: {unit_string}")