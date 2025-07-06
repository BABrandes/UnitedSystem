"""String formatting and representation for RealUnitedScalar."""

from typing import TYPE_CHECKING, Union
import math

if TYPE_CHECKING:
    from ...real_united_scalar import RealUnitedScalar
    from ....units.simple.simple_dimension import SimpleDimension
    from ....units.simple.simple_unit import SimpleUnit

# Move imports out of methods to fix import issues
try:
    from ....units.simple.simple_unit import SimpleUnit
    from ....units.simple.simple_dimension import SimpleDimension
    _IMPORTS_AVAILABLE = True
except ImportError:
    _IMPORTS_AVAILABLE = False

def _to_std_str(value: float) -> str:
    """Convert float to standard string representation."""
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    elif math.isnan(value):
        return "NaN"
    else:
        return str(value)

class FormattingMixin:
    """String formatting and representation for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "SimpleDimension"
    display_unit: Union["SimpleUnit", None]

    def __str__(self) -> str:
        """Return string representation using format method."""
        return self.format(None, 3)

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return f"RealUnitedScalar(canonical_value={self.canonical_value}, dimension={self.dimension}, display_unit={self.display_unit})"

    def format(self, unit: Union[str, "SimpleUnit", None] = None, decimals: int = 0, do_not_print_unit: bool = False) -> str:
        """
        Format the scalar as a string with optional unit specification.
        
        Args:
            unit: Unit to display the value in. If None, uses auto-suggestion.
            decimals: Number of decimal places to show.
            do_not_print_unit: If True, only show the numerical value.
            
        Returns:
            Formatted string representation.
        """
        if not _IMPORTS_AVAILABLE:
            # Fallback to just showing the value
            return f"{self.canonical_value:.{decimals}f}"
        
        if unit is None:
            if isinstance(self.dimension, SimpleDimension):
                _unit, _ = SimpleUnit.suggest_units(self.dimension, self.canonical_value)
            else:
                _unit = self.dimension.canonical_unit
        elif isinstance(unit, str):
            _unit: SimpleUnit = SimpleUnit.parse_string(unit)
        else:
            _unit: SimpleUnit = unit
        
        if not _unit.compatible_to(self.dimension):
            raise ValueError(f"The requested display unit {_unit} is not compatible with the scalar's dimension {self.dimension}")
        
        # Convert canonical value to the desired unit
        display_value = _unit.from_canonical_value(self.canonical_value)
        
        # Format the value and remove trailing zeros
        if display_value == 0:
            value_str = "0"
        else:
            value_str = f"{display_value:.{decimals}f}".rstrip('0').rstrip('.')
        
        if do_not_print_unit:
            return value_str
        else:
            return f"{value_str} {_unit}"

    def simple_str(self) -> str:
        """Simple string representation using display unit if available."""
        if self.display_unit is None:
            return _to_std_str(self.canonical_value)
        else:
            display_value = self.display_unit.from_canonical_value(self.canonical_value)
            return f"{_to_std_str(display_value)} {self.display_unit}" 