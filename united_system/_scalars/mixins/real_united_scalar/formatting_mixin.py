"""String formatting and representation for RealUnitedScalar."""

from typing import Union, Optional, TYPE_CHECKING
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from ...._units_and_dimension.unit import Unit
    from ...._units_and_dimension.dimension import Dimension
    from ...._scalars.real_united_scalar import RealUnitedScalar

class FormattingMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """String formatting and representation for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    def __str__(self) -> str:
        """
        Return string representation using format method.
        
        Returns:
            String representation with 3 decimal places and trailing zeros.
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> str(scalar)
            '1.000 kg'
        """
        return self.format(None, 3, trailing_zeros=True)

    def __repr__(self) -> str:
        """Return detailed string representation for debugging."""
        return f"RealUnitedScalar(canonical_value={self.canonical_value}, dimension={self.dimension}, display_unit={self._display_unit})"

    def format(self, unit: Union[str, "Unit", None] = None, max_decimals: int = 0, trailing_zeros: bool = False) -> str:
        """
        Format the scalar as a string with optional unit specification.
        
        Args:
            unit: Unit to display the value in. If None, uses auto-suggestion.
            max_decimals: Maximum number of decimal places to show.
            trailing_zeros: If True, show trailing zeros up to max_decimals.
            
        Returns:
            Formatted string representation.
            
        Raises:
            ValueError: If the unit is not compatible with the scalar's dimension.
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> scalar.format("g", max_decimals=2, trailing_zeros=True)
            '1000.00 g'
            >>> scalar.format(max_decimals=1)
            '1.0 kg'
            >>> scalar.format("g", max_decimals=0)
            '1000 g'
        """
        from ...._units_and_dimension.unit import Unit
        if unit is None:
            _unit: "Unit" = self.unit.reduced
        elif isinstance(unit, str):
            _unit: "Unit" = Unit(unit)
        else:
            _unit: "Unit" = unit
        if _unit.dimension != self.dimension:
            raise ValueError(f"The requested display unit {_unit} is not compatible with the scalar's dimension {self.dimension}")
        display_value = _unit.from_canonical_value(self.canonical_value)
        # Always show the requested number of decimal places
        value_str = f"{display_value:.{max_decimals}f}" if trailing_zeros else f"{display_value}"
        unit_str = _unit.format_string(as_fraction=False)
        return f"{value_str} {unit_str}"

    def simple_str(self) -> str:
        """
        Return a simple string representation without trailing zeros.
        
        Returns:
            Simple string representation with natural formatting.
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> scalar.simple_str()
            '1 kg'
            >>> scalar = RealUnitedScalar(1.5, Unit.parse_string("kg"))
            >>> scalar.simple_str()
            '1.5 kg'
        """
        return self.format(None, 0, trailing_zeros=False) 