"""Unit conversion methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....unit import Unit
    from .....dimension import Dimension

class ConversionMixin:
    """Unit conversion functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: Dimension
    _display_unit: Optional[Unit]

    def with_suggested_unit(self) -> RealUnitedScalar:
        """Return a new scalar with a suggested display unit."""
        
        suggested_unit, _ = Unit.suggest_units(self.dimension, float(self.canonical_value), n=1)
        return RealUnitedScalar(self.canonical_value, self.dimension, suggested_unit)

    def to_canonical_unit(self) -> RealUnitedScalar:
        """Return a new scalar with the canonical unit as display unit."""
        
        canonical_unit = self.dimension.canonical_unit
        return RealUnitedScalar(self.canonical_value, self.dimension, canonical_unit)

    def to_unit(self, unit: Unit) -> "RealUnitedScalar":
        """Return a new scalar with the specified unit as display unit."""
        
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        return RealUnitedScalar(self.canonical_value, self.dimension, unit)

    def as_float_in_unit(self, unit: Unit) -> float:
        """Convert to float in specified unit. Only works for numeric values."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        
        return unit.from_canonical_value(self.canonical_value)
            
    def as_float_in_canonical_unit(self) -> float:
        """Get the value of this scalar in SI units."""
        unit = self.dimension.canonical_unit
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_value)
    
    def as_float_in_display_unit(self) -> float:
        """Get the value of this scalar in the display unit."""
        if self._display_unit is None:
            raise ValueError("This scalar has no display unit")
        return self._display_unit.from_canonical_value(self.canonical_value)