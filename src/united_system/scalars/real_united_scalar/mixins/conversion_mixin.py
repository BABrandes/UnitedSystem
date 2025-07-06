"""Unit conversion methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..real_united_scalar import RealUnitedScalar
    from ....units.simple.simple_unit import SimpleUnit
    from ....units.simple.simple_dimension import SimpleDimension

class ConversionMixin:
    """Unit conversion functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "SimpleDimension"
    display_unit: Union["SimpleUnit", None]

    def suggest_display_unit(self) -> "RealUnitedScalar":
        """Return a new scalar with a suggested display unit."""
        from ....units.simple.simple_unit import SimpleUnit
        from ..real_united_scalar import RealUnitedScalar
        
        suggested_unit, _ = SimpleUnit.suggest_units(self.dimension, self.canonical_value, n=1)
        return RealUnitedScalar(self.canonical_value, self.dimension, suggested_unit)

    def in_unit(self, new_display_unit: "SimpleUnit") -> "RealUnitedScalar":
        """Return a new scalar with a different display unit."""
        from ..real_united_scalar import RealUnitedScalar
        
        if new_display_unit.compatible_to(self.dimension):
            # Use the new display unit
            return RealUnitedScalar(self.canonical_value, self.dimension, new_display_unit)
        else:
            raise ValueError(f"The suggested display unit {new_display_unit} is not compatible with the canonical dimension {self.dimension}")

    def to_canonical_unit(self) -> "RealUnitedScalar":
        """Return a new scalar with the canonical unit as display unit."""
        from ..real_united_scalar import RealUnitedScalar
        
        canonical_unit = self.dimension.canonical_unit
        return RealUnitedScalar(self.canonical_value, self.dimension, canonical_unit)

    def to_unit(self, unit: "SimpleUnit") -> "RealUnitedScalar":
        """Return a new scalar with the specified unit as display unit."""
        from ..real_united_scalar import RealUnitedScalar
        
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")

    def as_float_in_unit(self, unit: "SimpleUnit") -> float:
        """Convert to float in specified unit. Only works for numeric values."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        
        match self.canonical_value:
            case float() | int():
                return unit.from_canonical_value(self.canonical_value)
            case complex():
                raise ValueError(f"Cannot convert complex value to float: {self.canonical_value}")
            case _:
                raise ValueError(f"Cannot convert non-numeric value to float: {self.canonical_value}")
            
    def as_float_in_canonical_unit(self) -> float:
        """Get the value of this scalar in SI units."""
        unit = self.dimension.canonical_unit
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_value)
    
    def as_float_in_display_unit(self) -> float:
        """Get the value of this scalar in the display unit."""
        if self.display_unit is None:
            raise ValueError(f"Display unit is not set for United_Value: {self}")
        else:
            return self.display_unit.from_canonical_value(self.canonical_value)