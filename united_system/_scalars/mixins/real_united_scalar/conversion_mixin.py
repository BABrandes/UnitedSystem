"""Unit conversion methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from ...._units_and_dimension.unit import Unit
    from ...._units_and_dimension.dimension import Dimension
    from ...._scalars.real_united_scalar import RealUnitedScalar

class ConversionMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Unit conversion functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    def scalar_in_canonical_unit(self) -> "RealUnitedScalar":
        """
        Convert to canonical unit representation.
        
        Returns:
            A new RealUnitedScalar with the canonical unit as display unit
            
        Example:
            >>> scalar = RealUnitedScalar(1000.0, Unit("g"))
            >>> result = scalar.scalar_in_canonical_unit()
            >>> result.canonical_value
            1.0
            >>> result._display_unit
            Unit("kg")  # Canonical unit for mass
        """
        from united_system._scalars.real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(
            self.canonical_value,
            self.dimension,
            self.dimension.canonical_unit)

    def scalar_in_unit(self, unit: "Unit|str") -> "RealUnitedScalar":
        """Return a new scalar with the specified unit as display unit."""
        from united_system._scalars.real_united_scalar import RealUnitedScalar
        if isinstance(unit, str):
            unit = Unit(unit)
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        return RealUnitedScalar(
            self.canonical_value,
            self.dimension,
            unit)