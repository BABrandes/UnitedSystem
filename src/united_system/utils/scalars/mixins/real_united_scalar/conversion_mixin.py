"""Unit conversion methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from .....unit import Unit
    from .....dimension import Dimension
    from .....real_united_scalar import RealUnitedScalar

class ConversionMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Unit conversion functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    def value_in_canonical_unit(self) -> float:
        """
        Get the scalar value in canonical units.
        
        Returns:
            The scalar value as a float in canonical units
            
        Example:
            >>> scalar = RealUnitedScalar(1000.0, Unit.parse_string("g"))
            >>> scalar.value_in_canonical_unit()
            1.0  # Returns value in kg (canonical unit for mass)
        """
        return self.canonical_value

    def value_in_display_unit(self) -> float:
        """
        Get the scalar value in display units.
        
        Returns:
            The scalar value as a float in display units
            
        Raises:
            ValueError: If no display unit is set
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> scalar.value_in_display_unit()
            1.0
            >>> scalar_no_display = RealUnitedScalar.create_from_canonical_value(1.0, mass_dim)
            >>> scalar_no_display.value_in_display_unit()  # Raises ValueError
        """
        if self._display_unit is None:
            raise ValueError("No display unit set")
        return self._display_unit.from_canonical_value(self.canonical_value)

    def to_canonical_unit(self) -> "RealUnitedScalar":
        """
        Convert to canonical unit representation.
        
        Returns:
            A new RealUnitedScalar with the canonical unit as display unit
            
        Example:
            >>> scalar = RealUnitedScalar(1000.0, Unit.parse_string("g"))
            >>> result = scalar.to_canonical_unit()
            >>> result.canonical_value
            1.0
            >>> result._display_unit
            Unit("kg")  # Canonical unit for mass
        """
        from .....real_united_scalar import RealUnitedScalar
        canonical_unit = self.dimension.canonical_unit
        return RealUnitedScalar.create_from_canonical_value(self.canonical_value, self.dimension, canonical_unit)

    def to_canonical(self, with_display_unit: bool = False) -> "RealUnitedScalar":
        """
        Return a new scalar with no display unit (canonical).
        
        Args:
            with_display_unit: If True, set display unit to canonical unit
            
        Returns:
            A new RealUnitedScalar with or without display unit
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> result = scalar.to_canonical()
            >>> result._display_unit is None
            True
            >>> result_with_unit = scalar.to_canonical(with_display_unit=True)
            >>> result_with_unit._display_unit
            Unit("kg")
        """
        from .....real_united_scalar import RealUnitedScalar
        display_unit = self.dimension.canonical_unit if with_display_unit else None
        return RealUnitedScalar.create_from_canonical_value(self.canonical_value, self.dimension, display_unit)

    def to_unit(self, unit: "Unit") -> "RealUnitedScalar":
        """Return a new scalar with the specified unit as display unit."""
        from .....real_united_scalar import RealUnitedScalar
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"The suggested display unit {unit} is not compatible with the canonical dimension {self.dimension}")
        return RealUnitedScalar.create_from_canonical_value(self.canonical_value, self.dimension, unit)

    def float_in_unit(self, unit: "Unit") -> float:
        """
        Convert the scalar to a float value in the specified unit.
        
        Args:
            unit: The unit to convert to
            
        Returns:
            The scalar value as a float in the specified unit
            
        Raises:
            ValueError: If the unit is not compatible with the scalar's dimension
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> scalar.float_in_unit(Unit.parse_string("g"))
            1000.0
        """
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_value)

    def canonical_float(self) -> float:
        """
        Get the scalar value as a float in canonical units.
        
        Returns:
            The scalar value as a float in canonical units
            
        Example:
            >>> scalar = RealUnitedScalar(1000.0, Unit.parse_string("g"))
            >>> scalar.canonical_float()
            1.0  # Returns value in kg (canonical unit for mass)
        """
        return self.canonical_value

    def display_float(self) -> float:
        """
        Get the scalar value as a float in display units.
        
        Returns:
            The scalar value as a float in display units
            
        Raises:
            ValueError: If no display unit is set
            
        Example:
            >>> scalar = RealUnitedScalar(1.0, Unit.parse_string("kg"))
            >>> scalar.display_float()
            1.0
            >>> scalar_no_display = RealUnitedScalar.create_from_canonical_value(1.0, mass_dim)
            >>> scalar_no_display.display_float()  # Raises ValueError
        """
        if self._display_unit is None:
            raise ValueError("No display unit set")
        return self._display_unit.from_canonical_value(self.canonical_value)

    def in_unit(self, unit: "Unit") -> "RealUnitedScalar":
        """Alias for to_unit method."""
        return self.to_unit(unit)