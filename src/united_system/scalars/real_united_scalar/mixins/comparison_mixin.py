"""Comparison operations for RealUnitedScalar."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..real_united_scalar import RealUnitedScalar
    from ....units.simple.simple_unit import SimpleUnit

class ComparisonMixin:
    """Comparison operations for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    display_unit: "SimpleUnit"
    
    # compatible_to method is provided by core mixin

    def __eq__(self, other: "RealUnitedScalar") -> bool:
        """Check if two scalars are equal."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare dimensions of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value == other.canonical_value

    def __ne__(self, other: "RealUnitedScalar") -> bool:
        """Check if two scalars are not equal."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare dimensions of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value != other.canonical_value

    def __lt__(self, other: "RealUnitedScalar") -> bool:
        """Check if this scalar is less than another."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare dimensions of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value < other.canonical_value

    def __le__(self, other: "RealUnitedScalar") -> bool:
        """Check if this scalar is less than or equal to another."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare dimensions of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value <= other.canonical_value

    def __gt__(self, other: "RealUnitedScalar") -> bool:
        """Check if this scalar is greater than another."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare dimensions of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value > other.canonical_value

    def __ge__(self, other: "RealUnitedScalar") -> bool:
        """Check if this scalar is greater than or equal to another."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare dimensions of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value >= other.canonical_value 