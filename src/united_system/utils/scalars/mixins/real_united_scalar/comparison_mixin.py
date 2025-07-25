"""Comparison operations for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from .....unit import Unit
    from .....dimension import Dimension
    from .....real_united_scalar import RealUnitedScalar

class ComparisonMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Comparison operations for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]
    
    # compatible_to method is provided by core mixin

    def __eq__(self, other: object) -> bool:
        """Check if two scalars are equal."""
        from .....real_united_scalar import RealUnitedScalar
        if not isinstance(other, RealUnitedScalar):
            return False
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare dimensions of different types: {self._display_unit} and {other.display_unit}")
        return self.canonical_value == other.canonical_value # type: ignore

    def __ne__(self, other: object) -> bool:
        """Check if two scalars are not equal."""
        from .....real_united_scalar import RealUnitedScalar
        if not isinstance(other, RealUnitedScalar):
            return True
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare dimensions of different types: {self._display_unit} and {other.display_unit}")
        return self.canonical_value != other.canonical_value # type: ignore

    def __lt__(self, other: object) -> bool:
        """Check if this scalar is less than another."""
        from .....real_united_scalar import RealUnitedScalar
        if not isinstance(other, RealUnitedScalar):
            return False
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare dimensions of different types: {self._display_unit} and {other.display_unit}")
        return self.canonical_value < other.canonical_value # type: ignore

    def __le__(self, other: object) -> bool:
        """Check if this scalar is less than or equal to another."""
        from .....real_united_scalar import RealUnitedScalar
        if not isinstance(other, RealUnitedScalar):
            return False
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare dimensions of different types: {self._display_unit} and {other.display_unit}")
        return self.canonical_value <= other.canonical_value # type: ignore

    def __gt__(self, other: object) -> bool:
        """Check if this scalar is greater than another."""
        from .....real_united_scalar import RealUnitedScalar
        if not isinstance(other, RealUnitedScalar):
            return False
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare dimensions of different types: {self._display_unit} and {other.display_unit}")
        return self.canonical_value > other.canonical_value # type: ignore

    def __ge__(self, other: object) -> bool:
        """Check if this scalar is greater than or equal to another."""
        from .....real_united_scalar import RealUnitedScalar
        if not isinstance(other, RealUnitedScalar):
            return False
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare dimensions of different types: {self._display_unit} and {other.display_unit}")
        return self.canonical_value >= other.canonical_value # type: ignore