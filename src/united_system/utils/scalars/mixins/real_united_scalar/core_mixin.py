"""Core functionality for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....unit import Unit
    from .....dimension import Dimension

class RealUnitedScalarCore:
    """Core functionality for RealUnitedScalar."""
    
    # These will be provided by the dataclass
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

# dimension is provided by the dataclass field, no property needed

    def compatible_to(self, *args: "RealUnitedScalar") -> bool:
        """Check if this scalar is compatible with other scalars."""
        if len(args) == 0:
            return True
        # Handle both single argument and multiple arguments
        if len(args) == 1:
            return self.dimension == args[0].dimension
        # Multiple arguments - check all
        for arg in args:
            if arg.dimension != self.dimension:
                return False
        return True

    def abs(self) -> "RealUnitedScalar":
        """Return the absolute value of this scalar."""
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(abs(self.canonical_value), self.dimension, self._display_unit)

    def __reduce__(self):
        """Custom serialization for multiprocessing."""
        from .....real_united_scalar import RealUnitedScalar
        return (RealUnitedScalar, (self.canonical_value, self.dimension, self._display_unit)) 