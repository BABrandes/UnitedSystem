"""Core functionality for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional, Union
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from ...._scalars.real_united_scalar import RealUnitedScalar
    from ...._units_and_dimension.unit import Unit
    from ...._units_and_dimension.dimension import Dimension

class RealUnitedScalarCore(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Core functionality for RealUnitedScalar."""
    
    # These will be provided by the dataclass
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

# dimension is provided by the dataclass field, no property needed

    def compatible_to(self, *args: Union["RealUnitedScalar", "Unit", "Dimension"]) -> bool:
        """
        Check if the dimension is compatible with other dimensions.
        Two dimensions are compatible if they have the same subscripts
        and the same proper exponents.
        """
        return Dimension.are_compatible(self.dimension, *args)  

    def abs(self) -> "RealUnitedScalar":
        """Return the absolute value of this scalar."""
        from ...._scalars.real_united_scalar import RealUnitedScalar
        display_unit: Optional["Unit"] = self._display_unit
        return RealUnitedScalar(abs(self.canonical_value), self.dimension, display_unit)

    def __reduce__(self):
        """Custom serialization for multiprocessing."""
        from ...._scalars.real_united_scalar import RealUnitedScalar
        return (RealUnitedScalar, (self.canonical_value, self.dimension, self._display_unit)) 