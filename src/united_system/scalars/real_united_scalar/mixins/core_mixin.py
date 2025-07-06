"""Core functionality for RealUnitedScalar."""

from typing import Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ....units.base_classes.base_unit import BaseUnit
from ....units.base_classes.base_dimension import BaseDimension
from ....units.simple.simple_dimension import SimpleDimension
from ....units.simple.simple_unit import SimpleUnit
from ....units.named_simple_dimensions import NamedSimpleDimension
from ....utils import str_to_float
import math

if TYPE_CHECKING:
    from ...real_united_scalar import RealUnitedScalar


class RealUnitedScalarCore:
    """Core functionality for RealUnitedScalar."""
    
    # These will be provided by the dataclass
    canonical_value: float
    dimension: SimpleDimension
    display_unit: Union[SimpleUnit, None]
    
    def __post_init__(self):
        """Validate the scalar after initialization."""
        if self.display_unit is not None and not self.display_unit.compatible_to(self.dimension):
            raise ValueError(f"The display unit {self.display_unit} is not compatible with the canonical dimension {self.dimension}.")

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
        from ...real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(abs(self.canonical_value), self.dimension, self.display_unit)

    def __reduce__(self):
        """Custom serialization for multiprocessing."""
        from ...real_united_scalar import RealUnitedScalar
        return (RealUnitedScalar, (self.canonical_value, self.dimension, self.display_unit)) 