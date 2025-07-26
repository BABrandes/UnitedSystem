"""Utility methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional
import math
import numpy as np
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from ...._units_and_dimension.unit import Unit
    from ...._units_and_dimension.dimension import Dimension

class UtilityMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Utility methods for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]
    
    # compatible_to method is provided by core mixin

    def is_positive(self) -> bool:
        """Check if the scalar is positive."""
        return np.greater(self.canonical_value, 0)
    
    def is_negative(self) -> bool:
        """Check if the scalar is negative."""
        return np.less(self.canonical_value, 0)
    
    def is_zero(self) -> bool:
        """Check if the scalar is zero."""
        return self.canonical_value == 0
    
    def is_non_zero(self) -> bool:
        """Check if the scalar is non-zero."""
        return self.canonical_value != 0

    def is_nan(self) -> bool:
        """Check if the scalar is NaN."""
        return math.isnan(self.canonical_value)
    
    def is_finite(self) -> bool:
        """Check if the scalar is finite."""
        return math.isfinite(self.canonical_value)
    
    def is_infinite(self) -> bool:
        """Check if the scalar is infinite."""
        return math.isinf(self.canonical_value)
    
    def is_positive_or_zero(self) -> bool:
        """Check if the scalar is positive or zero."""
        return np.greater_equal(self.canonical_value, 0)
    
    def is_negative_or_zero(self) -> bool:
        """Check if the scalar is negative or zero."""
        return np.less_equal(self.canonical_value, 0)
    
    def is_strictly_positive(self) -> bool:
        """Check if the scalar is strictly positive."""
        return np.greater(self.canonical_value, 0)
    
    def is_strictly_negative(self) -> bool:
        """Check if the scalar is strictly negative."""
        return np.less(self.canonical_value, 0)

    def is_in_range(self, min_val: "RealUnitedScalar", max_val: "RealUnitedScalar") -> bool:
        """Check if this value is within the specified range (inclusive)."""
        if self.dimension != min_val.dimension or self.dimension != max_val.dimension:
            raise ValueError("Cannot check range with incompatible dimensions")
        if not self.is_finite():
            raise ValueError("Cannot check range for non-numeric values")
        return min_val <= self <= max_val

    def clamp(self, min_val: "RealUnitedScalar|float", max_val: "RealUnitedScalar|float") -> "RealUnitedScalar":
        """Clamp the scalar to a range."""
        from .....real_united_scalar import RealUnitedScalar
        
        # Handle float inputs by converting to canonical values
        if hasattr(min_val, 'canonical_value') and hasattr(min_val, 'dimension'):
            min_canonical: float = min_val.canonical_value # type: ignore
            if self.dimension != min_val.dimension: # type: ignore
                raise ValueError(f"min_val {min_val} is not compatible with {self}")
        else:
            min_canonical: float = min_val # type: ignore
            
        if hasattr(max_val, 'canonical_value') and hasattr(max_val, 'dimension'):
            max_canonical: float = max_val.canonical_value # type: ignore
            if self.dimension != max_val.dimension: # type: ignore
                raise ValueError(f"max_val {max_val} is not compatible with {self}")
        else:
            max_canonical: float = max_val # type: ignore
            
        if not self.is_finite():
            raise ValueError(f"Cannot clamp non-finite value {self}")

        clamped_value: float = np.clip(self.canonical_value, min_canonical, max_canonical) # type: ignore
        return RealUnitedScalar.create_from_canonical_value(clamped_value, self.dimension, self._display_unit)

    def min(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Return the minimum of this scalar and another."""
        from .....real_united_scalar import RealUnitedScalar
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare {self} and {other} because they have incompatible dimensions.")
        
        if self.canonical_value <= other.canonical_value:
            return RealUnitedScalar.create_from_canonical_value(self.canonical_value, self.dimension, self._display_unit)
        else:
            return RealUnitedScalar.create_from_canonical_value(other.canonical_value, other.dimension, other.unit)
    
    def max(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Return the maximum of this scalar and another."""
        from .....real_united_scalar import RealUnitedScalar
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare {self} and {other} because they have incompatible dimensions.")
        
        if self.canonical_value >= other.canonical_value:
            return RealUnitedScalar.create_from_canonical_value(self.canonical_value, self.dimension, self._display_unit)
        else:
            return RealUnitedScalar.create_from_canonical_value(other.canonical_value, other.dimension, other.unit)

    @classmethod
    def sum(cls, values: list["RealUnitedScalar"]) -> "RealUnitedScalar":
        """Sum a list of scalars."""
        from .....real_united_scalar import RealUnitedScalar
        if not values:
            raise ValueError("Cannot sum an empty list")
        
        # Check all values are compatible
        first = values[0]
        for i, v in enumerate(values[1:], 1):
            if not first.compatible_to(v):
                raise ValueError(f"Value at index {i} ({v}) is not compatible with first value ({first})")
        
        # Sum all canonical values
        total_canonical = sum(v.canonical_value for v in values)
        
        # Return with first value's dimension and display unit
        return RealUnitedScalar.create_from_canonical_value(total_canonical, values[0].dimension, values[0].unit)
    
    @classmethod
    def mean(cls, values: list["RealUnitedScalar"]) -> "RealUnitedScalar":
        """Calculate the mean of a list of scalars."""
        from .....real_united_scalar import RealUnitedScalar
        if not values:
            raise ValueError("Cannot calculate mean of an empty list")
        
        # Check all values are compatible
        first = values[0]
        for i, v in enumerate(values[1:], 1):
            if not first.compatible_to(v):
                raise ValueError(f"Value at index {i} ({v}) is not compatible with first value ({first})")
        
        # Calculate mean of canonical values
        total_canonical = sum(v.canonical_value for v in values)
        
        # Return with first value's dimension and display unit
        return RealUnitedScalar.create_from_canonical_value(total_canonical / len(values), values[0].dimension, values[0].unit) 