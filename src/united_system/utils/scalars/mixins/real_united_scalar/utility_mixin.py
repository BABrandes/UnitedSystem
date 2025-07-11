"""Utility methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional
import math
import numpy as np

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....unit import Unit
    from .....dimension import Dimension

class UtilityMixin:
    """Utility methods for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: Dimension
    _display_unit: Optional[Unit]
    
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

    def is_in_range(self, min_val: RealUnitedScalar, max_val: RealUnitedScalar) -> bool:
        """Check if this value is within the specified range (inclusive)."""
        if self.dimension != min_val.dimension or self.dimension != max_val.dimension:
            raise ValueError("Cannot check range with incompatible dimensions")
        if not self.is_finite():
            raise ValueError("Cannot check range for non-numeric values")
        return min_val <= self <= max_val

    def clamp(self, min_val: RealUnitedScalar, max_val: RealUnitedScalar) -> RealUnitedScalar:
        """Clamp the scalar to a range."""
        
        if self.dimension != min_val.dimension or self.dimension != max_val.dimension:
            raise ValueError(f"min_val {min_val} and max_val {max_val} are not compatible with {self}")
        if not self.is_finite():
            raise ValueError(f"max_val {max_val} is not compatible with {self}")

        clamped_value: float = np.clip(self.canonical_value, min_val.canonical_value, max_val.canonical_value)
        return RealUnitedScalar(clamped_value, self.dimension, self._display_unit)

    def min(self, other: RealUnitedScalar) -> RealUnitedScalar:
        """Return the minimum of this scalar and another."""
        
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare {self} and {other} because they have incompatible dimensions.")
        
        if self.canonical_value <= other.canonical_value:
            return RealUnitedScalar(self.canonical_value, self.dimension, self._display_unit)
        else:
            return RealUnitedScalar(other.canonical_value, other.dimension, other.display_unit)
    
    def max(self, other: RealUnitedScalar) -> RealUnitedScalar:
        """Return the maximum of this scalar and another."""
        
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot compare {self} and {other} because they have incompatible dimensions.")
        
        if self.canonical_value >= other.canonical_value:
            return RealUnitedScalar(self.canonical_value, self.dimension, self._display_unit)
        else:
            return RealUnitedScalar(other.canonical_value, other.dimension, other.display_unit)

    @classmethod
    def sum(cls, values: list[RealUnitedScalar]) -> RealUnitedScalar:
        """Sum a list of scalars."""
        
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
        return RealUnitedScalar(total_canonical, values[0].dimension, values[0].display_unit)
    
    @classmethod
    def mean(cls, values: list["RealUnitedScalar"]) -> "RealUnitedScalar":
        """Calculate the mean of a list of scalars."""
        
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
        return RealUnitedScalar(total_canonical / len(values), values[0].dimension, values[0].display_unit) 