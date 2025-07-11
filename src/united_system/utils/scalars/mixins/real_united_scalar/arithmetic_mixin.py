"""Arithmetic operations for RealUnitedScalar."""

from typing import TYPE_CHECKING, Optional
import math

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....dimension import Dimension
    from .....unit import Unit

class ArithmeticMixin:
    """Arithmetic operations for RealUnitedScalar."""

    # These will be provided by the core class
    canonical_value: float
    dimension: Dimension
    _display_unit: Optional[Unit]
    
    # compatible_to and is_nan methods are provided by other mixins

    def __add__(self, other: RealUnitedScalar) -> RealUnitedScalar:
        """Add two scalars."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add {self} and {other} because they have incompatible dimensions.")
        
        return RealUnitedScalar(self.canonical_value + other.canonical_value, self.dimension, self._display_unit)

    def __radd__(self, other: RealUnitedScalar) -> RealUnitedScalar:
        """Add two scalars (reverse operation)."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add {other} and {self} because they have incompatible dimensions.")
        
        return RealUnitedScalar(other.canonical_value + self.canonical_value, other.dimension, other.display_unit)

    def __sub__(self, other: RealUnitedScalar) -> RealUnitedScalar:
        """Subtract two scalars."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract {other} from {self} because they have incompatible dimensions.")
        
        return RealUnitedScalar(self.canonical_value - other.canonical_value, self.dimension, self._display_unit)

    def __rsub__(self, other: RealUnitedScalar) -> RealUnitedScalar:
        """Subtract two scalars (reverse operation)."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract {self} from {other} because they have incompatible dimensions.")
        
        return RealUnitedScalar(other.canonical_value - self.canonical_value, other.dimension, other.display_unit)

    def __mul__(self, other: RealUnitedScalar|float|int) -> RealUnitedScalar:
        """Multiply scalars or scalar by number."""

        if isinstance(other, RealUnitedScalar):
            new_dimension: Dimension = self.dimension + other.dimension
            value: float = other.canonical_value
        else:
            new_dimension: Dimension = self.dimension
            value: float = other

        # Check for NaN
        if math.isnan(self.canonical_value) or math.isnan(value):
            return RealUnitedScalar(math.nan, new_dimension, None)
        
        # Multiplication
        match math.isfinite(self.canonical_value), math.isfinite(value):
            case True, True:
                # Both finite - normal multiplication
                return RealUnitedScalar(self.canonical_value * value, new_dimension, None)
            case True, False:
                # Other is infinite
                if self.canonical_value == 0:
                    return RealUnitedScalar(math.nan, new_dimension, None)
                sign = 1 if self.canonical_value > 0 else -1
                inf_sign = 1 if value > 0 else -1
                return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None)
            case False, True:
                # Self is infinite
                if other == 0:
                    return RealUnitedScalar(math.nan, new_dimension, None)
                sign = 1 if value > 0 else -1
                inf_sign = 1 if self.canonical_value > 0 else -1
                return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None)
            case False, False:
                # Both infinite
                sign = 1 if (self.canonical_value > 0 and value > 0) or (self.canonical_value < 0 and value < 0) else -1
                return RealUnitedScalar(float('inf') * sign, new_dimension, None)
            case _:
                raise ValueError(f"Cannot multiply {self} and {other} because: {math.isfinite(self.canonical_value)} and {math.isfinite(value)} are not finite")

    def __rmul__(self, other: float|int) -> RealUnitedScalar:
        """Multiply by number (reverse operation)."""
        return self * other

    def __truediv__(self, other: RealUnitedScalar|float|int) -> RealUnitedScalar:
        """Divide scalars or scalar by number."""
        if isinstance(other, RealUnitedScalar):
            new_dimension: Dimension = self.dimension - other.dimension
            value: float = other.canonical_value
        else:
            new_dimension: Dimension = self.dimension
            value: float = other

        # Check for NaN
        if math.isnan(self.canonical_value) or math.isnan(value):
            return RealUnitedScalar(math.nan, new_dimension, None)
        
        # Division
        match math.isfinite(self.canonical_value), math.isfinite(value):
            case True, True: # (x / y)
                # Both finite - normal division
                return RealUnitedScalar(self.canonical_value / value, new_dimension, None)
            case True, False: # (x / inf)
                # Other is infinite
                if self.canonical_value == 0:
                    return RealUnitedScalar(math.nan, new_dimension, None)
                else:
                    return RealUnitedScalar(0, new_dimension, None)
            case False, True: # (inf / x)
                # Self is infinite
                if other == 0:
                    return RealUnitedScalar(math.nan, new_dimension, None)
                else:
                    sign = 1 if value > 0 else -1
                    inf_sign = 1 if self.canonical_value > 0 else -1
                return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None)
            case False, False: # (inf / inf)
                # Both infinite
                return RealUnitedScalar(math.nan, new_dimension, None)
            case _:
                raise ValueError(f"Cannot divide {self} by {other} because: {math.isfinite(self.canonical_value)} and {math.isfinite(value)} are not finite")

    def __rtruediv__(self, other: RealUnitedScalar|float|int) -> RealUnitedScalar:
        """Divide number by scalar."""
        if isinstance(other, RealUnitedScalar):
            new_dimension: Dimension = other.dimension - self.dimension
            value: float = other.canonical_value
        else:
            new_dimension: Dimension = self.dimension * -1
            value: float = other

        # Check for NaN
        if math.isnan(self.canonical_value) or math.isnan(value):
            return RealUnitedScalar(math.nan, new_dimension, None)
        
        # Division
        match math.isfinite(value), math.isfinite(self.canonical_value):
            case True, True: # (y / x)
                # Both finite - normal division
                return RealUnitedScalar(value / self.canonical_value, new_dimension, None)
            case True, False: # (y / inf)
                # Other is infinite
                if self.canonical_value == 0: # 0/inf = NaN
                    return RealUnitedScalar(math.nan, new_dimension, None)
                else: # y/inf = 0
                    return RealUnitedScalar(0, new_dimension, None)
            case False, True: # (inf / x)
                # Self is infinite
                if other == 0: # inf/0 = NaN
                    return RealUnitedScalar(math.nan, new_dimension, None)
                else: # inf/x = inf
                    sign = 1 if value > 0 else -1
                    inf_sign = 1 if self.canonical_value > 0 else -1
                    return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None)
            case False, False: # (inf / inf)
                # Both infinite
                return RealUnitedScalar(math.nan, new_dimension, None)
            case _:
                raise ValueError(f"Cannot divide {other} by {self} because: {math.isfinite(value)} and {math.isfinite(self.canonical_value)} are not finite")

    def __pow__(self, exponent: float) -> RealUnitedScalar:
        """Raise scalar to a power."""
        
        if math.isnan(exponent):
            return RealUnitedScalar(math.nan, self.dimension * float(exponent), None)
        
        if exponent == 0:
            # Any number to the power of 0 is 1 (dimensionless)
            from ....units.named_simple_dimensions import NamedSimpleDimension
            return RealUnitedScalar(1.0, NamedSimpleDimension.NUMBER.dimension, None)
        
        if exponent == 1:
            # Return a copy of self
            return RealUnitedScalar(self.canonical_value, self.dimension, self._display_unit)
        
        # For other exponents, calculate the result
        new_value = self.canonical_value ** exponent
        new_dimension = self.dimension * float(exponent)
        
        return RealUnitedScalar(new_value, new_dimension, None) 