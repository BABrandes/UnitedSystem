"""Arithmetic operations for RealUnitedScalar."""

from typing import Union, TYPE_CHECKING
import math

if TYPE_CHECKING:
    from ...real_united_scalar import RealUnitedScalar
    from ....units.simple.simple_dimension import SimpleDimension
    from ....units.simple.simple_unit import SimpleUnit

class ArithmeticMixin:
    """Arithmetic operations for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "SimpleDimension"
    display_unit: Union["SimpleUnit", None]
    
    # compatible_to and is_nan methods are provided by other mixins

    def __add__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Add two scalars."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot add {self} and {other} because they have incompatible dimensions.")
        
        from ..real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(self.canonical_value + other.canonical_value, self.dimension, self.display_unit)

    def __radd__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Add two scalars (reverse operation)."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot add {other} and {self} because they have incompatible dimensions.")
        
        from ..real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(other.canonical_value + self.canonical_value, other.dimension, other.display_unit)

    def __sub__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Subtract two scalars."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot subtract {other} from {self} because they have incompatible dimensions.")
        
        from ..real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(self.canonical_value - other.canonical_value, self.dimension, self.display_unit)

    def __rsub__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Subtract two scalars (reverse operation)."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot subtract {self} from {other} because they have incompatible dimensions.")
        
        from ..real_united_scalar import RealUnitedScalar
        return RealUnitedScalar(other.canonical_value - self.canonical_value, other.dimension, other.display_unit)

    def __mul__(self, other: "RealUnitedScalar|float|int") -> "RealUnitedScalar":
        """Multiply scalars or scalar by number."""
        from ..real_united_scalar import RealUnitedScalar
        
        if self.is_nan():
            return RealUnitedScalar(math.nan, self.dimension, None)
        
        if hasattr(other, 'canonical_value') and hasattr(other, 'dimension'):
            if math.isnan(other.canonical_value):
                return RealUnitedScalar(math.nan, self.dimension + other.dimension, None)
            
            # Calculate new unit dimension by adding dimensions (multiplication rule)
            new_dimension = self.dimension + other.dimension
            
            match math.isfinite(self.canonical_value), math.isfinite(other.canonical_value):
                case True, True:
                    # Both finite - normal multiplication
                    return RealUnitedScalar(
                        self.canonical_value * other.canonical_value, 
                        new_dimension, 
                        None
                    )
                case True, False:
                    # Other is infinite
                    if self.canonical_value == 0:
                        return RealUnitedScalar(math.nan, new_dimension, None)
                    sign = 1 if self.canonical_value > 0 else -1
                    inf_sign = 1 if other.canonical_value > 0 else -1
                    return RealUnitedScalar(
                        float('inf') * sign * inf_sign, 
                        new_dimension, 
                        None
                    )
                case False, True:
                    # Self is infinite
                    if other.canonical_value == 0:
                        return RealUnitedScalar(math.nan, new_dimension, None)
                    sign = 1 if other.canonical_value > 0 else -1
                    inf_sign = 1 if self.canonical_value > 0 else -1
                    return RealUnitedScalar(
                        float('inf') * sign * inf_sign, 
                        new_dimension, 
                        None
                    )
                case False, False:
                    # Both infinite
                    sign = 1 if (self.canonical_value > 0 and other.canonical_value > 0) or (self.canonical_value < 0 and other.canonical_value < 0) else -1
                    return RealUnitedScalar(
                        float('inf') * sign, 
                        new_dimension, 
                        None
                    )
                    
        elif isinstance(other, (float, int)):
            # Scalar multiplication - dimension stays the same
            if math.isnan(other):
                return RealUnitedScalar(math.nan, self.dimension, None)
            
            match math.isfinite(self.canonical_value), math.isfinite(other):
                case True, True:
                    # Both finite
                    return RealUnitedScalar(
                        self.canonical_value * other, 
                        self.dimension, 
                        self.display_unit
                    )
                case True, False:
                    # Other is infinite
                    if self.canonical_value == 0:
                        return RealUnitedScalar(math.nan, self.dimension, None)
                    sign = 1 if self.canonical_value > 0 else -1
                    inf_sign = 1 if other > 0 else -1
                    return RealUnitedScalar(
                        float('inf') * sign * inf_sign, 
                        self.dimension, 
                        self.display_unit
                    )
                case False, True:
                    # Self is infinite
                    if other == 0:
                        return RealUnitedScalar(math.nan, self.dimension, None)
                    sign = 1 if other > 0 else -1
                    inf_sign = 1 if self.canonical_value > 0 else -1
                    return RealUnitedScalar(
                        float('inf') * sign * inf_sign, 
                        self.dimension, 
                        self.display_unit
                    )
                case False, False:
                    # Both infinite
                    sign = 1 if (self.canonical_value > 0 and other > 0) or (self.canonical_value < 0 and other < 0) else -1
                    return RealUnitedScalar(
                        float('inf') * sign, 
                        self.dimension, 
                        self.display_unit
                    )
        else:
            raise TypeError(f"Cannot multiply RealUnitedScalar with {type(other)}")

    def __rmul__(self, other: "float|int") -> "RealUnitedScalar":
        """Multiply by number (reverse operation)."""
        return self * other

    def __truediv__(self, other: "RealUnitedScalar|float|int") -> "RealUnitedScalar":
        """Divide scalars or scalar by number."""
        from ..real_united_scalar import RealUnitedScalar
        
        if self.is_nan():
            return RealUnitedScalar(math.nan, self.dimension, None)
        
        if hasattr(other, 'canonical_value') and hasattr(other, 'dimension'):
            if math.isnan(other.canonical_value):
                return RealUnitedScalar(math.nan, self.dimension - other.dimension, None)
            
            # Calculate new unit dimension by subtracting dimensions (division rule)
            new_dimension = self.dimension - other.dimension
            
            match math.isfinite(self.canonical_value), math.isfinite(other.canonical_value):
                case True, True:
                    match self.canonical_value, other.canonical_value:
                        case 0, 0:
                            # 0 / 0 = NaN
                            return RealUnitedScalar(math.nan, new_dimension, None)
                        case _, 0:
                            # finite / 0 = ±inf - no display unit for performance
                            return RealUnitedScalar(
                                float('inf') if self.canonical_value > 0 else float('-inf'), 
                                new_dimension, 
                                None
                            )
                        case 0, _:
                            # 0 / finite = 0 - no display unit for performance
                            return RealUnitedScalar(0, new_dimension, None)
                        case _, _:
                            # finite / finite - no display unit for performance
                            return RealUnitedScalar(
                                self.canonical_value / other.canonical_value, 
                                new_dimension, 
                                None
                            )
                case True, False:
                    # finite / inf = 0 - no display unit for performance
                    return RealUnitedScalar(0, new_dimension, None)
                case False, True:
                    # inf / finite = ±inf - no display unit for performance
                    if other.canonical_value == 0:
                        return RealUnitedScalar(math.nan, new_dimension, None)
                    sign = 1 if (self.canonical_value > 0 and other.canonical_value > 0) or (self.canonical_value < 0 and other.canonical_value < 0) else -1
                    return RealUnitedScalar(
                        float('inf') * sign, 
                        new_dimension, 
                        None
                    )
                case False, False:
                    # inf / inf = NaN
                    return RealUnitedScalar(math.nan, new_dimension, None)
                    
        elif isinstance(other, (float, int)):
            # Scalar division - dimension stays the same
            if math.isnan(other):
                return RealUnitedScalar(math.nan, self.dimension, None)
            
            match math.isfinite(self.canonical_value), math.isfinite(other):
                case True, True:
                    match self.canonical_value, other:
                        case 0, 0:
                            # 0 / 0 = NaN
                            return RealUnitedScalar(math.nan, self.dimension, None)
                        case _, 0:
                            # finite / 0 = ±inf - preserve units!
                            return RealUnitedScalar(
                                float('inf') if self.canonical_value > 0 else float('-inf'), 
                                self.dimension, 
                                self.display_unit
                            )
                        case 0, _:
                            # 0 / finite = 0
                            return RealUnitedScalar(0, self.dimension, self.display_unit)
                        case _, _:
                            # finite / finite
                            return RealUnitedScalar(
                                self.canonical_value / other, 
                                self.dimension, 
                                self.display_unit
                            )
                case True, False:
                    # finite / inf = 0
                    return RealUnitedScalar(0, self.dimension, self.display_unit)
                case False, True:
                    # inf / finite = ±inf - preserve units!
                    if other == 0:
                        return RealUnitedScalar(math.nan, self.dimension, None)
                    sign = 1 if (self.canonical_value > 0 and other > 0) or (self.canonical_value < 0 and other < 0) else -1
                    return RealUnitedScalar(
                        float('inf') * sign, 
                        self.dimension, 
                        self.display_unit
                    )
                case False, False:
                    # inf / inf = NaN
                    return RealUnitedScalar(math.nan, self.dimension, None)
        else:
            raise TypeError(f"Cannot divide RealUnitedScalar by {type(other)}")

    def __rtruediv__(self, other: "float|int") -> "RealUnitedScalar":
        """Divide number by scalar (reverse operation)."""
        from ..real_united_scalar import RealUnitedScalar
        
        # Create dimensionless dimension for reverse division
        from ....units.named_simple_dimensions import NamedSimpleDimension
        dimensionless = NamedSimpleDimension.NUMBER.simple_unit_dimension
        new_dimension = dimensionless - self.dimension
        
        if self.is_nan():
            return RealUnitedScalar(math.nan, new_dimension, None)
        
        if math.isnan(other):
            return RealUnitedScalar(math.nan, new_dimension, None)
        
        match math.isfinite(self.canonical_value), math.isfinite(other):
            case True, True:
                match self.canonical_value, other:
                    case 0, 0:
                        # 0 / 0 = NaN
                        return RealUnitedScalar(math.nan, new_dimension, None)
                    case 0, _:
                        # finite / 0 = ±inf - no display unit for reverse division
                        return RealUnitedScalar(
                            float('inf') if other > 0 else float('-inf'), 
                            new_dimension, 
                            None
                        )
                    case _, 0:
                        # 0 / finite = 0 - no display unit for performance
                        return RealUnitedScalar(0, new_dimension, None)
                    case _, _:
                        # finite / finite - no display unit for performance
                        return RealUnitedScalar(
                            other / self.canonical_value, 
                            new_dimension, 
                            None
                        )
            case True, False:
                # inf / finite = ±inf - no display unit for performance
                if self.canonical_value == 0:
                    return RealUnitedScalar(math.nan, new_dimension, None)
                sign = 1 if (other > 0 and self.canonical_value > 0) or (other < 0 and self.canonical_value < 0) else -1
                return RealUnitedScalar(
                    float('inf') * sign, 
                    new_dimension, 
                    None
                )
            case False, True:
                # finite / inf = 0 - no display unit for performance
                return RealUnitedScalar(0, new_dimension, None)
            case False, False:
                # inf / inf = NaN
                return RealUnitedScalar(math.nan, new_dimension, None)

    def __pow__(self, exponent: float) -> "RealUnitedScalar":
        """Raise scalar to a power."""
        from ..real_united_scalar import RealUnitedScalar
        
        if math.isnan(exponent):
            return RealUnitedScalar(math.nan, self.dimension * exponent, None)
        
        if exponent == 0:
            # Any number to the power of 0 is 1 (dimensionless)
            from ....units.named_simple_dimensions import NamedSimpleDimension
            return RealUnitedScalar(1.0, NamedSimpleDimension.NUMBER.simple_unit_dimension, None)
        
        if exponent == 1:
            # Return a copy of self
            return RealUnitedScalar(self.canonical_value, self.dimension, self.display_unit)
        
        # For other exponents, calculate the result
        new_value = self.canonical_value ** exponent
        new_dimension = self.dimension * exponent
        
        return RealUnitedScalar(new_value, new_dimension, None) 