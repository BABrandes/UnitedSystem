"""Arithmetic operations for RealUnitedScalar."""


from typing import TYPE_CHECKING, Optional, overload, Union
import math
from ...._units_and_dimension.unit import Unit
from ...._units_and_dimension.dimension import Dimension
from ...._units_and_dimension.unit_symbol import UnitSymbol
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from ...._units_and_dimension.dimension import Dimension
    from .....real_united_scalar import RealUnitedScalar

class ArithmeticMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Arithmetic operations for RealUnitedScalar."""

    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]
    
    # compatible_to and is_nan methods are provided by other mixins

########################################################
# Dunder methods #######################################
########################################################

    # Addition # ---------------------------------------

    def __add__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Add two scalars."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add {self} and {other} because they have incompatible dimensions.")
        
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(self.canonical_value + other.canonical_value, self.dimension, self._display_unit)

    def __radd__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Add two scalars (reverse operation)."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot add {other} and {self} because they have incompatible dimensions.")
        
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(other.canonical_value + self.canonical_value, other.dimension, other.unit)

    # Subtraction # -------------------------------------

    def __sub__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Subtract two scalars."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract {other} from {self} because they have incompatible dimensions.")
        
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(self.canonical_value - other.canonical_value, self.dimension, self._display_unit)

    def __rsub__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Subtract two scalars (reverse operation)."""
        if self.dimension != other.dimension:
            raise ValueError(f"Cannot subtract {self} from {other} because they have incompatible dimensions.")
        
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(other.canonical_value - self.canonical_value, other.dimension, other.unit)

    # Multiplication # -----------------------------------

    @overload
    def __mul__(self, other: float|int) -> "RealUnitedScalar":
        ...
    @overload
    def __mul__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        ...
    def __mul__(self, other: Union["RealUnitedScalar", float, int]) -> "RealUnitedScalar":
        """Multiply scalars or scalar by number"""

        if hasattr(other, 'canonical_value') and hasattr(other, 'dimension'):
            new_dimension: "Dimension" = self.dimension * other.dimension # type: ignore
            value: float = other.canonical_value # type: ignore
        else:
            new_dimension: "Dimension" = self.dimension
            value: float = other # type: ignore

        # Check for NaN
        if math.isnan(self.canonical_value) or math.isnan(value): # type: ignore
            from .....real_united_scalar import RealUnitedScalar
            return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
        
        # Multiplication
        match math.isfinite(self.canonical_value), math.isfinite(value): # type: ignore
            case True, True:
                # Both finite - normal multiplication
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(self.canonical_value * value, new_dimension, None) # type: ignore
            case True, False:
                # Other is infinite
                if self.canonical_value == 0:
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
                sign = 1 if self.canonical_value > 0 else -1
                inf_sign = 1 if value > 0 else -1
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None) # type: ignore
            case False, True:
                # Self is infinite
                if other == 0:
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
                sign = 1 if value > 0 else -1
                inf_sign = 1 if self.canonical_value > 0 else -1
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None) # type: ignore
            case False, False:
                # Both infinite
                sign = 1 if (self.canonical_value > 0 and value > 0) or (self.canonical_value < 0 and value < 0) else -1
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(float('inf') * sign, new_dimension, None) # type: ignore
            case _:
                raise ValueError(f"Cannot multiply {self} and {other} because: {math.isfinite(self.canonical_value)} and {math.isfinite(value)} are not finite") # type: ignore
            
    @overload
    def __rmul__(self, other: float|int) -> "RealUnitedScalar":
        ...
    @overload
    def __rmul__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        ...
    def __rmul__(self, other: Union[float, int, "RealUnitedScalar"]) -> "RealUnitedScalar":
        """Multiply by number (reverse operation)."""
        return self * other

    # Division # -----------------------------------------

    @overload
    def __truediv__(self, other: float|int) -> "RealUnitedScalar":
        ...
    @overload
    def __truediv__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        ...
    def __truediv__(self, other: Union["RealUnitedScalar", float, int]) -> "RealUnitedScalar":
        """Divide scalars or scalar by number."""
        if hasattr(other, 'canonical_value') and hasattr(other, 'dimension'):
            new_dimension: "Dimension" = self.dimension / other.dimension # type: ignore
            value: float = other.canonical_value # type: ignore
        else:
            new_dimension: "Dimension" = self.dimension
            value: float = other # type: ignore

        # Check for NaN
        if math.isnan(self.canonical_value) or math.isnan(value): # type: ignore
            from .....real_united_scalar import RealUnitedScalar
            return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
        
        # Division
        match math.isfinite(self.canonical_value), math.isfinite(value): # type: ignore
            case True, True: # (x / y)
                # Both finite - normal division
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(self.canonical_value / value, new_dimension, None) # type: ignore
            case True, False: # (x / inf)
                # Other is infinite
                if self.canonical_value == 0:
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
                else:
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(0, new_dimension, None) # type: ignore
            case False, True: # (inf / x)
                # Self is infinite
                if other == 0: # inf/0 = NaN
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
                else: # inf/x = inf
                    sign = 1 if value > 0 else -1
                    inf_sign = 1 if self.canonical_value > 0 else -1
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None) # type: ignore
            case False, False: # (inf / inf)
                # Both infinite
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
            case _:
                raise ValueError(f"Cannot divide {self} by {other} because: {math.isfinite(self.canonical_value)} and {math.isfinite(value)} are not finite") # type: ignore

    @overload
    def __rtruediv__(self, other: float|int) -> "RealUnitedScalar":
        ...
    @overload
    def __rtruediv__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        ...
    def __rtruediv__(self, other: Union["RealUnitedScalar", float, int]) -> "RealUnitedScalar":
        """Divide number by scalar."""
        if hasattr(other, 'canonical_value') and hasattr(other, 'dimension'):
            new_dimension: "Dimension" = other.dimension / self.dimension # type: ignore
            value: float = other.canonical_value # type: ignore
        else:
            new_dimension: "Dimension" = self.dimension ** -1 # type: ignore
            value: float = other # type: ignore

        # Check for NaN
        if math.isnan(self.canonical_value) or math.isnan(value): # type: ignore
            from .....real_united_scalar import RealUnitedScalar
            return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
        
        # Division
        match math.isfinite(value), math.isfinite(self.canonical_value): # type: ignore
            case True, True: # (y / x)
                # Both finite - normal division
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(value / self.canonical_value, new_dimension, None) # type: ignore
            case True, False: # (y / inf)
                # Other is infinite
                if self.canonical_value == 0: # 0/inf = NaN
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
                else: # y/inf = 0
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(0, new_dimension, None) # type: ignore
            case False, True: # (inf / x)
                # Self is infinite
                if other == 0: # inf/0 = NaN
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
                else: # inf/x = inf
                    sign = 1 if value > 0 else -1
                    inf_sign = 1 if self.canonical_value > 0 else -1
                    from .....real_united_scalar import RealUnitedScalar
                    return RealUnitedScalar(float('inf') * sign * inf_sign, new_dimension, None) # type: ignore
            case False, False: # (inf / inf)
                # Both infinite
                from .....real_united_scalar import RealUnitedScalar
                return RealUnitedScalar(math.nan, new_dimension, None) # type: ignore
            case _:
                raise ValueError(f"Cannot divide {other} by {self} because: {math.isfinite(value)} and {math.isfinite(self.canonical_value)} are not finite") # type: ignore

    # Power # --------------------------------------------

    def __pow__(self, exponent: float) -> "RealUnitedScalar":
        """Raise scalar to a power."""
        
        if math.isnan(exponent):
            from .....real_united_scalar import RealUnitedScalar
            return RealUnitedScalar.create_from_canonical_value(math.nan, self.dimension ** exponent, None)
        
        if exponent == 0:
            # Any number to the power of 0 is 1 (dimensionless)
            from .....real_united_scalar import RealUnitedScalar
            return RealUnitedScalar.create_from_canonical_value(1.0, Dimension.dimensionless_dimension(), None)
        
        if exponent == 1:
            # Return a copy of self
            from .....real_united_scalar import RealUnitedScalar
            return RealUnitedScalar.create_from_canonical_value(self.canonical_value, self.dimension, self._display_unit)
        
        # For other exponents, calculate the result
        new_value = self.canonical_value ** exponent
        new_dimension = self.dimension ** exponent
        
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(new_value, new_dimension, None)
    
    def __rpow__(self, other: float|int) -> "RealUnitedScalar":
        """
        Raise a number to the power of the scalar.

        This method computes `other ** self`, with full support for edge cases involving 0, infinities, and NaN.
        It assumes the scalar (`self`) is the exponent and `other` is the base.

        Special cases handled:

        - If `self` has dimension: raises ValueError unless dimensionless or log-level.
        - NaN in either base or exponent → returns NaN.
        - 0 ** 0 → NaN
        - x ** 0 → 1
        - 0 ** x → 0 if x > 0; inf if x < 0
        - Finite ** Finite → normal power
        - Finite ** ±inf:
            - abs(base) == 1 → 1
            - abs(base) > 1 → inf if exp > 0 else 0
            - abs(base) < 1 → 0 if exp > 0 else inf
        - ±inf ** Finite:
            - +inf ** +x → inf
            - +inf ** -x → 0
            - -inf ** +odd x → -inf
            - -inf ** +even x → inf
            - -inf ** -odd x → -0.0
            - -inf ** -even x → 0.0
            - -inf ** 0 → 1
        - ±inf ** ±inf:
            - +inf ** +inf → inf
            - +inf ** -inf → 0
            - -inf ** ±inf → NaN (undefined)

        Returns:
            RealUnitedScalar: The result of the power operation.
        """

        if self.dimension.is_dimensionless:
            dimension: "Dimension" = self.dimension.exp()
            if self._display_unit is not None:
                display_unit: Optional["Unit"] = self._display_unit.exp()
            else:
                display_unit: Optional["Unit"] = None
        else:
            #E.g. 5 ^(3 m/s) is not allowed
            raise ValueError(f"Cannot raise {self} to the power of {other} because it has dimension {self.dimension}")
        
        from .....real_united_scalar import RealUnitedScalar

        if math.isnan(other) or math.isnan(self.canonical_value):
            return RealUnitedScalar(math.nan)

        def is_odd_integer(x: float) -> bool:
            return x == int(x) and int(x) % 2 == 1

        base = other
        exp = self.canonical_value

        if base == 0 and exp == 0:
            return RealUnitedScalar(math.nan, dimension, display_unit) # type: ignore
        elif exp == 0:
            return RealUnitedScalar(1, dimension, display_unit) # type: ignore
        elif base == 0:
            return RealUnitedScalar(0 if exp > 0 else math.inf, dimension, display_unit) # type: ignore

        match math.isfinite(base), math.isfinite(exp):
            case True, True:
                return RealUnitedScalar(base ** exp, dimension, display_unit) # type: ignore
            case True, False:
                # x^±inf depends on abs(x)
                abs_base = abs(base)
                if abs_base == 1:
                    return RealUnitedScalar(1)
                elif abs_base > 1:
                    return RealUnitedScalar(math.inf if exp > 0 else 0, dimension, display_unit) # type: ignore
                else:
                    return RealUnitedScalar(0 if exp > 0 else math.inf, dimension, display_unit) # type: ignore
            case False, True:
                if base > 0:
                    return RealUnitedScalar(math.inf if exp > 0 else 0)
                else:
                    if exp > 0:
                        return RealUnitedScalar(-math.inf if is_odd_integer(exp) else math.inf, dimension, display_unit) # type: ignore
                    elif exp < 0:
                        return RealUnitedScalar(-0.0 if is_odd_integer(exp) else 0.0, dimension, display_unit) # type: ignore
                    else:
                        return RealUnitedScalar(1, dimension, display_unit) # type: ignore
            case False, False:
                if base > 0:
                    return RealUnitedScalar(math.inf if exp > 0 else 0, dimension, display_unit) # type: ignore
                else:
                    # -inf^±inf = nan (ambiguous or undefined)
                    return RealUnitedScalar(math.nan, dimension, display_unit) # type: ignore
            case _:
                raise ValueError(f"Cannot raise {base} to the power of {exp}: unexpected case")

    # Negation # ------------------------------------------

    def __neg__(self) -> "RealUnitedScalar":
        """Negate the scalar."""
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(-self.canonical_value, self.dimension, self._display_unit)
    
    # Absolute value # ------------------------------------

    def __abs__(self) -> "RealUnitedScalar":
        """Return the absolute value of the scalar."""
        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(abs(self.canonical_value), self.dimension, self._display_unit)
    
########################################################
# Other methods #########################################
########################################################

    # Logarithm # -----------------------------------------

    def log(self, base: float=math.e, log_level_unit_if_needed: str|Unit|UnitSymbol=UnitSymbol.NEPER) -> "RealUnitedScalar":
        """
        Return the logarithm of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the logarithm of the canonical value.

        Args:
            base: The base of the logarithm. Default is math.e.

        Returns:
            RealUnitedScalar: The logarithm of the scalar.

        Raises:
            ValueError: If the scalar has dimension LOG_LEVEL_DIMENSION.
        """

        dimension: "Dimension" = self.dimension.log()
        if self._display_unit is not None:
            display_unit: Optional["Unit"] = self._display_unit.log()
        else:
            display_unit: Optional["Unit"] = None

        if self.canonical_value < 0:
            canonical_value = math.nan
        elif self.canonical_value == 0:
            canonical_value = -math.inf
        else:
            if base == math.e:
                canonical_value = math.log(self.canonical_value)
            else:
                canonical_value = math.log(self.canonical_value, base)

        from .....real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(canonical_value, dimension, display_unit) # type: ignore
    
    def ln(self) -> "RealUnitedScalar":
        """
        Return the natural logarithm of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the natural logarithm of the canonical value.
        """
        return self.log(math.e)


    def log10(self) -> "RealUnitedScalar":
        """
        Return the logarithm of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the logarithm of the canonical value.
        """
        return self.log(10)
    
    def log2(self) -> "RealUnitedScalar":
        """
        Return the logarithm of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the logarithm of the canonical value.
        """
        return self.log(2)
    
    # Exponential # ----------------------------------------

    def exp(self) -> "RealUnitedScalar":
        """
        Return the exponential of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the exponential of the canonical value.
        """
        return self.__rpow__(math.e)
    
    def exp10(self) -> "RealUnitedScalar":
        """
        Return the exponential of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the exponential of the canonical value.
        """
        return self.__rpow__(10)

    
    def exp2(self) -> "RealUnitedScalar":
        """
        Return the exponential of the scalar. The dimension of the result is LOG_LEVEL_DIMENSION and the value corresponds to the exponential of the canonical value.
        """
        return self.__rpow__(2)
    
    # Exponential # ----------------------------------------
    