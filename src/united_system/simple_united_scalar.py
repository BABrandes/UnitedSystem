from dataclasses import dataclass
from .united_scalar import UnitedScalar
from .unit import Unit
from .named_canonical_quantities import NamedCanonicalQuantity
from .utils import str_to_float
import math







@dataclass(frozen=True, slots=True)
class SimpleUnitedScalar(UnitedScalar):
    """United value with automatic unit conversion support.
    
    This class stores values in canonical units internally and provides unit-aware operations.
    
    **Constructor Usage:**
    - The constructor `United_Value(canonical_value, display_unit)` takes the value in CANONICAL units
    - For user-friendly creation, use `United_Value.create(value, unit)` which takes display values
    - For parsing strings with units, use `United_Value.parse(string_with_unit)` 
    
    **Examples:**
    - `United_Value(273.15, TEMPERATURE.C)` - stores 273.15K, displays as Celsius
    - `United_Value.create(0, TEMPERATURE.C)` - creates 0°C (273.15K internally)
    - `United_Value.create(100, TEMPERATURE.C)` - creates 100°C (373.15K internally)
    - `United_Value.parse("100 °C")` - parses "100 °C" into 373.15K internally
    - `United_Value.parse("50 m")` - parses "50 m" into 50m internally (m is canonical)
    
    **API Methods:**
    - `create(value, unit)` - Create from display value and unit
    - `parse(string_with_unit)` - Parse from string like "100 °C" or "50 m"
    - Constructor - For internal use with canonical values
    
    Supports numeric values (float) with units, and non-numeric values (bool, str, datetime) 
    with NO_NUMBER unit.
    """
    canonical_value: float
    canonical_quantity: CanonicalQuantity
    display_unit: Unit|None

    def __post_init__(self):
        if self.display_unit is not None and not self.display_unit.compatible_to(self.canonical_quantity):
            raise ValueError(f"The display unit {self.display_unit} is not compatible with the canonical quantity {self.canonical_quantity}.")

    @classmethod
    def create(cls, value: float, unit: Unit) -> "UnitedScalar":
        """
        This is the safest way to create a numerical United_Value.
        """
        return cls(unit.to_canonical_value(value), unit.canonical_quantity, unit)
    
    @classmethod
    def create_from_canonical_value(cls, canonical_value: float, quantity: CanonicalQuantity, display_unit: Unit|None=None) -> "UnitedScalar":
        """
        This is the safest way to create a numerical United_Value from a canonical float.
        """
        return cls(canonical_value, quantity, display_unit)

    @classmethod
    def parse(cls, united_value_string: str) -> "UnitedScalar":
        split_string: list[str] = united_value_string.strip().split(" ")
        if len(split_string) == 1:
            # Just a number as string
            return cls(str_to_float(split_string[0]), NamedCanonicalQuantity.NUMBER.canonical_quantity, NamedCanonicalQuantity.NUMBER.canonical_unit)
        elif len(split_string) == 2:
            # Number and unit string
            if split_string[0] == "nan":
                value: float = math.nan
            else:
                value: float = str_to_float(split_string[0])
            unit_string: str = split_string[1]
            parsed_unit: Unit = SimpleUnit.parse(unit_string)
            canonical_value: float = parsed_unit.to_canonical_value(value)
            return cls(canonical_value, parsed_unit.canonical_quantity, parsed_unit)
        else:
            raise ValueError(f"Invalid united number: {united_value_string}")

    @property
    def canonical_float(self) -> float:
        if isinstance(self.canonical_value, float):
            return self.canonical_value
        else:
            raise ValueError(f"Cannot convert non-numeric value to float: {self.canonical_value}")
        
    @property
    def display_float(self) -> float:
        if self.display_unit is None:
            raise ValueError(f"Display unit is not set for United_Value: {self}")
        else:
            match self.canonical_value:
                case float() | int():
                    return self.display_unit.from_canonical_value(self.canonical_value)
                case complex():
                    raise ValueError(f"Cannot convert complex value to float: {self.canonical_value}")
                case _:
                    raise ValueError(f"Cannot convert non-numeric value to float: {self}")

    @property
    def display_value(self) -> float:
        if self.display_unit is None:
            raise ValueError(f"Display unit is not set for United_Value: {self}")
        else:
            return self.display_unit.from_canonical_value(self.canonical_value)

    def change_display_unit(self, new_display_unit: Unit) -> "SimpleUnitedScalar":
        """Convert to a different unit. Only works for numeric values."""
        if new_display_unit.compatible_to(self.canonical_quantity):
            return SimpleUnitedScalar(self.canonical_value, self.canonical_quantity, new_display_unit)
        else:
            raise ValueError(f"The suggested display unit {new_display_unit.nice_string} is not compatible with the canonical quantity {self.canonical_quantity}")
    
    def change_display_unit_to_canonical(self) -> "SimpleUnitedScalar":
        """Convert to canonical unit."""
        raise NotImplementedError("Not implemented")
    
    def as_float(self, unit: Unit) -> float:
        """Convert to float in specified unit. Only works for numeric values."""
        if not unit.compatible_to(self.canonical_quantity):
            raise ValueError(f"The suggested display unit {unit.nice_string} is not compatible with the canonical quantity {self.canonical_quantity}")
        match self.canonical_value:
            case float() | int():
                return unit.from_canonical_value(self.canonical_value)
            case complex():
                raise ValueError(f"Cannot convert complex value to float: {self.canonical_value}")
            case _:
                raise ValueError(f"Cannot convert non-numeric value to float: {self.canonical_value}")
            
    def as_complex(self, unit: Unit) -> complex:
        if not unit.compatible_to(self.canonical_quantity):
            raise ValueError(f"The suggested display unit {unit.nice_string} is not compatible with the canonical quantity {self.canonical_quantity}")
        return complex(unit.from_canonical_value(self.canonical_value))

    def __le__(self, other: "SimpleUnitedScalar") -> bool:
        """Evaluates to: self <= other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare units of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value <= other.canonical_value
    
    def __ge__(self, other: "SimpleUnitedScalar") -> bool:
        """Evaluates to: self >= other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare units of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value >= other.canonical_value # type: ignore
    
    def __lt__(self, other: "SimpleUnitedScalar") -> bool:
        """Evaluates to: self < other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare units of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value < other.canonical_value # type: ignore
    
    def __gt__(self, other: "SimpleUnitedScalar") -> bool:
        """Evaluates to: self > other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare units of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value > other.canonical_value # type: ignore
    
    def __eq__(self, other: "SimpleUnitedScalar") -> bool:
        """Evaluates to: self == other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare units of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value == other.canonical_value # type: ignore
    
    def __ne__(self, other: "SimpleUnitedScalar") -> bool:
        """Evaluates to: self != other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot compare units of different types: {self.display_unit} and {other.display_unit}")
        return self.canonical_value != other.canonical_value # type: ignore
    
    def __add__(self, other: "SimpleUnitedScalar") -> "SimpleUnitedScalar":
        """Evaluates to: self + other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot add units of different types: {self.display_unit} and {other.display_unit}")
        return SimpleUnitedScalar(self.canonical_value + other.canonical_value, self.canonical_quantity, self.display_unit)
    
    def __radd__(self, other: "SimpleUnitedScalar") -> "SimpleUnitedScalar":
        """Evaluates to: other + self."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot add units of different types: {self.display_unit} and {other.display_unit}")
        return SimpleUnitedScalar(other.canonical_value + self.canonical_value, other.canonical_quantity, other.display_unit)
    
    def __sub__(self, other: "SimpleUnitedScalar") -> "SimpleUnitedScalar":
        """Evaluates to: self - other."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot subtract units of different types: {self.display_unit} and {other.display_unit}")
        return SimpleUnitedScalar(self.canonical_value - other.canonical_value, self.canonical_quantity, self.display_unit)
    
    def __rsub__(self, other: "SimpleUnitedScalar") -> "SimpleUnitedScalar":
        """Evaluates to: other - self."""
        if not self.compatible_to(other):
            raise ValueError(f"Cannot subtract units of different types: {self.display_unit} and {other.display_unit}")
        return SimpleUnitedScalar(other.canonical_value - self.canonical_value, other.canonical_quantity, other.display_unit)
    
    def __truediv__(self, other: "SimpleUnitedScalar|float|int") -> "SimpleUnitedScalar":
        """Evaluates to: self / other."""
        if self.is_nan():
            return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
        if isinstance(other, SimpleUnitedScalar):
            if math.isnan(other.canonical_value):
                return SimpleUnitedScalar(math.nan, self.canonical_quantity - other.canonical_quantity, None)
            match math.isfinite(self.canonical_value), math.isfinite(other.canonical_value):
                case True, True:
                    match self.canonical_value, other.canonical_value:
                        case 0, 0:
                            # 0 divided by 0 is nan
                            return SimpleUnitedScalar(math.nan, self.canonical_quantity - other.canonical_quantity, None)
                        case 0, _:
                            # 0 divided by finite is 0
                            return SimpleUnitedScalar(float('inf') if self.canonical_value > 0 else float('-inf'), self.canonical_quantity - other.canonical_quantity, self.display_unit)
                        case _, 0:
                            # finite divided by 0 is inf or -inf
                            return SimpleUnitedScalar(float('inf') if self.canonical_value > 0 else float('-inf'), self.canonical_quantity - other.canonical_quantity, self.display_unit)
                        case _, _:
                            # finite divided by finite is finite
                            return SimpleUnitedScalar(self.canonical_value / other.canonical_value, self.canonical_quantity - other.canonical_quantity, None)
                case True, False:
                    # finite divided by inf is 0
                    return SimpleUnitedScalar(0, self.canonical_quantity - other.canonical_quantity, self.display_unit)
                case False, False:
                    # inf divided by inf is nan
                    return SimpleUnitedScalar(math.nan, self.canonical_quantity - other.canonical_quantity, None)
                case _, _:
                    raise ValueError(f"Cannot divide non-numeric values: {self.canonical_value} and {other.canonical_value}")
        elif isinstance(other, float|int):
            if math.isnan(other):
                return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
            match math.isfinite(self.canonical_value), math.isfinite(other) == 0:
                case True, True:
                    match self.canonical_value, other:
                        case 0, 0:
                            # 0 divided by 0 is nan
                            return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
                        case 0, _:
                            # 0 divided by finite is 0
                            return SimpleUnitedScalar(float('inf') if self.canonical_value > 0 else float('-inf'), self.canonical_quantity, self.display_unit)
                        case _, 0:
                            # finite divided by 0 is inf or -inf
                            return SimpleUnitedScalar(float('inf') if self.canonical_value > 0 else float('-inf'), self.canonical_quantity, self.display_unit)
                        case _, _:
                            # finite divided by finite is finite
                            return SimpleUnitedScalar(self.canonical_value / other, self.canonical_quantity, None)
                case True, False:
                    # finite divided by inf is 0
                    return SimpleUnitedScalar(0, self.canonical_quantity, self.display_unit)
                case False, False:
                    # inf divided by inf is nan
                    return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
                case _, _:
                    raise ValueError(f"Cannot divide non-numeric values: {self.canonical_value} and {other}")

    def __rtruediv__(self, other: float|int) -> "SimpleUnitedScalar":
        """Evaluates to: other / self."""
        if self.is_nan():
            return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
        if math.isnan(other):
            return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
        match math.isfinite(self.canonical_value), math.isfinite(other):
            case True, True:
                match self.canonical_value, other:
                    case 0, 0:
                        # 0 divided by 0 is nan
                        return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
                    case 0, _:
                        # 0 divided by finite is 0
                        return SimpleUnitedScalar(0, self.canonical_quantity, self.display_unit)
                    case _, 0:
                        # finite divided by 0 is inf or -inf
                        return SimpleUnitedScalar(float('inf') if self.canonical_value > 0 else float('-inf'), self.canonical_quantity, self.display_unit)
                    case _, _:
                        # finite divided by finite is finite
                        return SimpleUnitedScalar(other / self.canonical_value, self.canonical_quantity, None)
            case True, False:
                # finite divided by inf is 0
                return SimpleUnitedScalar(0, self.canonical_quantity, self.display_unit)
            case False, False:
                # inf divided by inf is nan
                return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
            case _, _:
                raise ValueError(f"Cannot divide non-numeric values: {other} and {self.canonical_value}")

    def __mul__(self, other: "SimpleUnitedScalar|float|int") -> "SimpleUnitedScalar":
        """Evaluates to: self * other."""
        if self.is_nan():
            return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
        if isinstance(other, SimpleUnitedScalar):
            if math.isnan(other.canonical_value):
                return SimpleUnitedScalar(math.nan, self.canonical_quantity - other.canonical_quantity, None)
            match math.isfinite(self.canonical_value), math.isfinite(other.canonical_value):
                case True, True:
                    match self.canonical_value, other.canonical_value:
                        case 0, 0:
                            # 0 * 0 is 0
                            return SimpleUnitedScalar(0, self.canonical_quantity - other.canonical_quantity, None)
                        case 0, _:
                            # 0 * finite is 0
                            return SimpleUnitedScalar(0, self.canonical_quantity - other.canonical_quantity, self.display_unit)
                        case _, 0:
                            # finite * 0 is 0
                            return SimpleUnitedScalar(0, self.canonical_quantity - other.canonical_quantity, self.display_unit)
                        case _, _:
                            # finite * finite is finite
                            return SimpleUnitedScalar(self.canonical_value * other.canonical_value, self.canonical_quantity - other.canonical_quantity, None)
                case True, False:
                    # finite * inf is inf
                    return SimpleUnitedScalar(float('inf') if self.canonical_value > 0 else float('-inf'), self.canonical_quantity - other.canonical_quantity, self.display_unit)
                case False, False:
                    # inf * inf is nan
                    return SimpleUnitedScalar(float('inf') if (self.canonical_value > 0 and other.canonical_value > 0) or (self.canonical_value < 0 and other.canonical_value < 0) else float('-inf'), self.canonical_quantity - other.canonical_quantity, None)
                case _, _:
                    raise ValueError(f"Cannot multiply non-numeric values: {self.canonical_value} and {other.canonical_value}")
        elif isinstance(other, float|int):
            if math.isnan(other):
                return SimpleUnitedScalar(math.nan, self.canonical_quantity, None)
            match math.isfinite(self.canonical_value), math.isfinite(other) == 0:
                case True, True:
                    match self.canonical_value, other:
                        case 0, 0:
                            # 0 * 0 is 0
                            return SimpleUnitedScalar(0, self.canonical_quantity, None)
                        case 0, _:
                            # 0 * finite is 0
                            return SimpleUnitedScalar(0, self.canonical_quantity, self.display_unit)
                        case _, 0:
                            # finite * 0 is 0
                            return SimpleUnitedScalar(0, self.canonical_quantity, self.display_unit)
                        case _, _:
                            # finite * finite is finite
                            return SimpleUnitedScalar(self.canonical_value * other, self.canonical_quantity, None)
                case True, False:
                    # finite * inf is inf
                    return SimpleUnitedScalar(0, self.canonical_quantity, self.display_unit)
                case False, False:
                    # inf * inf is inf
                    return SimpleUnitedScalar(float('inf') if (self.canonical_value > 0 and other > 0) or (self.canonical_value < 0 and other < 0) else float('-inf'), self.canonical_quantity, None)
                case _, _:
                    raise ValueError(f"Cannot multiply non-numeric values: {self.canonical_value} and {other}")
        
    
    def __rmul__(self, other: float|int) -> "SimpleUnitedScalar":
        """Evaluates to: other * self"""
        return self * other
        
    def is_positive(self) -> bool:
        """Evaluates to: self > 0"""
        return self.canonical_value > 0
    
    def is_negative(self) -> bool:
        """Evaluates to: self < 0"""
        return self.canonical_value < 0
    
    def is_zero(self) -> bool:
        """Evaluates to: self == 0"""
        return self.canonical_value == 0
    
    def is_non_zero(self) -> bool:
        """Evaluates to: self != 0"""
        return self.canonical_value != 0

    def is_nan(self) -> bool:
        """Evaluates to: self is NaN"""
        return math.isnan(self.canonical_value)
    
    def is_finite(self) -> bool:
        """Evaluates to: self is finite"""
        return math.isfinite(self.canonical_value)
    
    def is_positive_or_zero(self) -> bool:
        """Evaluates to: self >= 0"""
        return self.canonical_value >= 0
    
    def is_negative_or_zero(self) -> bool:
        """Evaluates to: self <= 0"""
        return self.canonical_value <= 0
    
    def is_strictly_positive(self) -> bool:
        """Evaluates to: self > 0"""
        return self.canonical_value > 0
    
    def is_strictly_negative(self) -> bool:
        """Evaluates to: self < 0"""
        return self.canonical_value < 0
    
    def format(self, unit: str|SimpleUnit|None=None, decimals: int = 0, do_not_print_unit: bool = False) -> str:

        if unit is None:
            if isinstance(self.canonical_quantity, SimpleCanonicalQuantity):
                _unit = SimpleUnit.suggest_unit_from_named_units(self.canonical_quantity, self.canonical_value)
            else:
                raise NotImplementedError("Not implemented to format with non-simple canonical quantity")
        elif isinstance(unit, str):
            _unit: SimpleUnit = SimpleUnit.parse(unit)
        else:
            _unit: SimpleUnit = unit
        
        if not _unit.compatible_to(self.canonical_quantity):
            raise ValueError(f"Cannot format with incompatible unit: {_unit} and {self.display_unit}")
        
        if self.canonical_value == 0:
            return f"0 {_unit.nice_string}"
        else:
            return f"{self.canonical_value:.{decimals}f} {_unit.nice_string}"
        
    def compatible_to(self, *args: "SimpleUnitedScalar") -> bool:
        """Evaluated if the quantities are the same"""
        if len(args) == 0:
            return True
        for arg in args:
            if arg.canonical_quantity != self.canonical_quantity:
                return False
        return True

    def __str__(self) -> str:
        return self.format(None, 3)
    
    def __repr__(self) -> str:
        return f"United_Value(cv={self.canonical_value}, du={self.display_unit.nice_string if self.display_unit is not None else None}, cq={self.canonical_quantity})"
    
    def __reduce__(self):
        """Custom serialization for multiprocessing."""
        return (SimpleUnitedScalar, (self.canonical_value, self.canonical_quantity, self.display_unit))
    
    @classmethod
    def sum(cls, values: list["SimpleUnitedScalar"]) -> "UnitedScalar":
        """Sum a list of United_Value objects. All values must be compatible and numeric."""
        if not values:
            raise ValueError("Cannot sum empty list of United_Value objects")
        if not all(uv.is_finite() for uv in values):
            raise ValueError("Cannot sum non-numeric United_Value objects")
        if not all(values[0].compatible_to(uv) for uv in values[1:]):
            raise ValueError("Cannot sum United_Value objects with incompatible units")
        total_canonical = sum(uv.canonical_value for uv in values) # type: ignore
        return cls(total_canonical, values[0].canonical_quantity, values[0].display_unit)

    @classmethod
    def mean(cls, values: list["SimpleUnitedScalar"]) -> "UnitedScalar":
        """Calculate mean of a list of United_Value objects. All values must be compatible and numeric."""
        if not values:
            raise ValueError("Cannot calculate mean of empty list of United_Value objects")
        if not all(uv.is_finite() for uv in values):
            raise ValueError("Cannot calculate mean non-numeric United_Value objects")
        if not all(values[0].compatible_to(uv) for uv in values[1:]):
            raise ValueError("Cannot calculate mean United_Value objects with incompatible units")
        total_canonical = sum(uv.canonical_value for uv in values) # type: ignore
        return cls(total_canonical / len(values), values[0].canonical_quantity, values[0].display_unit)

    def is_in_range(self, min_val: "SimpleUnitedScalar", max_val: "SimpleUnitedScalar") -> bool:
        """Check if this value is within the specified range (inclusive)."""
        if not self.compatible_to(min_val, max_val):
            raise ValueError("Cannot check range with incompatible units")
        if not self.is_finite():
            raise ValueError("Cannot check range for non-numeric values")
        return min_val <= self <= max_val
    
    def clamp(self, min_val: "SimpleUnitedScalar", max_val: "SimpleUnitedScalar") -> "SimpleUnitedScalar":
        """
        Clamp this value to the specified range.
        
        This method constrains the current value to be within the specified minimum and maximum bounds.
        If the value is less than the minimum, it returns the minimum value. If the value is greater
        than the maximum, it returns the maximum value. Otherwise, it returns the original value unchanged.
        
        Args:
            min_val (United_Value): The minimum allowed value. Must be compatible with this value's unit.
            max_val (United_Value): The maximum allowed value. Must be compatible with this value's unit.
            
        Returns:
            United_Value: A new United_Value object with the clamped value in the same unit as this value.
            
        Raises:
            United_Value_Exception: If the units are incompatible, if any values are non-numeric,
                                  or if the minimum value is greater than the maximum value.
                                  
        Examples:
            >>> uv = United_Value(15.0, LENGTH.m)
            >>> min_val = United_Value(10.0, LENGTH.m)
            >>> max_val = United_Value(20.0, LENGTH.m)
            >>> clamped = uv.clamp(min_val, max_val)
            >>> print(clamped)  # United_Value(15.0, LENGTH.m) - unchanged
            
            >>> uv = United_Value(5.0, LENGTH.m)
            >>> clamped = uv.clamp(min_val, max_val)
            >>> print(clamped)  # United_Value(10.0, LENGTH.m) - clamped to minimum
            
            >>> uv = United_Value(25.0, LENGTH.m)
            >>> clamped = uv.clamp(min_val, max_val)
            >>> print(clamped)  # United_Value(20.0, LENGTH.m) - clamped to maximum
            
        Note:
            - All values must be numeric and have compatible units
            - If min_val > max_val, they will be automatically swapped
            - The returned value maintains the same unit as the original value        
        """

        if not self.compatible_to(min_val, max_val):
            raise ValueError("Cannot clamp with incompatible units")
        if not self.is_finite():
            raise ValueError("Cannot clamp non-numeric values")
        if min_val.canonical_value > max_val.canonical_value:
            min_val, max_val = max_val, min_val
        if self < min_val:
            return min_val
        elif self > max_val:
            return max_val
        else:
            return self

    def to_json(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "canonical_value": self.canonical_value,
            "canonical_quantity": self.canonical_quantity.to_json(),
            "unit": self.display_unit.nice_string if self.display_unit is not None else None,
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "SimpleUnitedScalar":
        """Create from JSON-serializable dictionary."""
        return cls(data["canonical_value"], SimpleCanonicalQuantity.from_json(data["canonical_quantity"]), SimpleUnit.parse(data["unit"]) if data["unit"] is not None else None)
    
    def to_hdf5(self, group: h5py.Group) -> None:
        group.attrs["class"] = "UnitedSimpleValue"
        group.create_dataset("canonical_value", data=self.canonical_value)
        dt = h5py.string_dtype(encoding='utf-8')
        group.create_dataset("canonical_quantity_as_unit_string", data=self.canonical_quantity.canonical_unit().nice_string(), dtype=dt)
        group.create_dataset("display_unit", data=self.display_unit.nice_string if self.display_unit is not None else None, dtype=dt)

    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> "SimpleUnitedScalar":

        canonical_value: float = group.get("canonical_value", None) # type: ignore
        canonical_quantity_as_unit_string: str = group.get("canonical_quantity_as_unit_string", None) # type: ignore
        canonical_quantity: SimpleCanonicalQuantity = SimpleUnit.parse(canonical_quantity_as_unit_string).canonical_quantity
        display_unit_as_string: str|None = group.get("display_unit", None) if group.get("display_unit", None) is not None else None # type: ignore
        display_unit: Unit|None = SimpleUnit.parse(display_unit_as_string) if display_unit_as_string is not None else None

        return cls(
            canonical_value=canonical_value,
            canonical_quantity=canonical_quantity,
            display_unit=display_unit
        )
    
