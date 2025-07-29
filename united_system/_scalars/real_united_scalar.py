"""RealUnitedScalar using mixins for better modularity."""

from dataclasses import dataclass, field
from typing import Optional, overload
from .._units_and_dimension.named_quantity import NamedQuantity
from .._units_and_dimension.dimension import Dimension
from .._units_and_dimension.unit import Unit
from .._utils.general import str_to_float

# Import all mixins
from .._scalars.mixins.real_united_scalar.core_mixin import RealUnitedScalarCore
from .._scalars.mixins.real_united_scalar.arithmetic_mixin import ArithmeticMixin
from .._scalars.mixins.real_united_scalar.comparison_mixin import ComparisonMixin
from .._scalars.mixins.real_united_scalar.conversion_mixin import ConversionMixin
from .._scalars.mixins.real_united_scalar.formatting_mixin import FormattingMixin
from .._scalars.mixins.real_united_scalar.factory_mixin import FactoryMixin
from .._scalars.mixins.real_united_scalar.utility_mixin import UtilityMixin
from .._scalars.mixins.real_united_scalar.serialization_mixin import SerializationMixin

@dataclass(frozen=True, slots=True, eq=False, init=False)
class RealUnitedScalar(
    ArithmeticMixin,
    ComparisonMixin,
    ConversionMixin,
    FormattingMixin,
    FactoryMixin,
    UtilityMixin,
    SerializationMixin,
    RealUnitedScalarCore,
):
    """
    A real-valued scalar with units using a mixin-based architecture.
    
    This class represents a real number with associated units and provides
    comprehensive unit-aware operations through multiple mixins:
    
    - RealUnitedScalarCore: Basic functionality and validation
    - ArithmeticMixin: Arithmetic operations (+, -, *, /, **)
    - ComparisonMixin: Comparison operations (==, !=, <, >, <=, >=)
    - ConversionMixin: Unit conversion and value extraction
    - FormattingMixin: String formatting and representation
    - FactoryMixin: Factory methods for creating instances
    - UtilityMixin: Utility methods (is_positive, sum, mean, clamp, etc.)
    - SerializationMixin: JSON and HDF5 serialization
    
    **Constructor Usage:**
    - The constructor takes values in CANONICAL units
    - For user-friendly creation, use factory methods like `create_from_value_and_unit()`
    - For parsing strings with units, use `parse_string()`
    
    **Examples:**
    ```python
    # Create from value and unit
    length = RealUnitedScalar.create_from_value_and_unit(5.0, SimpleUnit.parse_string("m"))
    
    # Parse from string
    voltage = RealUnitedScalar.parse_string("12 V")
    
    # Arithmetic operations
    total_length = length + RealUnitedScalar.create_from_value_and_unit(3.0, SimpleUnit.parse_string("m"))
    
    # Unit conversions
    length_in_km = length.in_unit(SimpleUnit.parse_string("km"))
    
    # Formatting
    print(f"Length: {length.format('km', decimals=3)}")
    ```
    """
    
    # Explicit dataclass fields
    dimension: Dimension
    _display_unit: Optional[Unit] = field(default=None, repr=False, compare=False, hash=False)

    @overload
    def __init__(self, value: str) -> None:
        """
        Initialize the scalar from a string which must contain a value and may contain a unit.
        "1 cm" -> {0.01, L, cm}
        "4" -> {4.0, L}
        "1 km/s" -> {1000, L/T, km/s}
        "1 mV/s" -> {0.001, V/T, mV/s}
        "1 m/s^2" -> {1.0, L/T^2, m/s^2}
        """
        ...
    @overload
    def __init__(self, value: float|int) -> None:
        """
        Initialize the scalar from a float.
        -3.0 -> {-3.0, , ""}
        4.0 -> {4.0, , ""}
        """
        ...
    @overload
    def __init__(self, value: float|int, unit_or_dimension: Unit|str) -> None:
        """
        Initialize the scalar from a float and a unit.
        -3.0, m -> {-3.0, L, m}
        4.0, m/s -> {4.0, L/T, m/s}
        """
        ...    
    @overload
    def __init__(self, value: float|int, unit_or_dimension: Dimension|NamedQuantity) -> None:
        """
        Initialize the scalar from a float (as canonical value) and a dimension.
        -3.0, L -> {-3.0, m}
        4.0, L/T -> {4.0, m/s}
        """
        ...
    @overload
    def __init__(self, value: float|int, unit_or_dimension: Dimension, display_unit: Unit|None = None) -> None:
        """
        Initialize the scalar from a float (as canonical value) and a dimension.
        -3.0, L, km -> {-0.003, L, km}
        4.0, L/T, km/h -> {4.0, m/s}
        """
        ...
    def __init__(self, value: str|float|int, unit_or_dimension: Unit|str|Dimension|NamedQuantity|None = None, display_unit: Unit|None = None) -> None:
        """Initialize the scalar."""

        if display_unit is not None:
            # Usually for internal use.
            if isinstance(value, float|int) and isinstance(unit_or_dimension, Dimension):
                canonical_value = value
                dimension = unit_or_dimension
                display_unit = display_unit
            else:
                raise ValueError("The constructor is not designed to be used with a display unit in such a way.")

        else:
            if unit_or_dimension is None:
                if isinstance(value, str):
                    canonical_value, display_unit = self._parse_string(value)
                    dimension: Dimension = display_unit.dimension
                else:
                    canonical_value = float(value)
                    display_unit = None
                    dimension = Dimension.dimensionless_dimension()
            else:

                # Parse the first argument.
                if isinstance(value, str):
                    value = value.strip()
                    value_and_unit = value.split(" ")
                    if len(value_and_unit) == 1:
                        _value: float = str_to_float(value_and_unit[0])
                    elif len(value_and_unit) == 2:
                        _value: float = str_to_float(value_and_unit[0])
                        display_unit = Unit(value[1])
                    else:
                        raise ValueError(f"Invalid value: {value}")
                else:
                    _value: float = float(value)

                # If a display unit was found in the first argument, unit_or_dimension must be None.
                if display_unit is not None and unit_or_dimension is not None: # type: ignore
                    raise ValueError("The constructor is not designed to be used with two units or a unit and a dimension.")

                # Parse the second argument.
                if isinstance(unit_or_dimension, str):
                    display_unit = Unit(unit_or_dimension)
                    canonical_value = display_unit.to_canonical_value(_value)
                    dimension = display_unit.dimension
                elif isinstance(unit_or_dimension, Unit):
                    display_unit = unit_or_dimension
                    canonical_value = display_unit.to_canonical_value(_value)
                    dimension = unit_or_dimension.dimension
                elif isinstance(unit_or_dimension, Dimension):
                    dimension = unit_or_dimension
                    canonical_value = _value
                    display_unit = None
                elif isinstance(unit_or_dimension, NamedQuantity): # type: ignore
                    dimension = Dimension(unit_or_dimension)
                    canonical_value = _value
                    display_unit = None
                else:
                    raise ValueError(f"Invalid unit or dimension: {unit_or_dimension}")

        object.__setattr__(self, "canonical_value", canonical_value)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", display_unit)        
        # Validate the scalar after initialization.
        if self._display_unit is not None and not self._display_unit.compatible_to(self.dimension):
            raise ValueError(f"The display unit {self._display_unit} is not compatible with the canonical dimension {self.dimension}.")

    @property
    def unit(self) -> Unit:
        if self._display_unit is None:
            object.__setattr__(self, "_display_unit", self.dimension.canonical_unit)
        if self._display_unit is None:
            raise ValueError("Display unit is None")
        return self._display_unit