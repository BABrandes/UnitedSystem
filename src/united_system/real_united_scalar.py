"""RealUnitedScalar using mixins for better modularity."""

from dataclasses import dataclass, field
from typing import Optional
from .utils.scalars.united_scalar import UnitedScalar
from .dimension import Dimension
from .unit import Unit

# Import all mixins
from .utils.scalars.mixins.real_united_scalar.core_mixin import RealUnitedScalarCore
from .utils.scalars.mixins.real_united_scalar.arithmetic_mixin import ArithmeticMixin
from .utils.scalars.mixins.real_united_scalar.comparison_mixin import ComparisonMixin
from .utils.scalars.mixins.real_united_scalar.conversion_mixin import ConversionMixin
from .utils.scalars.mixins.real_united_scalar.formatting_mixin import FormattingMixin
from .utils.scalars.mixins.real_united_scalar.factory_mixin import FactoryMixin
from .utils.scalars.mixins.real_united_scalar.utility_mixin import UtilityMixin
from .utils.scalars.mixins.real_united_scalar.serialization_mixin import SerializationMixin

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
    UnitedScalar["RealUnitedScalar", Unit, Dimension, float]
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
    canonical_value: float
    dimension: Dimension
    _display_unit: Optional[Unit] = field(default=None, repr=False, compare=False, hash=False)
    
    def __init__(self, canonical_value: float, dimension: Dimension, display_unit: Optional[Unit] = None):
        """Initialize the scalar."""
        object.__setattr__(self, "canonical_value", canonical_value)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", display_unit)        
        # Validate the scalar after initialization.
        if self._display_unit is not None and not self._display_unit.compatible_to(self.dimension):
            raise ValueError(f"The display unit {self._display_unit} is not compatible with the canonical dimension {self.dimension}.")

    @property
    def display_unit(self) -> Unit:
        if self._display_unit is None:
            object.__setattr__(self, "_display_unit", self.dimension.canonical_unit)
        if self._display_unit is None:
            raise ValueError("Display unit is None")
        return self._display_unit
    
    @property
    def active_unit(self) -> Unit:
        if self._display_unit is None:
            return self.dimension.canonical_unit
        return self._display_unit
    
    @property
    def active_float(self) -> float:
        return self.canonical_value * self.active_unit.factor + self.active_unit.offset