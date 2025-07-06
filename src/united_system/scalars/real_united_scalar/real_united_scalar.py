"""RealUnitedScalar using mixins for better modularity."""

from dataclasses import dataclass
from ..united_scalar import UnitedScalar
from ...units.simple.simple_unit import SimpleUnit, SimpleDimension

# Import all mixins
from .mixins.core_mixin import RealUnitedScalarCore
from .mixins.arithmetic_mixin import ArithmeticMixin
from .mixins.comparison_mixin import ComparisonMixin
from .mixins.conversion_mixin import ConversionMixin
from .mixins.formatting_mixin import FormattingMixin
from .mixins.factory_mixin import FactoryMixin
from .mixins.utility_mixin import UtilityMixin
from .mixins.serialization_mixin import SerializationMixin

@dataclass(frozen=True, slots=True, eq=False)
class RealUnitedScalar(
    ArithmeticMixin,
    ComparisonMixin,
    ConversionMixin,
    FormattingMixin,
    FactoryMixin,
    UtilityMixin,
    SerializationMixin,
    RealUnitedScalarCore,
    UnitedScalar["RealUnitedScalar", SimpleUnit, SimpleDimension, float]
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
    
    def __post_init__(self):
        """Initialize and validate the scalar."""
        # Call the core mixin's post_init for validation
        RealUnitedScalarCore.__post_init__(self) 