"""Factory methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Union, Optional
from ...._units_and_dimension.unit import Unit
from ...._units_and_dimension.dimension import Dimension
from .protocol import RealUnitedScalarProtocol
from ...._units_and_dimension.named_quantity import NamedQuantity

if TYPE_CHECKING:
    from ...._scalars.real_united_scalar import RealUnitedScalar

class FactoryMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Factory methods for RealUnitedScalar."""

    @classmethod
    def create_from_value_and_unit(cls, value: float, unit: "Unit") -> "RealUnitedScalar":
        """
        Create a RealUnitedScalar from a value and unit.
        
        Args:
            value: The value in the specified unit
            unit: The unit of the value
            
        Returns:
            A new RealUnitedScalar with the specified value and unit
            
        Example:
            >>> kg_unit = Unit("kg")
            >>> scalar = RealUnitedScalar.create_from_value_and_unit(5.0, kg_unit)
            >>> scalar.canonical_value
            5.0
            >>> scalar._display_unit
            Unit("kg")
        """
        canonical_value: float = unit.to_canonical_value(value)
        return cls.create_from_canonical_value(canonical_value, unit.dimension, unit)

    @classmethod
    def create_from_canonical_value(cls, canonical_value: float, dimension_or_unit: Union["Dimension", "Unit"], display_unit: Optional["Unit"] = None) -> "RealUnitedScalar":
        """
        Create a RealUnitedScalar from a canonical value and dimension/unit.
        
        Args:
            canonical_value: The value in canonical units
            dimension_or_unit: The dimension or unit (if unit, its dimension is used)
            display_unit: Optional display unit (if None, no display unit is set)
            
        Returns:
            A new RealUnitedScalar with the specified canonical value
            
        Example:
            >>> mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
            >>> scalar = RealUnitedScalar.create_from_canonical_value(1.0, mass_dim)
            >>> scalar.canonical_value
            1.0
            >>> scalar._display_unit is None
            True
            
            >>> kg_unit = Unit("kg")
            >>> scalar_with_display = RealUnitedScalar.create_from_canonical_value(1.0, mass_dim, kg_unit)
            >>> scalar_with_display._display_unit
            Unit("kg")
        """
        
        if isinstance(dimension_or_unit, Unit):
            dimension: Dimension = dimension_or_unit.dimension
        else:
            dimension: Dimension = dimension_or_unit
        return cls(canonical_value, dimension, display_unit) # type: ignore
    
    @classmethod
    def _parse_string(cls, string: str) -> tuple[float, "Unit"]:
        """
        Parse a string like '50 m' or '100 \u00b0C' into a float and a unit.

        Give the canonical value and the display unit.
        """
        from ...._utils.general import str_to_float

        split_string = string.rsplit(" ", 1)
        if len(split_string) == 1:
            # No unit specified - assume dimensionless
            value: float = str_to_float(split_string[0])
            return value, Unit.dimensionless_unit()
        elif len(split_string) == 2:
            # Value and unit specified
            value_str, unit_str = split_string
            parsed_unit: "Unit" = Unit(unit_str)
            canonical_value = parsed_unit.to_canonical_value(str_to_float(value_str))
            return canonical_value, parsed_unit
        else:
            raise ValueError(f"String {string} could not be parsed")
        
    @classmethod
    def parse_string(cls, string: str) -> "RealUnitedScalar":
        """
        Parse a string like '50 m' or '100 °C' into a RealUnitedScalar.
        
        Args:
            string: String to parse in format "value unit"
            
        Returns:
            A new RealUnitedScalar parsed from the string
            
        Raises:
            ValueError: If the string format is invalid
            
        Example:
            >>> scalar = RealUnitedScalar.parse_string("5.0 kg")
            >>> scalar.canonical_value
            5.0
            >>> scalar._display_unit
            Unit("kg")
            
            >>> temp_scalar = RealUnitedScalar.parse_string("100 °C")
            >>> temp_scalar.canonical_value
            373.15  # Temperature in Kelvin
        """
        canonical_value, unit = cls._parse_string(string)
        return cls.create_from_canonical_value(canonical_value, unit.dimension, unit)

    @classmethod
    def create_dimensionless(cls, value: float) -> "RealUnitedScalar":
        """Create a dimensionless scalar."""
        from ...._scalars.real_united_scalar import RealUnitedScalar
        return RealUnitedScalar.create_from_canonical_value(value, Dimension.dimensionless_dimension(), None)

    #########################################################
    # Factory methods for special values (cached)
    #########################################################

    _CACHE_SPECIAL_VALUES: dict[float, dict[Unit|Dimension|NamedQuantity|None, "RealUnitedScalar"]] = {}

    @classmethod
    def positive_infinity(cls, unit_or_dimension: Unit|Dimension|NamedQuantity|None = None) -> "RealUnitedScalar":
        """
        Returns a scalar with positive infinity from the cache.

        Args:
            unit_or_dimension: The unit or dimension of the scalar

        Returns:
            A new RealUnitedScalar with positive infinity
        """
        if float("inf") not in cls._CACHE_SPECIAL_VALUES:
            cls._CACHE_SPECIAL_VALUES[float("inf")] = {}
        if unit_or_dimension not in cls._CACHE_SPECIAL_VALUES[float("inf")]:
            cls._CACHE_SPECIAL_VALUES[float("inf")][unit_or_dimension] = RealUnitedScalar(float("inf"), unit_or_dimension)
        return cls._CACHE_SPECIAL_VALUES[float("inf")][unit_or_dimension]
    
    @classmethod
    def negative_infinity(cls, unit_or_dimension: Unit|Dimension|NamedQuantity|None = None) -> "RealUnitedScalar":
        """
        Returns a scalar with negative infinity from the cache.

        Args:
            unit_or_dimension: The unit or dimension of the scalar

        Returns:
            A new RealUnitedScalar with negative infinity
        """
        if float("-inf") not in cls._CACHE_SPECIAL_VALUES:
            cls._CACHE_SPECIAL_VALUES[float("-inf")] = {}
        if unit_or_dimension not in cls._CACHE_SPECIAL_VALUES[float("-inf")]:
            cls._CACHE_SPECIAL_VALUES[float("-inf")][unit_or_dimension] = RealUnitedScalar(float("-inf"), unit_or_dimension)
        return cls._CACHE_SPECIAL_VALUES[float("-inf")][unit_or_dimension]
    
    @classmethod
    def nan(cls, unit_or_dimension: Unit|Dimension|NamedQuantity|None = None) -> "RealUnitedScalar":
        """
        Returns a scalar with NaN from the cache.

        Args:
            unit_or_dimension: The unit or dimension of the scalar

        Returns:
            A new RealUnitedScalar with NaN
        """
        if float("nan") not in cls._CACHE_SPECIAL_VALUES:
            cls._CACHE_SPECIAL_VALUES[float("nan")] = {}
        if unit_or_dimension not in cls._CACHE_SPECIAL_VALUES[float("nan")]:
            cls._CACHE_SPECIAL_VALUES[float("nan")][unit_or_dimension] = RealUnitedScalar(float("nan"), unit_or_dimension)
        return cls._CACHE_SPECIAL_VALUES[float("nan")][unit_or_dimension]
        
    @classmethod
    def zero(cls, unit_or_dimension: Unit|Dimension|NamedQuantity|None = None) -> "RealUnitedScalar":
        """
        Returns a scalar with zero from the cache.

        Args:
            unit_or_dimension: The unit or dimension of the scalar

        Returns:
            A new RealUnitedScalar with zero
        """
        if 0.0 not in cls._CACHE_SPECIAL_VALUES:
            cls._CACHE_SPECIAL_VALUES[0.0] = {}
        if unit_or_dimension not in cls._CACHE_SPECIAL_VALUES[0.0]:
            cls._CACHE_SPECIAL_VALUES[0.0][unit_or_dimension] = RealUnitedScalar(0.0, unit_or_dimension)
        return cls._CACHE_SPECIAL_VALUES[0.0][unit_or_dimension]
        
    @classmethod
    def one(cls, unit_or_dimension: Unit|Dimension|NamedQuantity|None = None) -> "RealUnitedScalar":
        """
        Returns a scalar with one from the cache.

        Args:
            unit_or_dimension: The unit or dimension of the scalar

        Returns:
            A new RealUnitedScalar with one
        """
        if 1.0 not in cls._CACHE_SPECIAL_VALUES:
            cls._CACHE_SPECIAL_VALUES[1.0] = {}
        if unit_or_dimension not in cls._CACHE_SPECIAL_VALUES[1.0]:
            cls._CACHE_SPECIAL_VALUES[1.0][unit_or_dimension] = RealUnitedScalar(1.0, unit_or_dimension)
        return cls._CACHE_SPECIAL_VALUES[1.0][unit_or_dimension]