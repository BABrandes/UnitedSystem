"""Factory methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from ..real_united_scalar import RealUnitedScalar
    from ...united_scalar import UnitedScalar
    from ....units.simple.simple_unit import SimpleUnit
    from ....units.simple.simple_dimension import SimpleDimension


class FactoryMixin:
    """Factory methods for RealUnitedScalar."""

    @classmethod
    def create_from_value_and_unit(cls, value: float, unit: "SimpleUnit") -> "RealUnitedScalar":
        """Create a RealUnitedScalar from a value and unit (converts to canonical)."""
        canonical_value = unit.to_canonical_value(value)
        return cls(canonical_value, unit.dimension, unit)

    @classmethod
    def create_from_canonical_value(cls, canonical_value: float, dimension: "SimpleDimension", display_unit: Union["SimpleUnit", None] = None) -> "UnitedScalar":
        """Create a RealUnitedScalar from a canonical value and dimension."""
        if display_unit is not None and not display_unit.compatible_to(dimension):
            raise ValueError(f"The display unit {display_unit} is not compatible with the canonical dimension {dimension}.")
        return cls(canonical_value, dimension, display_unit)

    @classmethod
    def parse_string(cls, string: str) -> "RealUnitedScalar":
        """Parse a string like '50 m' or '100 °C' into a RealUnitedScalar."""
        from ....utils import str_to_float
        from ....units.simple.simple_unit import SimpleUnit
        from ....units.named_simple_dimensions import NamedSimpleDimension
        
        split_string = string.rsplit(" ", 1)
        if len(split_string) == 1:
            # No unit specified - assume dimensionless
            value = str_to_float(split_string[0])
            return cls(value, NamedSimpleDimension.NUMBER.simple_unit_dimension, NamedSimpleDimension.NUMBER.canonical_unit)
        elif len(split_string) == 2:
            # Value and unit specified
            value_str, unit_str = split_string
            parsed_unit = SimpleUnit.parse_string(unit_str)
            canonical_value = parsed_unit.to_canonical_value(str_to_float(value_str))
            return cls(canonical_value, parsed_unit.dimension, parsed_unit)
        else:
            raise ValueError(f"String {string} could not be parsed")

    @classmethod
    def create_dimensionless(cls, value: float) -> "RealUnitedScalar":
        """Create a dimensionless scalar."""
        from ....units.named_simple_dimensions import NamedSimpleDimension
        from ..real_united_scalar import RealUnitedScalar
        return cls(value, NamedSimpleDimension.NUMBER.simple_unit_dimension, NamedSimpleDimension.NUMBER.canonical_unit)