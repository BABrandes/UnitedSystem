"""Factory methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....unit import Unit
    from .....dimension import Dimension
    from ....units.named_simple_dimensions import NamedSimpleDimension

class FactoryMixin:
    """Factory methods for RealUnitedScalar."""

    @classmethod
    def create_from_value_and_unit(cls, value: float, unit: Unit) -> RealUnitedScalar:
        """Create a RealUnitedScalar from a value and unit (converts to canonical)."""
        canonical_value = unit.to_canonical_value(value)
        return RealUnitedScalar(canonical_value, unit.dimension, unit)

    @classmethod
    def create_from_canonical_value(cls, canonical_value: float, dimension: Dimension, unit: Union[Unit, None] = None) -> RealUnitedScalar:
        """Create a RealUnitedScalar from a canonical value and dimension."""
        if unit is not None and not unit.compatible_to(dimension):
            raise ValueError(f"The display unit {unit} is not compatible with the canonical dimension {dimension}.")
        return RealUnitedScalar(canonical_value, dimension, unit)

    @classmethod
    def parse_string(cls, string: str) -> RealUnitedScalar:
        """Parse a string like '50 m' or '100 °C' into a RealUnitedScalar."""
        from ....general import str_to_float
        
        split_string = string.rsplit(" ", 1)
        if len(split_string) == 1:
            # No unit specified - assume dimensionless
            value = str_to_float(split_string[0])
            return RealUnitedScalar(value, NamedSimpleDimension.NUMBER.dimension, NamedSimpleDimension.NUMBER.canonical_unit)
        elif len(split_string) == 2:
            # Value and unit specified
            value_str, unit_str = split_string
            parsed_unit = Unit.parse_string(unit_str)
            canonical_value = parsed_unit.to_canonical_value(str_to_float(value_str))
            return RealUnitedScalar(canonical_value, parsed_unit.dimension, parsed_unit)
        else:
            raise ValueError(f"String {string} could not be parsed")

    @classmethod
    def create_dimensionless(cls, value: float) -> RealUnitedScalar:
        """Create a dimensionless scalar."""
        from ....units.named_simple_dimensions import NamedSimpleDimension
        return RealUnitedScalar(value, NamedSimpleDimension.NUMBER.dimension, NamedSimpleDimension.NUMBER.canonical_unit)