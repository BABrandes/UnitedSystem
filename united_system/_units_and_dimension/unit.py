"""
Unit Module for United System

This module provides the Unit class for handling physical units in scientific calculations.
Units represent the specific measurement systems (meters, seconds, kilograms, etc.) and their relationships.

Key Features:
- Support for SI base units, derived units, and non-SI units
- Subscript support for distinguishing different contexts (e.g., m_elec vs m_geo)
- Logarithmic units using DEC() notation and angle units using ANG() notation
- Arithmetic operations (multiplication, division, powers, logarithms)
- Serialization to JSON and HDF5 formats
- Immutable design for thread safety
- Unit conversion and canonical value handling

Examples:
    # Create simple units
    meter = Unit("m")           # Meter unit
    second = Unit("s")          # Second unit
    kilogram = Unit("kg")       # Kilogram unit
    
    # Create complex units
    velocity = Unit("m/s")      # Meters per second
    force = Unit("kg*m/s^2")    # Kilogram * meter / second^2 (newton)
    energy = Unit("kg*m^2/s^2") # Kilogram * meter^2 / second^2 (joule)
    
    # Use subscripts to distinguish contexts
    elec_length = Unit("m_elec")  # Electrical length in meters
    geo_length = Unit("m_geo")    # Geometric length in meters
    
    # Logarithmic units
    log_force = Unit("DEC(N)")    # Logarithm of force in newtons
    
    # Arithmetic operations
    area = meter * meter        # m * m = m^2
    density = kilogram / (meter**3)  # kg / m^3
    log_area = area.log()       # DEC(m^2)
    
    # Unit conversion
    canonical_value = 5.0 * meter.to_canonical_value(2.5)  # Convert 2.5 m to canonical
    display_value = meter.from_canonical_value(5.0)        # Convert canonical to 2.5 m
    
    # Serialization
    json_str = force.to_json()  # "kg*m/s^2"
    reparsed = Unit.from_json(json_str)
    
    # Check if dimensionless
    dimensionless = Unit("")    # Empty string creates dimensionless unit
    assert dimensionless.is_dimensionless
"""

from typing import TYPE_CHECKING, overload, Union, Optional, Tuple, List, Sequence, Any, cast, Literal
from dataclasses import dataclass, field
from types import MappingProxyType
from h5py import Group
import numpy as np
import pandas as pd

from .named_quantity import NamedQuantity
from .unit_element import UnitElement
from .dimension import Dimension
from .unit_symbol import UnitSymbol
from .unit_prefix import UnitPrefix
from .utils import seperate_string
from .has_unit_protocol import HasUnit

if TYPE_CHECKING:
    from .._scalars.real_united_scalar import RealUnitedScalar
    from .._arrays.real_united_array import RealUnitedArray
    from .._scalars.complex_united_scalar import ComplexUnitedScalar
    from .._arrays.complex_united_array import ComplexUnitedArray

# Runtime imports for type checking
from .._utils.value_type import VALUE_TYPE
from .._utils.scalar_type import SCALAR_TYPE

EPSILON: float = 1e-12

# Cache for parsed units (outside the class to avoid slots conflict)
_UNIT_CACHE: dict[str, "Unit"] = {}

def clear_unit_cache():
    """Clear the unit cache to force re-parsing of unit strings."""
    global _UNIT_CACHE
    _UNIT_CACHE.clear()

@dataclass(frozen=True, slots=True)
class Unit:
    """
    A class representing physical units for scientific calculations.
    
    The Unit class handles the mathematical representation of physical units,
    supporting SI units, subscripts, logarithmic units, and arithmetic operations.
    
    Attributes:
        _unit_elements: Dictionary mapping subscripts to tuples of unit elements
        _log_units: Dictionary mapping log units to their exponents
    
    Examples:
        # Basic usage
        meter = Unit("m")
        second = Unit("s")
        kilogram = Unit("kg")
        
        # Complex units
        velocity = Unit("m/s")
        acceleration = Unit("m/s^2")
        force = Unit("kg*m/s^2")
        
        # With subscripts
        elec_length = Unit("m_elec")
        geo_length = Unit("m_geo")
        
        # Logarithmic units
        log_force = Unit("DEC(N)")
        
        # Arithmetic operations
        area = meter * meter
        volume = area * meter
        density = kilogram / volume
        
        # Powers and roots
        area_squared = area ** 2
        length_cubed = meter ** 3
        
        # Logarithms (applied to canonical values)
        log_area = area.log()  # dec(L^2) - decades of area
        log_current = Unit("mA").log()  # dec(A) - decades of current (not decades of milliampere)
        
        # Unit conversion
        canonical = meter.to_canonical_value(2.5)  # 2.5 m in canonical units
        display = meter.from_canonical_value(2.5)  # 2.5 canonical units in m
        
        # Serialization
        json_str = force.to_json()
        reparsed = Unit.from_json(json_str)
        
        # Validation
        assert not force.is_dimensionless
        assert Unit("").is_dimensionless
    """

    _unit_elements: MappingProxyType[str, Tuple[UnitElement, ...]] = field(default_factory=lambda: MappingProxyType({}))
    _log_units: List[Tuple[UnitElement, Dimension]] = field(default_factory=lambda: [])
    _dimension: Dimension = field(init=False)

########################################################
# Initialization
########################################################

    @overload
    def __init__(self) -> None:
        ...

    @overload
    def __init__(self, value: Optional[str]=None) -> None:
        ...

    @overload
    def __init__(self, value: Optional[NamedQuantity]=None, subscript: Optional[str]=None) -> None:
        ...

    def __init__(
            self,
            value: Optional[Union[str, NamedQuantity]]=None, 
            subscript: Optional[str]=None
    ) -> None:
        """
        Initialize a Unit object.
        
        Args:
            value: The unit specification. Can be:
                - A string representing a unit (e.g., "m", "kg*m/s^2", "DEC(N)")
                - A dictionary mapping subscripts to tuples of UnitElement objects
                - None for dimensionless unit
            subscript: Optional subscript (currently unused)
        
        Examples:
            Unit("m")                    # Simple unit
            Unit("kg*m/s^2")             # Complex unit
            Unit("DEC(N)")               # Logarithmic unit
            Unit("m_elec")               # Unit with subscript
        """
        if isinstance(value, str):
            return
        elif isinstance(value, NamedQuantity):
            return
        elif value is None:
            # Create dimensionless unit
            object.__setattr__(self, "_unit_elements", MappingProxyType({})) # type: ignore
            object.__setattr__(self, "_log_units", [])
        else:
            raise ValueError(f"Invalid unit value: {value}")

    def __new__(
        cls,
        value: Optional[Union[str, NamedQuantity]]=None,
        subscript: Optional[str]=None,
    ) -> "Unit":
        """
        Create a new Unit instance with caching for string inputs.
        
        Args:
            value: The unit specification
            subscript: Optional subscript when creating from NamedQuantity
        
        Returns:
            A new Unit instance
        """
        if value is None:
            # Create empty unit
            instance: Unit = object.__new__(cls)
            empty_unit_elements: dict[str, Tuple[UnitElement, ...]] = {}
            object.__setattr__(instance, "_unit_elements", MappingProxyType(empty_unit_elements))
            object.__setattr__(instance, "_log_units", [])
            return instance
        elif isinstance(value, str):
            # Check cache first
            if value in _UNIT_CACHE:
                return _UNIT_CACHE[value]
            
            # Create new instance
            instance: Unit = object.__new__(cls)

            # Parse string representation
            unit_elements, log_units = cls._parse_string(value)
            object.__setattr__(instance, "_unit_elements", MappingProxyType(unit_elements))
            object.__setattr__(instance, "_log_units", log_units)
            
            # Cache the result
            _UNIT_CACHE[value] = instance
            return instance
        
        elif isinstance(value, NamedQuantity): # type: ignore
            # Create a new instance
            if subscript is None:
                instance: Unit = value.dimension.canonical_unit
            else:
                instance: Unit = Dimension(value, subscript).canonical_unit
            return instance
        
        else:
            raise ValueError(f"Invalid unit value: {value}")

    @classmethod
    def _construct(
        cls,
        unit_elements: MappingProxyType[str, Tuple[UnitElement, ...]],
        log_units: List[Tuple[UnitElement, "Dimension"]] = [],
    ) -> "Unit":
        """
        Construct a Unit from its internal components.
        
        Args:
            unit_elements: Dictionary mapping subscripts to tuples of unit elements
            log_units: List of tuples containing (log_unit_element, dimension)
        
        Returns:
            A new Unit instance
        """
        instance: "Unit" = object.__new__(cls)
        object.__setattr__(instance, "_unit_elements", unit_elements)
        object.__setattr__(instance, "_log_units", log_units)
        return instance

########################################################
# Helper methods
########################################################



    ########################################################
    # Properties
    ########################################################

    @property
    def unit_elements(self) -> dict[str, Tuple[UnitElement, ...]]:
        """Get the unit elements dictionary."""
        return dict(self._unit_elements)

    @property
    def log_units(self) -> List[Tuple[UnitElement, "Dimension"]]:
        """Get the log units list."""
        return list(self._log_units)

    @property
    def dimension(self) -> Dimension:
        """Get the dimension of this unit."""
        # Start with dimensionless dimension
        from .dimension import Dimension
        if hasattr(self, "_dimension"):
            return self._dimension
        else:
            result_dimension: Dimension = Dimension(self)
            object.__setattr__(self, "_dimension", result_dimension)
        
        return result_dimension
    
    @property
    def factor(self) -> float:
        """Get the conversion factor to canonical units."""
        factor = 1.0
        for elements in self._unit_elements.values():
            for element in elements:
                factor *= element.canonical_factor
        return factor
    
    @property
    def offset(self) -> float:
        """Get the offset for unit conversion."""
        # Only one offset is allowed per unit
        offset = 0.0
        for elements in self._unit_elements.values():
            for element in elements:
                if element.canonical_offset != 0.0:
                    if offset != 0.0:
                        raise ValueError("Only one offset is allowed per unit")
                    offset = element.canonical_offset
        return offset

    @property
    def is_dimensionless(self) -> bool:
        """Check if this unit is dimensionless."""
        if not self._unit_elements and not self._log_units:
            return True
        
        # Check if all unit elements have zero exponents
        for elements in self._unit_elements.values():
            for element in elements:
                if abs(element.exponent) > EPSILON:
                    return False
        
        # Check if all log units have dimensionless inner dimensions
        for _, inner_dimension in self._log_units:
            if not inner_dimension.is_dimensionless:
                return False
        
        return True
    
    @property
    def includes_log_level(self) -> bool:
        """Check if this unit includes logarithmic components."""
        return len(self._log_units) > 0    

    ########################################################
    # Arithmetic operations
    ########################################################

    def __add__(self, other: "Unit") -> "Unit":
        """
        Add two units.
        
        For regular units, addition is only valid if the units have identical dimensions.
        For logarithmic units, addition follows logarithmic identities.
        
        Args:
            other: The unit to add to this one
        
        Returns:
            A new Unit representing the sum
        
        Raises:
            ValueError: If the units cannot be added
        
        Examples:
            # Regular units (must be identical)
            force1 = Unit("kg*m/s^2")
            force2 = Unit("N")
            total_force = force1 + force2  # Valid
            
            # Logarithmic units
            log_mass = Unit("dec(kg)")
            log_length = Unit("dec(m)")
            log_mass_length = log_mass + log_length  # dec(kg*m)
        """
        if not Unit.is_valid_for_addition(self, other):
            raise ValueError("Invalid units for addition.")
        
        # If both units have log components, apply logarithmic identities
        if self._log_units and other._log_units:
            # For log units: log(a) + log(b) = log(a*b)
            # Work with the inner dimensions and apply logarithmic identities
            
            # Get the inner dimensions from both log units
            self_inner_dim, other_inner_dim = self._log_units[0][1], other._log_units[0][1]
            
            # Apply logarithmic identity: log(a) + log(b) = log(a*b)
            # This means we multiply the inner dimensions
            combined_inner_dim = self_inner_dim * other_inner_dim
            
            # Create a new log unit with the combined inner dimension
            log_unit_element, _ = self._log_units[0]
            return Unit._construct(MappingProxyType({}), [(log_unit_element, combined_inner_dim)])
        
        # If only one unit has log components, just return the unit with log components
        elif self._log_units:
            return Unit._construct(self._unit_elements, self._log_units)
        elif other._log_units:
            return Unit._construct(other._unit_elements, other._log_units)
        else:
            # No log units, just return the unit elements (should be the same for both)
            return Unit._construct(self._unit_elements)

    def __sub__(self, other: "Unit") -> "Unit":
        """
        Subtract two units.
        
        For regular units, subtraction is only valid if the units have identical dimensions.
        For logarithmic units, subtraction follows logarithmic identities.
        
        Args:
            other: The unit to subtract from this one
        
        Returns:
            A new Unit representing the difference
        
        Raises:
            ValueError: If the units cannot be subtracted
        
        Examples:
            # Regular units (must be identical)
            force1 = Unit("kg*m/s^2")
            force2 = Unit("N")
            net_force = force1 - force2  # Valid
            
            # Logarithmic units
            log_mass = Unit("dec(kg)")
            log_length = Unit("dec(m)")
            log_ratio = log_mass - log_length  # dec(kg/m)
        """
        if not Unit.is_valid_for_addition(self, other):
            raise ValueError("Invalid units for subtraction.")
        
        # If both units have log components, apply logarithmic identities
        if self._log_units and other._log_units:
            # For log units: log(a) - log(b) = log(a/b)
            # Work with the inner dimensions and apply logarithmic identities
            
            # Get the inner dimensions from both log units
            self_inner_dim, other_inner_dim = self._log_units[0][1], other._log_units[0][1]
            
            # Apply logarithmic identity: log(a) - log(b) = log(a/b)
            # This means we divide the inner dimensions
            combined_inner_dim = self_inner_dim / other_inner_dim
            
            # If the result is dimensionless, return a log unit with dimensionless inner dimension
            # This will format as "dec"
            if combined_inner_dim.is_dimensionless:
                log_unit_element, _ = self._log_units[0]
                return Unit._construct(MappingProxyType({}), [(log_unit_element, combined_inner_dim)])
            
            # Create a new log unit with the combined inner dimension
            log_unit_element, _ = self._log_units[0]
            return Unit._construct(MappingProxyType({}), [(log_unit_element, combined_inner_dim)])
        
        # If only one unit has log components, just return the unit with log components
        elif self._log_units:
            return Unit._construct(self._unit_elements, self._log_units)
        elif other._log_units:
            return Unit._construct(other._unit_elements, other._log_units)
        else:
            # No log units, just return the unit elements (should be the same for both)
            return Unit._construct(self._unit_elements, [])

    def __mul__(self, other: "Unit") -> "Unit":
        """
        Multiply two units.
        
        Args:
            other: The unit to multiply by
        
        Returns:
            A new unit representing the product
        
        Examples:
            Unit("m") * Unit("s") -> Unit("m*s")
            Unit("kg") * Unit("m/s^2") -> Unit("kg*m/s^2")
        """
        # Handle numpy arrays by returning NotImplemented to trigger __rmul__
        if isinstance(other, np.ndarray):
            return NotImplemented
            
        # Combine unit elements
        new_unit_elements: dict[str, List[UnitElement]] = {}
        
        # Collect all elements from both units
        all_elements: dict[Tuple[str, UnitSymbol, Optional[UnitPrefix]], UnitElement] = {}
        for subscript, elements in self._unit_elements.items():
            for element in elements:
                if not isinstance(element.unit_symbol, UnitSymbol):
                    raise ValueError(f"Invalid unit symbol: {element.unit_symbol}")
                key: Tuple[str, UnitSymbol, Optional[UnitPrefix]] = (subscript, element.unit_symbol, element.prefix)
                if key in all_elements:
                    # Combine exponents
                    combined_exponent: float = all_elements[key].exponent + element.exponent
                    all_elements[key] = UnitElement(element.prefix, element.unit_symbol, combined_exponent)
                else:
                    all_elements[key] = element
        
        for subscript, elements in other._unit_elements.items():
            for element in elements:
                if not isinstance(element.unit_symbol, UnitSymbol):
                    raise ValueError(f"Invalid unit symbol: {element.unit_symbol}")
                key: Tuple[str, UnitSymbol, Optional[UnitPrefix]] = (subscript, element.unit_symbol, element.prefix)
                if key in all_elements:
                    # Combine exponents
                    combined_exponent = all_elements[key].exponent + element.exponent
                    all_elements[key] = UnitElement(element.prefix, element.unit_symbol, combined_exponent)
                else:
                    all_elements[key] = element
        
        # Group by subscript
        for (subscript, _, _), element in all_elements.items():
            if abs(element.exponent) > EPSILON:  # Only include non-zero exponents
                if subscript not in new_unit_elements:
                    new_unit_elements[subscript] = []
                new_unit_elements[subscript].append(element)
        
        # Convert lists to tuples
        new_unit_elements_as_tuples: dict[str, Tuple[UnitElement, ...]] = {k: tuple(v) for k, v in new_unit_elements.items()}
        
        # Combine log units
        new_log_units = list(self._log_units)
        new_log_units.extend(other._log_units)
        
        # Create the unit and reduce it
        result = self._construct(MappingProxyType(new_unit_elements_as_tuples), new_log_units)
        return self.reduce_unit(result)

    def __truediv__(self, other: "Unit") -> "Unit":
        """
        Divide two units.
        
        Args:
            other: The unit to divide by
        
        Returns:
            A new unit representing the quotient
        
        Examples:
            Unit("m") / Unit("s") -> Unit("m/s")
            Unit("kg*m/s^2") / Unit("m") -> Unit("kg/s^2")
        """
        # Invert the other unit and multiply
        result = self * other.invert()
        return self.reduce_unit(result)

    def __pow__(self, exponent: float|int) -> "Unit":
        """
        Raise a unit to a power.
        
        Args:
            exponent: The power to raise the unit to
        
        Returns:
            A new unit raised to the specified power
        
        Examples:
            Unit("m") ** 2 -> Unit("m^2")
            Unit("m/s") ** 3 -> Unit("m^3/s^3")
        """
        # Raise unit elements to power
        new_unit_elements: dict[str, Tuple[UnitElement, ...]] = {}
        for subscript, elements in self._unit_elements.items():
            new_elements = tuple(element.pow(exponent) for element in elements)
            new_unit_elements[subscript] = new_elements
        
        # For log units, we don't raise them to powers in the same way
        # Log units maintain their structure
        new_log_units: List[Tuple[UnitElement, Dimension]] = []
        for log_unit_element, inner_dimension in self._log_units:
            new_log_unit_element = UnitElement(log_unit_element.prefix, log_unit_element.unit_symbol, log_unit_element.exponent * exponent)
            new_log_units.append((new_log_unit_element, inner_dimension))
        
        return self._construct(MappingProxyType(new_unit_elements), new_log_units)

    def __invert__(self) -> "Unit":
        """
        Invert a unit.
        
        Returns:
            A new unit representing the inverse
        
        Examples:
            ~Unit("m") -> Unit("1/m")
            ~Unit("kg*m/s^2") -> Unit("s^2/(kg*m)")
        """
        return self ** (-1)

    def invert(self) -> "Unit":
        """Invert a unit (same as ~self)."""
        return ~self

    def log(self) -> "Unit":
        """
        Take the logarithm of a unit.
        
        The logarithm is always applied to the canonical value, so prefixes are stripped.
        For example, dec(mA) and dec(kA) both become dec(A) because they represent
        decades of the canonical current unit.
        
        Returns:
            A new unit representing the logarithm
        
        Examples:
            Unit("m").log() -> dec(L)  # decades of length
            Unit("mA").log() -> dec(A)  # decades of current (not decades of milliampere)
            Unit("km").log() -> dec(m)  # decades of length (not decades of kilometer)
        """
        # Create a new unit with this unit's dimension in the log component
        from .unit_symbol import LOG_UNIT_SYMBOLS
        log_unit_element = UnitElement(None, LOG_UNIT_SYMBOLS.BASE_10, 1.0)
        log_units = [(log_unit_element, self.dimension)]
        
        return self._construct(MappingProxyType({}), log_units)

    def exp(self) -> "Unit":
        """
        Exponentiate a unit (inverse of log).
        
        Only valid for dimensionless units or single logarithmic units.
        
        Returns:
            A new unit representing the exponential
        
        Raises:
            ValueError: If the unit is not valid for exponential
        
        Examples:
            # Dimensionless unit
            dimensionless = Unit("")
            exp_dimless = dimensionless.exp()  # Still dimensionless
            
            # Logarithmic unit
            log_mass = Unit("dec(kg)")
            mass = log_mass.exp()  # kg
        """
        # If dimensionless, return dimensionless
        if self.is_dimensionless and not self._log_units:
            return Unit.dimensionless_unit()
        
        # If it's a single log unit, extract the inner dimension
        elif len(self._log_units) == 1 and not self._unit_elements:
            log_unit_element, inner_dimension = self._log_units[0]
            if abs(log_unit_element.exponent - 1.0) <= EPSILON:
                # Create a unit from the inner dimension
                # The inner dimension represents what the log unit is logging
                # We need to create a unit that represents this dimension
                # For now, let's create a unit with the dimension as the inner dimension of a log unit
                # This is a temporary approach - we need a better way to convert Dimension to Unit
                from .unit_symbol import LOG_UNIT_SYMBOLS
                log_unit_element = UnitElement(None, LOG_UNIT_SYMBOLS.BASE_10, 1.0)
                # Create a unit with the inner dimension as a regular unit element
                # This is not quite right - we need to think about this differently
                return Unit._construct(MappingProxyType({}), [(log_unit_element, inner_dimension)])
            else:
                raise ValueError("Invalid unit for exponential.")
        else:
            raise ValueError("Invalid unit for exponential.")
    
########################################################
    # Compatibility and comparison
########################################################

    @classmethod
    def are_compatible(cls, *others: Optional[Union["Unit", Dimension, HasUnit]]) -> bool:
        """
        Check if this unit is compatible with another unit or dimension.
        """        
        return Dimension.are_compatible(*others)

    def compatible_to(self, *others: Union["Unit", Dimension, HasUnit]) -> bool:
        """
        Check if this unit is compatible with another unit or dimension.
        
        Args:
            *others: The units or dimensions to check compatibility with
        
        Returns:
            True if compatible, False otherwise
        """
        return self.dimension.compatible_to(*others)
    
    def effectively_equal_to(self, other: "Unit") -> bool:
        """
        Check if this unit is effectively equal to another unit, meaning that they have the same dimension, factor and offset.
        """
        if self.dimension != other.dimension:
            return False
        if self.factor != other.factor:
            return False
        if self.offset != other.offset:
            return False
        return True
    
    @classmethod
    def effectively_equal(cls, *units: "Unit") -> bool:
        """
        Check if units are effectively equal, meaning that they have the same dimension, factor and offset.
        """
        for unit in units:
            if not unit.effectively_equal_to(units[0]):
                return False
        return True

    @staticmethod
    def is_valid_for_addition(unit_1: "Unit", unit_2: "Unit") -> bool:
        """
        Check if two units can be added or subtracted.
        
        Two units can be added/subtracted if they have compatible dimensions
        and the same log units.
        
        Args:
            unit_1: First unit
            unit_2: Second unit
        
        Returns:
            True if the units can be added/subtracted, False otherwise
        
        Examples:
            force1 = Unit("kg*m/s^2")
            force2 = Unit("N")
            assert Unit.is_valid_for_addition(force1, force2)
            
            # Different units cannot be added
            mass = Unit("kg")
            length = Unit("m")
            assert not Unit.is_valid_for_addition(mass, length)
        """
        # For log units, we don't need to check dimension compatibility
        # since we're doing logarithmic arithmetic
        if not unit_1._log_units and not unit_2._log_units:
            # Only check dimension compatibility for non-log units
            if not Dimension.is_valid_for_addition(unit_1.dimension, unit_2.dimension):
                return False
        
        # For log units, we only need to check that both have log units
        # The log function type and inner dimensions can be different (they'll be combined in arithmetic)
        if len(unit_1._log_units) != len(unit_2._log_units):
            return False
        
        return True

    def __eq__(self, other: object) -> bool:
        """Check if two units are equal."""
        if not isinstance(other, Unit):
            return False
        
        return (self._unit_elements == other._unit_elements and
                self._log_units == other._log_units)

    def __ne__(self, other: object) -> bool:
        """Check if two units are not equal."""
        return not self == other

    def __hash__(self) -> int:
        """Hash based on unit elements and log units."""
        unit_elements_tuple = tuple(sorted(self._unit_elements.items()))
        log_units_tuple = tuple(sorted(self._log_units))
        return hash((unit_elements_tuple, log_units_tuple))

########################################################
    # String representation
########################################################

    def __str__(self) -> str:
        """Get string representation of the unit."""
        return self.format_string()

    def format_string(self, as_fraction: bool = True) -> str:
        """
        Format the unit as a string.
        
        Args:
            as_fraction: Whether to format as a fraction
        
        Returns:
            String representation of the unit
        
        Examples:
            Unit("m").format_string() -> "m"
            Unit("m/s").format_string() -> "m/s"
            Unit("kg*m/s^2").format_string() -> "kg*m/s^2"
        """
        # Special case: log units with dimensionless inner dimensions should format as "dec"
        if self.is_dimensionless and self._log_units:
            log_unit_element = self._log_units[0][0]
            log_symbol = log_unit_element.unit_symbol.value.symbols[0]
            return f"{log_unit_element.prefix}{log_symbol}"
        
        if self.is_dimensionless:
            return ""
        
        # Format regular unit elements
        nominator_parts: List[str] = []
        denominator_parts: List[str] = []
        
        for subscript, elements in self._unit_elements.items():
            for element in elements:
                if abs(element.exponent) > EPSILON:
                    part_str, position = element.format_string(as_fraction)
                    if subscript:
                        # Insert subscript before the exponent
                        if "^" in part_str:
                            # Split at ^ to put subscript before exponent
                            base_part, exponent_part = part_str.split("^", 1)
                            part_str = f"{base_part}_{subscript}^{exponent_part}"
                        else:
                            # No exponent, just append subscript
                            part_str = f"{part_str}_{subscript}"
                    
                    if position == "nominator":
                        nominator_parts.append(part_str)
                    else:  # denominator
                        denominator_parts.append(part_str)
        
        # Format log units (always in nominator for now)
        for log_unit_element, inner_dimension in self._log_units:
            # Use the first symbol from the log unit symbol
            log_symbol = log_unit_element.unit_symbol.value.symbols[0]
            part = f"{log_unit_element.prefix}{log_symbol}({inner_dimension.format_string()})"
            if abs(log_unit_element.exponent - 1.0) > EPSILON:
                part += f"^{log_unit_element.exponent}"
            nominator_parts.append(part)
        
        # Combine parts
        if not nominator_parts and not denominator_parts:
            return ""
        
        result = "*".join(nominator_parts) if nominator_parts else "1"
        
        if denominator_parts:
            denominator_str = "/".join(denominator_parts)
            result = f"{result}/{denominator_str}"
        
        return result

    def __repr__(self) -> str:
        """Get detailed string representation of the unit."""
        try:
            return f"Unit('{self.format_string()}')"
        except RecursionError:
            return f"Unit(<recursion_error>)"
        except Exception:
            return f"Unit(<format_error>)"

    ########################################################
    # Parsing
    ########################################################

    @classmethod
    def _parse_string(cls, string: str) -> Tuple[MappingProxyType[str, Tuple[UnitElement, ...]], List[Tuple[UnitElement, "Dimension"]]]:
        """
        Parse a unit string into internal components.
        
        Args:
            string: The unit string to parse
        
        Returns:
            Dictionary with 'unit_elements' and 'log_units'
        
        Examples:
            "m" -> {"unit_elements": {"": (UnitElement("", METER, 1),)}, "log_units": {}}
            "m/s" -> {"unit_elements": {"": (UnitElement("", METER, 1), UnitElement("", SECOND, -1))}, ...}
            "dec(N)" -> {"unit_elements": {}, "log_units": [(UnitElement("", LOG_UNIT_SYMBOLS.BASE_10, 1), Dimension("N"))]}
        """
        if not string.strip():
            return (MappingProxyType({}), [])
        
        # Handle dimensionless unit "1"
        if string.strip() == "1":
            return (MappingProxyType({}), [])
        
        # Parse the string into components
        unit_elements: dict[str, List[UnitElement]] = {}
        log_units: List[Tuple[UnitElement, "Dimension"]] = []
        
        # Split by multiplication
        parts = seperate_string(string, "nominator")
        
        for separator, part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Handle special cases (log units) FIRST
            log_unit_found = False
            from .unit_symbol import LOG_UNIT_SYMBOLS
            for log_symbol_enum in LOG_UNIT_SYMBOLS:
                for log_symbol in log_symbol_enum.value.symbols:
                    if log_symbol in part and "(" in part:
                        # Find the log symbol and any prefix
                        log_start = part.find(log_symbol)
                        if log_start > 0:
                            # There's a prefix before the log symbol
                            prefix_string: str = part[:log_start]
                            log_part = part[log_start:]
                        else:
                            # No prefix
                            prefix_string: str = ""
                            log_part = part

                        prefix: Optional[UnitPrefix] = UnitPrefix.get_prefix(prefix_string)
                        
                        if log_part.startswith(f"{log_symbol}("):
                            # Find the closing parenthesis
                            paren_count = 1  # Start at 1 since we're after the opening parenthesis
                            inner_start = len(log_symbol) + 1
                            inner_end = -1
                            
                            for i, char in enumerate(log_part[inner_start:], inner_start):
                                if char == '(':
                                    paren_count += 1
                                elif char == ')':
                                    paren_count -= 1
                                    if paren_count == 0:
                                        inner_end = i
                                        break
                            
                            if inner_end != -1:
                                # Extract the inner content and any remaining part (exponent)
                                inner_content = log_part[inner_start:inner_end]
                                remaining_part = log_part[inner_end + 1:]
                                
                                # Validate that inner content is not empty
                                if not inner_content.strip():
                                    raise ValueError(f"Empty content in log unit: {part}")
                                
                                # Parse the inner content as a dimension
                                from .dimension import Dimension
                                
                                # Convert unit symbols in inner_content to dimension symbols
                                # Note: Logarithmic units are always applied to canonical values,
                                # so dec(mA) becomes dec(A) - decades of current, not decades of milliampere
                                
                                # Parse the inner content as a dimension
                                # The Dimension parser will handle unit symbols and convert them to dimension symbols
                                dimension_string = inner_content
                                inner_dimension = Dimension(dimension_string)
                                
                                # Create the log unit element
                                log_unit_symbol: Optional[LOG_UNIT_SYMBOLS] = None
                                for log_symbol_enum in LOG_UNIT_SYMBOLS:
                                    if log_symbol in log_symbol_enum.value.symbols:
                                        log_unit_symbol = log_symbol_enum
                                        break
                                if log_unit_symbol is None:
                                    raise ValueError(f"Invalid log unit symbol: {log_symbol}")
                                
                                # Parse exponent from remaining part
                                exponent = 1.0
                                if remaining_part.startswith("^"):
                                    try:
                                        exponent = float(remaining_part[1:])
                                    except ValueError:
                                        raise ValueError(f"Invalid exponent in log unit: {part}")
                                
                                # Create UnitElement for the log symbol with prefix and exponent
                                log_unit_element = UnitElement(prefix, log_unit_symbol, exponent)
                                
                                log_units.append((log_unit_element, inner_dimension))
                                log_unit_found = True
                                break
                    elif log_symbol in part and "(" not in part:
                        # Log symbol without parentheses is invalid
                        raise ValueError(f"Log symbol '{log_symbol}' must be followed by parentheses: {part}")
            
            if not log_unit_found:
                # Handle subscripts for regular unit elements
                # Parse the part to extract unit_part and subscript, handling both ^ and _ in any order
                
                if part.count('^') == 0 and part.count('_') == 0:
                    # No exponent or subscript
                    unit_part = part
                    subscript = ""
                
                elif part.count('^') == 1 and part.count('_') == 1:
                    # Both exponent and subscript present
                    caret_pos = part.index('^')
                    underscore_pos = part.index('_')
                    
                    if caret_pos < underscore_pos:
                        # Format: unit^exp_subscript
                        # Split at underscore to separate unit^exp from subscript
                        unit_with_exp, subscript = part.split("_", 1)
                        unit_part = unit_with_exp
                    else:
                        # Format: unit_subscript^exp
                        # Split at underscore to separate unit from subscript^exp
                        unit_part, subscript_with_exp = part.split("_", 1)
                        # The subscript_with_exp contains the subscript and exponent
                        # We need to extract just the subscript part and reconstruct unit_part with exponent
                        if "^" in subscript_with_exp:
                            subscript, exponent_part = subscript_with_exp.split("^", 1)
                            # Reconstruct unit_part to include the exponent
                            unit_part = unit_part + "^" + exponent_part
                        else:
                            subscript = subscript_with_exp
                
                elif part.count('^') == 1 and part.count('_') == 0:
                    # Only exponent present
                    unit_part = part
                    subscript = ""

                elif part.count('^') == 0 and part.count('_') == 1:
                    # Only subscript present
                    unit_part, subscript = part.split("_", 1)

                else:
                    raise ValueError(f"Invalid unit string: {part}")
                
                # Regular unit element
                position = "nominator" if separator == "*" else "denominator"
                unit_element = UnitElement.parse_string(unit_part, position)
                
                if subscript not in unit_elements:
                    unit_elements[subscript] = []

                unit_elements[subscript].append(unit_element) # type: ignore
        
        # Convert lists to tuples
        unit_elements_as_tuples: dict[str, tuple[UnitElement, ...]] = {k: tuple(v) for k, v in unit_elements.items()}
        
        return MappingProxyType(unit_elements_as_tuples), log_units

########################################################
    # Unit conversion
########################################################

    @overload
    def to_canonical_value(self, value_in_unit: float|int) -> float:
        """
        Convert a value from this unit to canonical units.
        """
        ...
    @overload
    def to_canonical_value(self, value_in_unit: complex) -> complex:
        """ 
        Convert a complex value from this unit to canonical units.
        """
        ...
    @overload
    def to_canonical_value(self, value_in_unit: np.ndarray) -> np.ndarray:
        """
        Convert a numpy array from this unit to canonical units.
        """
        ...
    @overload
    def to_canonical_value(self, value_in_unit: "pd.Series[Any]") -> "pd.Series[Any]":
        """
        Convert a pandas series from this unit to canonical units.
        """
        ...
    def to_canonical_value(self, value_in_unit: Union[float, int, complex, np.ndarray, "pd.Series[Any]"]) -> Union[float, int, complex, np.ndarray, "pd.Series[Any]"]:
        """
        Convert a value from this unit to canonical units.
        
        Args:
            value_in_unit: The value in this unit
        
        Returns:
            The value in canonical units
        
        Examples:
            Unit("km").to_canonical_value(2.5) -> 2500.0
            Unit("°C").to_canonical_value(25) -> 298.15
        """
        if isinstance(value_in_unit, (int, float)):
            return value_in_unit * self.factor + self.offset
        elif isinstance(value_in_unit, complex):
            return value_in_unit * self.factor + self.offset
        elif isinstance(value_in_unit, np.ndarray): # type: ignore
            return value_in_unit * self.factor + self.offset
        elif isinstance(value_in_unit, pd.Series): # type: ignore
            return pd.Series(value_in_unit * self.factor + self.offset) # type: ignore
        else:
            raise ValueError(f"Invalid value type: {type(value_in_unit)}")
    
    @overload
    def from_canonical_value(self, canonical_value: float) -> float:
        """
        Convert a value from canonical units to this unit.
        """
        ...
    @overload
    def from_canonical_value(self, canonical_value: complex) -> complex:
        """
        Convert a complex value from canonical units to this unit.
        """
        ...
    @overload
    def from_canonical_value(self, canonical_value: np.ndarray) -> np.ndarray:
        """
        Convert a numpy array from canonical units to this unit.
        """
        ...
    @overload
    def from_canonical_value(self, canonical_value: "pd.Series[Any]") -> "pd.Series[Any]":
        """
        Convert a pandas series from canonical units to this unit.
        """
        ...
    def from_canonical_value(self, canonical_value: Union[float, int, complex, np.ndarray, "pd.Series[Any]"]) -> Union[float, int, complex, np.ndarray, "pd.Series[Any]"]:
        """
        Convert a value from canonical units to this unit.
        
        Args:
            canonical_value: The value in canonical units
        
        Returns:
            The value in this unit
        
        Examples:
            Unit("km").from_canonical_value(2500.0) -> 2.5
            Unit("°C").from_canonical_value(298.15) -> 25.0
        """
        if isinstance(canonical_value, (int, float)):
            return (canonical_value - self.offset) / self.factor
        elif isinstance(canonical_value, complex):
            return (canonical_value - self.offset) / self.factor
        elif isinstance(canonical_value, np.ndarray): # type: ignore
            return (canonical_value - self.offset) / self.factor
        elif isinstance(canonical_value, pd.Series): # type: ignore
            return pd.Series((canonical_value - self.offset) / self.factor) # type: ignore
        else:
            raise ValueError(f"Invalid value type: {type(canonical_value)}")
        
    @classmethod
    def convert(cls, value: Union[float, int, complex, np.ndarray, "pd.Series[Any]"], from_unit: "Unit", to_unit: "Unit") -> Union[float, int, complex, np.ndarray, "pd.Series[Any]"]:
        """
        Create a unit from a canonical value.
        """
        return to_unit.from_canonical_value(from_unit.to_canonical_value(value))
        
########################################################
    # Scalar/Array creation
########################################################

    @overload
    def __rmul__(self, other: int) -> "RealUnitedScalar": # type: ignore
        """
        Multiply a unit by a scalar. Represents int * Unit.
        
        Args:
            other: The scalar to multiply by
        
        Returns:
            A new unit representing the product
        
        Examples:
            Unit("m") * 2 -> Unit("2*m")
            Unit("kg") * 2 -> Unit("2*kg")
        """
        ...

    @overload
    def __rmul__(self, other: float) -> "RealUnitedScalar":
        """
        Multiply a unit by a scalar. Represents float * Unit.
        
        Args:
            other: The scalar to multiply by
        
        Returns:
            A new unit representing the product
        
        Examples:
            Unit("m") * 2 -> Unit("2*m")
            Unit("kg") * 2 -> Unit("2*kg")
        """
        ...

    @overload
    def __rmul__(self, other: complex) -> "ComplexUnitedScalar":
        """
        Multiply a unit by a complex scalar. Represents complex * Unit.
        
        Args:
            other: The complex scalar to multiply by
        
        Returns:
            A new unit representing the product
        
        Examples:
            Unit("m") * 2.0 -> Unit("2.0*m")
            Unit("kg") * 2.0 -> Unit("2.0*kg")
        """
        ...

    @overload
    def __rmul__(self, other: Sequence[float|int]|Sequence[complex]) -> "RealUnitedArray|ComplexUnitedArray":
        """
        Multiply a unit by a numpy array. Represents np.ndarray * Unit.
        """
        ...

    def __rmul__(self, other: float|int|complex|Sequence[float|int]|Sequence[complex]) -> "RealUnitedScalar|ComplexUnitedScalar|RealUnitedArray|ComplexUnitedArray":
        """
        Multiply a unit by a scalar. Represents float|int|complex|np.ndarray * Unit.
        """
        
        if isinstance(other, (float, int)):
            from .._scalars.real_united_scalar import RealUnitedScalar
            return RealUnitedScalar(other, self)
        elif isinstance(other, complex):
            from .._scalars.complex_united_scalar import ComplexUnitedScalar # type: ignore
            raise NotImplementedError("Complex multiplication is not supported for units.")
        elif isinstance(other, Sequence): # type: ignore
            from .._arrays.real_united_array import RealUnitedArray
            from .._arrays.complex_united_array import ComplexUnitedArray
            numpy_array: np.ndarray = np.array(other)
            if np.iscomplexobj(numpy_array):
                return ComplexUnitedArray(numpy_array, self) # type: ignore
            else:
                return RealUnitedArray(numpy_array, self)
        else:
            raise ValueError(f"Invalid scalar: {other}")

    def __rtruediv__(self, other: Union[float, int, complex, Sequence[float|int|complex]]) -> Union["RealUnitedScalar", "ComplexUnitedScalar", "RealUnitedArray", "ComplexUnitedArray"]:
        """
        Divide a scalar/array by this unit.
        
        Args:
            other: The scalar or array to divide
        
        Returns:
            A united scalar or array with the inverse of this unit
        
        Examples:
            2.5 / Unit("m") -> RealUnitedScalar(2.5, Unit("1/m"))
        """
        return (~self).__rmul__(other)

########################################################
    # Other
########################################################

    @classmethod
    def get_value(cls, item: Optional[VALUE_TYPE|SCALAR_TYPE], unit: Optional["Unit"]=None) -> Optional[VALUE_TYPE]:
        """
        Get a value from an item and unit in a very flexible way.

        Args:
            item: The item to get the value from
            unit: The unit to get the value from

        Returns:
            The value in the unit

        Examples:
            Unit.get_value(2.5, Unit("m")) -> 2.5
            Unit.get_value(2.5, Unit("km")) -> 2500.0
            Unit.get_value(2.5, Unit("km")) -> 2500.0
            Unit.get_value(2.5, Unit("km")) -> 2500.0
        """

        match (item, unit):
            case (None, None):
                return None
            case (None, _):
                return None
            case (_, None):
                if isinstance(item, HasUnit):
                    raise ValueError("Item has a unit, but no unit was provided")
                else:
                    return item
            case (_, _):
                if isinstance(item, HasUnit):
                    if item.compatible_to(unit):
                        return item.value_in_unit(unit)
                    else:
                        raise ValueError("Item is not compatible with the provided unit")
                else:
                    return item
            case _: # type: ignore
                raise ValueError("Invalid item and unit")

########################################################
    # Serialization
########################################################

    #----------- JSON Serialization -----------
    
    def to_json(self) -> str:
        """
        Convert the unit to JSON string representation.
        
        Returns:
            JSON string representation
        
        Examples:
            Unit("m").to_json() -> '"m"'
        """
        return self.format_string()
    
    @classmethod
    def from_json(cls, json_string: str) -> "Unit":
        """
        Create a unit from JSON string representation.
        
        Args:
            json_string: The JSON string representation
        
        Returns:
            A new Unit instance
        
        Examples:
            Unit.from_json('"m"') -> Unit("m")
        """
        return cls(json_string)

    #----------- HDF5 Serialization -----------

    def to_hdf5(self, hdf5_group: Group) -> None:
        """
        Save the unit to an HDF5 group.
        
        Args:
            hdf5_group: The HDF5 group to save to
        """
        hdf5_group["unit"] = self.format_string()

    @classmethod
    def from_hdf5(cls, hdf5_group: Group) -> "Unit":
        """
        Create a unit from an HDF5 group.
        
        Args:
            hdf5_group: The HDF5 group to read from
            
        Returns:
            A new Unit instance
        """
        unit_string: str = hdf5_group["unit"].asstr()[()] # type: ignore
        if isinstance(unit_string, bytes):
            unit_string = unit_string.decode("utf-8")
        return cls(unit_string)
    
    #----------- For pickle compatibility -----------
        
    def __getstate__(self) -> dict[str, Any]:
        """Custom pickle state management for slotted dataclass."""
        # Ensure _dimension is computed before pickling
        _ = self.dimension  # This will initialize _dimension if not already set
        
        # For slotted dataclasses, manually collect field values
        # Convert MappingProxyType to dict for pickling
        return {
            "_unit_elements": dict(self._unit_elements),  # Convert MappingProxyType to dict
            "_log_units": self._log_units,
            "_dimension": self._dimension
        }
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom pickle state restoration for slotted dataclass."""
        # Convert dict back to MappingProxyType
        from types import MappingProxyType
        if "_unit_elements" in state:
            state["_unit_elements"] = MappingProxyType(state["_unit_elements"])
        
        # Restore all attributes
        for key, value in state.items():
            object.__setattr__(self, key, value)
        
        # Ensure _dimension is properly set (in case it wasn't in the state)
        if not hasattr(self, "_dimension"):
            from .dimension import Dimension
            object.__setattr__(self, "_dimension", Dimension(self))

########################################################
    # Factory methods
########################################################

    @classmethod
    def dimensionless_unit(cls) -> "Unit":
        """
        Create a dimensionless unit.
        
        Returns:
            A dimensionless unit
        
        Examples:
            Unit.dimensionless_unit() -> Unit("")
        """
        return cls("")

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the unit cache."""
        clear_unit_cache()

    @property
    def reduced(self) -> "Unit":
        """
        Reduce this unit to its simplest form.
        """
        return self.reduce_unit(self)

    @staticmethod
    def reduce_unit(unit: "Unit") -> "Unit":
        """
        Attempt to reduce a unit by using derived units from NamedQuantity enums.
        
        This method tries to replace combinations of base units with derived units
        where possible, e.g., J/s -> W, kg*m/s^2 -> N, etc.
        
        Args:
            unit: The unit to reduce
            
        Returns:
            A new unit that is as reduced as possible
        
        Examples:
            Unit.reduce_unit(Unit("J/s")) -> Unit("W")
            Unit.reduce_unit(Unit("kg*m/s^2")) -> Unit("N")
            Unit.reduce_unit(Unit("kg*m^2/s^2")) -> Unit("J")
        """
        if unit.is_dimensionless:
            return unit
        
        # Import here to avoid circular imports
        from .reduce_unit_elements import reduce_unit_elements
        from .dimension_symbol import UnitSymbol

        def consolidate_unit_elements(unit_elements: Sequence[UnitElement]) -> Sequence[UnitElement]:
            """
            Consolidate a group of unit elements with the same symbol by combining their exponents.
            
            Args:
                unit_elements: Tuple of unit elements to consolidate
                
            Returns:
                Tuple of consolidated unit elements
            """
            if len(unit_elements) <= 1:
                return unit_elements
            
            # Group elements by their unit symbol and prefix
            grouped_elements: dict[tuple[UnitSymbol, Optional[UnitPrefix]], UnitElement] = {}
            for element in unit_elements:
                if not isinstance(element.unit_symbol, UnitSymbol):
                    raise AssertionError("Unit symbol is not a UnitSymbol")
                key: tuple[UnitSymbol, Optional[UnitPrefix]] = (element.unit_symbol, element.prefix)
                if key in grouped_elements:
                    # Combine exponents
                    combined_exponent: float = grouped_elements[key].exponent + element.exponent
                    grouped_elements[key] = UnitElement(element.prefix, element.unit_symbol, combined_exponent)
                else:
                    grouped_elements[key] = element
            
            # Return only elements with non-zero exponents
            result = [element for element in grouped_elements.values() if abs(element.exponent) > EPSILON]
            return tuple(result)

        # Handle log units by converting them to dimension symbols
        new_log_units: list[tuple[UnitElement, Dimension]] = []
        for log_unit_element, inner_dimension in unit._log_units:
            # For log units, we keep the log structure but reduce the inner content
            new_log_units.append((log_unit_element, inner_dimension))

        # Process regular unit elements
        new_unit_elements: dict[str, tuple[UnitElement, ...]] = {}
        
        # First, consolidate all elements across subscripts
        all_elements: list[tuple[str, UnitElement]] = []
        for subscript, elements in unit._unit_elements.items():
            for element in elements:
                # Create a new element with the subscript in the unit symbol if needed
                if subscript and subscript != "":
                    # For now, we'll keep the subscript structure as is
                    all_elements.append((subscript, element))
                else:
                    all_elements.append(("", element))
        
        # Group by subscript and consolidate within each group
        grouped_by_subscript: dict[str, list[UnitElement]] = {}
        for subscript, element in all_elements:
            if subscript not in grouped_by_subscript:
                grouped_by_subscript[subscript] = []
            grouped_by_subscript[subscript].append(element)
        
        # Reduce each subscript group
        for subscript, elements in grouped_by_subscript.items():
            consolidated_elements: Sequence[UnitElement] = consolidate_unit_elements(elements)
            reduced_elements: Sequence[UnitElement] = reduce_unit_elements(consolidated_elements)
            if len(reduced_elements) > 0:
                new_unit_elements[subscript] = tuple(reduced_elements)
        
        return Unit._construct(MappingProxyType(new_unit_elements), new_log_units)
    
    @staticmethod
    def extract(obj: Any) -> Optional["Unit"]:
        """
        Extract a unit from an object.
        """
        if isinstance(obj, Unit):
            return obj
        elif isinstance(obj, Dimension):
            return obj.canonical_unit
        elif isinstance(obj, HasUnit):
            return obj.unit
        elif isinstance(obj, NamedQuantity):
            return obj.unit
        else:
            return None
    
    @overload
    @staticmethod
    def get_unit_options(symbols_and_prefixes: dict[UnitSymbol, Sequence[UnitPrefix]]) -> list["Unit"]:
        """
        Get all possible units for a given set of symbols and prefixes.
        """
        ...

    @overload
    @staticmethod
    def get_unit_options(symbols_and_prefixes: dict[str, dict[UnitSymbol, Sequence[UnitPrefix]]]) -> list["Unit"]:
        """
        Get all possible units for a given set of symbols and prefixes for a given subscript.
        """
        ...

    @staticmethod
    def get_unit_options(symbols_and_prefixes: dict[str, dict[UnitSymbol, Sequence[UnitPrefix]]]|dict[UnitSymbol, Sequence[UnitPrefix]]) -> list["Unit"]:
        """
        Get all possible units for a given set of symbols and prefixes.
        """

        def create_unit_list(subscript: str, symbols_and_prefixes: dict[UnitSymbol, Sequence[UnitPrefix]]) -> list["Unit"]:
            raise NotImplementedError("Not implemented")

        unit_list: list["Unit"] = []
        if len(symbols_and_prefixes) > 0:
            first_element: Any = next(iter(symbols_and_prefixes.values()))
            if isinstance(first_element, str):
                s_and_p_dict: dict[str, dict[UnitSymbol, Sequence[UnitPrefix]]] = cast(dict[str, dict[UnitSymbol, Sequence[UnitPrefix]]], symbols_and_prefixes)
                for subscript, symbols_and_prefixes in s_and_p_dict.items():
                    unit_list.extend(create_unit_list(subscript, symbols_and_prefixes))
            else:
                s_and_p: dict[UnitSymbol, Sequence[UnitPrefix]] = cast(dict[UnitSymbol, Sequence[UnitPrefix]], symbols_and_prefixes)
                unit_list = create_unit_list("", s_and_p)
        return unit_list
    
    @staticmethod
    def get_simple_units(unit: "Unit", number_of_units: int, unit_order: Literal["by_factor_ascending", "by_factor_descending"]="by_factor_ascending") -> list["Unit"]:
        """
        Get a list of simple units for a given unit.
        """

        def create_unit_list() -> list["Unit"]:
            raise NotImplementedError("Not implemented")

        units: list[Unit] = create_unit_list()
        if unit_order in ["by_factor_ascending", "by_factor_descending"]:
            units = sorted(units, key=lambda x: x.factor)
            if unit_order == "by_factor_descending":
                units.reverse()
        return units

########################################################
# Preset units
########################################################
 
DIMENSIONLESS_UNIT: Unit = Unit.dimensionless_unit()
# DECADE: Unit = Unit("dec(1)")  # Decade is a dimensionless log unit - commented out for now