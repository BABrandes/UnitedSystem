from dataclasses import dataclass
from typing import Tuple, List, TYPE_CHECKING, Any, overload, Union
from .utils.units.dimension_group import DimensionGroup
import h5py
from enum import Enum

if TYPE_CHECKING:
    from .unit import Unit

# Cache for canonical units to avoid repeated expensive calculations
_CANONICAL_UNIT_CACHE: dict["Dimension", "Unit"] = {}

class BASE_DIMENSION_SYMBOLS(Enum):
    MASS = ("M", 0)
    TIME = ("T", 1)
    LENGTH = ("L", 2)
    CURRENT = ("I", 3)
    TEMPERATURE = ("Θ", 4)
    AMOUNT = ("N", 5)
    LUMINOUS = ("J", 6)

@dataclass(frozen=True, slots=True, init=False)
class Dimension():
    dimension_groups: dict[str,DimensionGroup]

    def __new__(cls, value: Union[dict[str,DimensionGroup], str]):
        if isinstance(value, str):
            return super().__new__(cls)
        else:
            return super().__new__(cls)

    @overload
    def __init__(self, value: dict[str,DimensionGroup]):
        ...
    @overload
    def __init__(self, value: str):
        ...
    def __init__(self, value: Union[dict[str,DimensionGroup], str]):
        if isinstance(value, str):
            pass

################################################################################
# Arithmetic operations
################################################################################

    def __mul__(self, other: "Dimension") -> "Dimension":
        """
        Multiplies two dimensions.
        
        {T*L^2/M} * {M/L^3} = {T/L}
        {T*L^2/M} * {M/L^3*dec(M)} = {T/L*dec(M)}
        {T*L^2/M*dec(L)} * {M/L^3*dec(M)} = X
        """
        new_dimension_groups: dict[str,DimensionGroup] = {}
        for subscript, dimension_group in self.dimension_groups.items():
            if subscript in other.dimension_groups:
                new_dimension_groups[subscript] = dimension_group * other.dimension_groups[subscript]
            else:
                new_dimension_groups[subscript] = dimension_group
        for subscript, dimension_group in other.dimension_groups.items():
            if not subscript in new_dimension_groups:
                new_dimension_groups[subscript] = dimension_group
        return Dimension(new_dimension_groups)

    def __truediv__(self, other: "Dimension") -> "Dimension":
        """
        Divides two dimensions.
        
        {T/L} / {M/L^3} = {T*L^2/M}
        {T/L} / {M/L^3*dec(M)} = {T*L^2/M^2/dec(M)}
        {T/L*dec(L)} / {M/L^3*dec(M)} = X
        """
        return self * other.invert()
    
    def __pow__(self, exponent: float|int) -> "Dimension":
        new_dimension_groups: dict[str,DimensionGroup] = {}
        for subscript, dimension_group in self.dimension_groups.items():
            new_dimension_groups[subscript] = dimension_group ** exponent
        return Dimension(new_dimension_groups)
    
    def __invert__(self) -> "Dimension":
        new_dimension_groups: dict[str,DimensionGroup] = {}
        for subscript, dimension_group in self.dimension_groups.items():
            new_dimension_groups[subscript] = dimension_group.invert()
        return Dimension(new_dimension_groups)

    def invert(self) -> "Dimension":
        return ~self
    
    def __log__(self) -> "Dimension":
        new_dimension_groups: dict[str,DimensionGroup] = {}
        for subscript, dimension_group in self.dimension_groups.items():
            new_dimension_groups[subscript] = dimension_group.log()
        return Dimension(new_dimension_groups)
    
################################################################################
# Comparison operations
################################################################################

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return False
        return self.dimension_groups == other.dimension_groups

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, Dimension):
            return True
        return not self.__eq__(other)
    
    def compatible_to(self, other: "Dimension") -> bool:
        if self.dimension_groups.keys() != other.dimension_groups.keys():
            return False
        for subscript, dimension_group in self.dimension_groups.items():
            if not dimension_group.compatible_to(other.dimension_groups[subscript]):
        return True

################################################################################
# Hash operations
################################################################################

    def __hash__(self) -> int:
        return hash(tuple(self.dimension_groups.values()))

################################################################################
# Properties
################################################################################

    @property
    def is_dimensionless(self) -> bool:
        return all(dimension_group.is_dimensionless for dimension_group in self.dimension_groups.values())

    @property
    def includes_angle(self) -> bool:
        for dimension_group in self.dimension_groups.values():
            if dimension_group.includes_angle:
                return True
        return False

    @property
    def includes_log_level(self) -> bool:
        for dimension_group in self.dimension_groups.values():
            if dimension_group.includes_log_level:
                return True
        return False
    
################################################################################
# String representation
################################################################################

    def __str__(self) -> str:
        return self.format_string()

    def format_string(self) -> str:
        """String representation with named dimension lookup and fallback to exponent notation."""
        # Import here to avoid circular imports
        from .utils.units.named_simple_dimensions import NamedSimpleDimension
        
        # Try to find a named dimension
        for named_dim in NamedSimpleDimension:
            if named_dim.dimension == self:
                return named_dim.name
        
        # Fallback to exponent representation
        if self.is_dimensionless:
            return "dimensionless"
        
        # Build string like "M^1 T^-2 L^1" for kg*m/s^2
        parts: list[str] = []
        labels = [
            BASE_DIMENSION_SYMBOLS.MASS.value[0],
            BASE_DIMENSION_SYMBOLS.TIME.value[0],
            BASE_DIMENSION_SYMBOLS.LENGTH.value[0],
            BASE_DIMENSION_SYMBOLS.CURRENT.value[0],
            BASE_DIMENSION_SYMBOLS.TEMPERATURE.value[0],
            BASE_DIMENSION_SYMBOLS.AMOUNT.value[0],
            BASE_DIMENSION_SYMBOLS.LUMINOUS.value[0],
            PSEUDO_DIMENSION_SYMBOLS.ANGLE.value[0],
            PSEUDO_DIMENSION_SYMBOLS.LOG_LEVEL.value[0]
        ]
        
        for i, exp in enumerate(self.dimension_exponents):
            if exp != 0:
                if exp == 1:
                    parts.append(labels[i])
                else:
                    parts.append(f"{labels[i]}^{exp}")
        
        # Add pseudo dimensions
        if self.pseudo_dimension_exponents[0] != 0:  # angle
            exp = self.pseudo_dimension_exponents[0]
            if exp == 1:
                parts.append("A")
            else:
                parts.append(f"A^{exp}")
                
        if self.pseudo_dimension_exponents[1] != 0:  # log level
            exp = self.pseudo_dimension_exponents[1]
            if exp == 1:
                parts.append("Log")
            else:
                parts.append(f"Log^{exp}")
        
        return " ".join(parts) if parts else "dimensionless"
    
    def __repr__(self) -> str:
        return self.format_string()
    
################################################################################
# Parsing
################################################################################

    @classmethod
    def parse_string(cls, string: str) -> "Dimension":
        """
        Parses a string into a named simple dimension.
        Examples:
        - "L" -> LENGTH
        - "L^2" -> AREA
        - "L^2/T" -> AREAL_FLOW_RATE
        - "M/L^3" -> DENSITY
        """
        raise NotImplementedError("Parsing of dimensions is not implemented")
    
################################################################################
# Canonical unit
################################################################################

    @property
    def canonical_unit(self) -> "Unit":
        """Convert this dimension to a canonical unit representation.
        
        This method converts a SimpleUnitDimension to its canonical SimpleUnit representation
        by following a two-step process:
        
        1. **Named Dimension Lookup**: First attempts to find an exact match in the 
           NamedSimpleUnitDimension enumeration. If found, returns the pre-defined 
           canonical unit (e.g., "N" for force, "J" for energy, "V" for voltage).
           
        2. **Base Dimension Decomposition**: If no named match exists, decomposes the 
           dimension into base SI units using the fundamental dimensions (kg, m, s, A, K, mol, cd, rad).
           Constructs a unit string by combining base units with appropriate exponents.
        
        Returns:
            SimpleUnit: The canonical unit representation with factor=1.0 and offset=0.
            For dimensionless quantities, returns an empty SimpleUnit with string representation "".
        
        Examples:
            # Named dimensions (preferred when available)
            SimpleUnitDimension([1, -2, 1, 0, 0, 0, 0], [0, 0]).canonical_unit  # Returns "N" (force)
            SimpleUnitDimension([1, -2, 2, 0, 0, 0, 0], [0, 0]).canonical_unit  # Returns "J" (energy)
            SimpleUnitDimension([1, -3, 2, -1, 0, 0, 0], [0, 0]).canonical_unit  # Returns "V" (voltage)
            
            # Base dimension decomposition (when no named unit exists)
            SimpleUnitDimension([2, -1, 1, -1, 0, 0, 0], [0, 0]).canonical_unit  # Returns "kg^2*m/s*A"
            SimpleUnitDimension([0, -1, 1, 0, 0, 0, 0], [0, 0]).canonical_unit   # Returns "m/s" (velocity)
            SimpleUnitDimension([1, 0, 2, 0, 0, 0, 0], [0, 0]).canonical_unit    # Returns "kg*m^2"
            
            # Special cases
            SimpleUnitDimension([0, 0, 0, 0, 0, 0, 0], [0, 0]).canonical_unit    # Returns "" (dimensionless)
            SimpleUnitDimension([0, 0, 0, 0, 0, 0, 0], [0, 1]).canonical_unit    # Returns "rad" (angle)
        
        Notes:
            - The method uses lazy imports to avoid circular dependencies
            - All returned units have factor=1.0 and offset=0 (canonical form)
            - Results are cached for performance (expensive base dimension decomposition)
            - Handles both regular dimensional exponents and pseudo-dimensional exponents (angle, log-level)
            - Floating-point precision is handled with 1e-10 tolerance for zero checks
            - Complex dimensions are decomposed using linear algebra on the base dimension matrix
            - Dimensionless quantities return an empty SimpleUnit with string representation ""
        
        Raises:
            ValueError: If the dimension cannot be decomposed into base dimensions
                       (should not occur with valid SimpleUnitDimension instances)
        """
        # Check cache first
        if self in _CANONICAL_UNIT_CACHE:
            return _CANONICAL_UNIT_CACHE[self]
            
        # Import here to avoid circular imports
        from .unit import Unit
        from .utils.units.named_simple_dimensions import NamedSimpleDimension, DimenensionTag
        
        # If this is a dimensionless quantity, return empty unit
        if self.is_dimensionless:
            result = Unit.dimensionless_unit()
            _CANONICAL_UNIT_CACHE[self] = result
            return result
        
        # First check if there's a named dimension that matches exactly
        named_dimension: NamedSimpleDimension | None = NamedSimpleDimension.find_named_unit_dimension(self)
        if named_dimension is not None:
            result = named_dimension.canonical_unit
            _CANONICAL_UNIT_CACHE[self] = result
            return result
        
        # If no exact match, build from dimensions
        dimensions = NamedSimpleDimension.unit_dimensions_by_tags({DimenensionTag.BASE_DIMENSION})
        
        unit_string_parts: list[str] = []
        current_dimension_exponents: list[float] = list(self.dimension_exponents)
        current_pseudo_dimension_exponents: list[int] = list(self.pseudo_dimension_exponents)

        # Build unit string from base dimensions
        for base_dimension in dimensions:
            # Find the first non-zero exponent in the current remaining dimension
            dimension_match_index = -1
            pseudo_dimension_match_index = -1
            
            # Check for dimension exponent matches
            for i, (current_exp, base_exp) in enumerate(zip(current_dimension_exponents, base_dimension.dimension.dimension_exponents)):
                if current_exp != 0 and base_exp != 0:
                    dimension_match_index = i
                    break
                    
            # Check for pseudo dimension exponent matches
            for i, (current_exp, base_exp) in enumerate(zip(current_pseudo_dimension_exponents, base_dimension.dimension.pseudo_dimension_exponents)):
                if current_exp != 0 and base_exp != 0:
                    pseudo_dimension_match_index = i
                    break
            
            # Determine unit exponent based on matches
            unit_exponent = 0.0
            if dimension_match_index != -1 and pseudo_dimension_match_index == -1:
                # Only dimension match
                unit_exponent = current_dimension_exponents[dimension_match_index] / base_dimension.dimension.dimension_exponents[dimension_match_index]
            elif dimension_match_index == -1 and pseudo_dimension_match_index != -1:
                # Only pseudo dimension match
                unit_exponent = current_pseudo_dimension_exponents[pseudo_dimension_match_index] / base_dimension.dimension.pseudo_dimension_exponents[pseudo_dimension_match_index]
            elif dimension_match_index != -1 and pseudo_dimension_match_index != -1:
                # Both matches - they should be consistent
                dim_ratio = current_dimension_exponents[dimension_match_index] / base_dimension.dimension.dimension_exponents[dimension_match_index]
                pseudo_ratio = current_pseudo_dimension_exponents[pseudo_dimension_match_index] / base_dimension.dimension.pseudo_dimension_exponents[pseudo_dimension_match_index]
                if abs(dim_ratio - pseudo_ratio) > 1e-10:  # Allow for floating point errors
                    raise ValueError(f"Inconsistent ratios for base dimension {base_dimension}: {dim_ratio} vs {pseudo_ratio}")
                unit_exponent = dim_ratio
            else:
                # No match - this base dimension is not needed
                continue
            
            # Add unit string part if exponent is non-zero
            if abs(unit_exponent) > 1e-10:
                base_unit_string = base_dimension.value.canonical_unit_string
                if abs(unit_exponent - 1.0) < 1e-10:
                    unit_string_parts.append(base_unit_string)
                elif abs(unit_exponent + 1.0) < 1e-10:
                    unit_string_parts.append(f"/{base_unit_string}")
                else:
                    if unit_exponent > 0:
                        unit_string_parts.append(f"{base_unit_string}^{unit_exponent}")
                    else:
                        unit_string_parts.append(f"/{base_unit_string}^{abs(unit_exponent)}")
                
                # Subtract this base dimension contribution from current exponents
                for i, base_exp in enumerate(base_dimension.dimension.dimension_exponents):
                    current_dimension_exponents[i] -= float(base_exp * unit_exponent)
                    
                for i, base_exp in enumerate(base_dimension.dimension.pseudo_dimension_exponents):
                    current_pseudo_dimension_exponents[i] -= int(base_exp * unit_exponent)
            
            # Check if we've accounted for all dimensions
            if all(abs(exp) < 1e-10 for exp in current_dimension_exponents) and all(abs(exp) < 1e-10 for exp in current_pseudo_dimension_exponents):
                break
        
        # Check if we successfully decomposed the dimension
        if not (all(abs(exp) < 1e-10 for exp in current_dimension_exponents) and all(abs(exp) < 1e-10 for exp in current_pseudo_dimension_exponents)):
            raise ValueError(f"Unable to decompose dimension {self} into base dimensions. Remaining: dim={current_dimension_exponents}, pseudo={current_pseudo_dimension_exponents}")
        
        # Create unit string and parse it
        if not unit_string_parts:
            result = Unit.dimensionless_unit()
            _CANONICAL_UNIT_CACHE[self] = result
            return result
        
        unit_string = "*".join(unit_string_parts)
        result = Unit.parse_string(unit_string)
        _CANONICAL_UNIT_CACHE[self] = result
        return result

################################################################################
# JSON serialization
################################################################################

    def to_json(self) -> dict[str, Any]:
        return self.canonical_unit.to_json()

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Dimension":
        from .unit import Unit
        unit = Unit.from_json(data)
        return cls(unit.dimension.dimension_exponents, unit.dimension.pseudo_dimension_exponents)
    
################################################################################
# HDF5 serialization
################################################################################

    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        self.canonical_unit.to_hdf5(hdf5_group)

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "Dimension":
        from .unit import Unit
        unit = Unit.from_hdf5(hdf5_group)
        return cls(unit.dimension.dimension_exponents, unit.dimension.pseudo_dimension_exponents)
    
    @classmethod
    def dimensionless_dimension(cls) -> "Dimension":
        return DIMENSIONLESS_DIMENSION

DIMENSIONLESS_DIMENSION: "Dimension" = Dimension.create([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0])
LOG_LEVEL_DIMENSION: "Dimension" = Dimension.create([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 1])
ANGLE_DIMENSION: "Dimension" = Dimension.create([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1, 0])

@dataclass(frozen=True, slots=True)
class Subscripted_Canonical_Dimension(BaseDimension[Any, Any]):
    subscript_dimension_exponents: List[Tuple[float, float, float, float, float, float, float]]
    log_lin_exp_specified: List[Tuple[int, int, int]]
    substripts: List[str]

