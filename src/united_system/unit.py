from .units.simple.simple_unit import SimpleUnit
from .dimension import Dimension
from .utils import JSONable, HDF5able
from typing import Union, TYPE_CHECKING
import h5py
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True, init=False)
class Unit(JSONable, HDF5able):
    """User-friendly wrapper for SimpleUnit with convenient constructors."""

########################################################

    # Fields

    _wrapped_unit: SimpleUnit = field(init=False, compare=False)
    _dimension: Dimension = field(init=False, hash=False, repr=False, compare=False)

########################################################

    # Constructor

    def __init__(self, unit_or_dimension: Union[str, SimpleUnit, Dimension]):
        """
        Create a Unit from various input formats.
        
        Examples:
            Unit("m/s")              # String parsing
            Unit("kg*m/s^2")         # Complex unit strings
            Unit("V")                # Simple units
            Unit(existing_unit)      # Wrap existing
        """
        
        match unit_or_dimension:
            case SimpleUnit():
                self._wrapped_unit = unit_or_dimension
                
            case str():
                self._wrapped_unit = SimpleUnit.parse_string(unit_or_dimension)

            case Dimension():
                self._wrapped_unit = SimpleUnit.parse_string(unit_or_dimension.canonical_unit.format_string(no_fraction=False))

########################################################

    # Fields of the wrapped unit

    @property
    def factor(self) -> float:
        """Get the conversion factor to canonical units."""
        return self._wrapped_unit.factor
    
    @property
    def offset(self) -> float:
        """Get the conversion offset to canonical units."""
        return self._wrapped_unit.offset
    
    @property
    def dimension(self) -> Dimension:
        """Get the dimension of this unit."""
        if self._dimension is None:
            self._dimension = Dimension(self._wrapped_unit.dimension)
        return self._dimension
    
########################################################

    # Other methods

    @property
    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless unit."""
        return self.dimension.is_dimensionless()
    
########################################################

    # Conversion methods

    def to_canonical_value(self, value: Union[float, int]) -> float:
        """Convert a value from this unit to canonical units."""
        return self._wrapped_unit.to_canonical_value(value)
    
    def from_canonical_value(self, canonical_value: Union[float, int]) -> float:
        """Convert a value from canonical units to this unit."""
        return self._wrapped_unit.from_canonical_value(canonical_value)

    def compatible_to(self, other: "Unit") -> bool:
        """Check if this unit is compatible with another."""
        return self._wrapped_unit.compatible_to(other._wrapped_unit)
    
    @classmethod
    def parse_string(cls, unit_str: str) -> "Unit":
        """Parse a unit string like "m/s", "kg*m/s^2", etc."""
        return cls(SimpleUnit.parse_string(unit_str))
    
########################################################

    # Unit arithmetic

    def __mul__(self, other: "Unit") -> "Unit":
        """Multiply units."""
        return Unit(self._wrapped_unit * other._wrapped_unit)
    
    def __truediv__(self, other: "Unit") -> "Unit":
        """Divide units."""
        return Unit(self._wrapped_unit / other._wrapped_unit)
    
    def __pow__(self, exponent: float|int) -> "Unit":
        """Raise unit to a power."""
        return Unit(self._wrapped_unit.pow(exponent))
    
########################################################

    # Comparison operations

    def __eq__(self, other: "Unit") -> bool:
        """Check equality."""
        return self._wrapped_unit.__eq__(other._wrapped_unit)
    
    def __ne__(self, other: "Unit") -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
########################################################
   
    # String representation
    
    def __str__(self) -> str:
        """String representation."""
        return str(self._wrapped_unit)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Unit({self._wrapped_unit!r})"
    
    def format_string(self, no_fraction: bool = False) -> str:
        """
        Format the unit as a string.
        
        Args:
            no_fraction: If True, avoid fraction notation (use negative exponents)
            
        Returns:
            Formatted unit string.
        """
        return self._wrapped_unit.format_string(no_fraction)
    
########################################################

    # Unit suggestion

    @classmethod
    def suggest_units(cls, dimension: Dimension, canonical_value: Union[float, int, None] = None, n: int = 10) -> list["Unit"]:
        """
        Suggest units for a given dimension and value.
        
        Args:
            dimension: The dimension to suggest units for
            value: The value to optimize suggestions for (optional)
            n: Number of suggestions to return
            
        Returns:
            List of suggested units, best first.
        """
        best_unit, alternatives = SimpleUnit.suggest_units(dimension, canonical_value, n=n)
        return [Unit(unit) for unit in [best_unit] + alternatives]
    
    @classmethod
    def suggest_best_unit(cls, dimension: Dimension, canonical_value: Union[float, int, None] = None) -> "Unit":
        """
        Suggest the best unit for a given dimension and value.
        
        Args:
            dimension: The dimension to suggest a unit for
            value: The value to optimize the suggestion for (optional)
            
        Returns:
            The best suggested unit.
        """
        return cls.suggest_units(dimension, canonical_value, n=1)[0]
    
########################################################

    # Factory methods

    @classmethod
    def dimensionless(cls) -> "Unit":
        """Create a dimensionless unit."""
        return cls("")
    
    @classmethod
    def meter(cls) -> "Unit":
        """Create a meter unit."""
        return cls("m")
    
    @classmethod
    def second(cls) -> "Unit":
        """Create a second unit."""
        return cls("s")
    
    @classmethod
    def kilogram(cls) -> "Unit":
        """Create a kilogram unit."""
        return cls("kg")
    
    @classmethod
    def ampere(cls) -> "Unit":
        """Create an ampere unit."""
        return cls("A")
    
    @classmethod
    def kelvin(cls) -> "Unit":
        """Create a kelvin unit."""
        return cls("K")
    
    @classmethod
    def mole(cls) -> "Unit":
        """Create a mole unit."""
        return cls("mol")
    
    @classmethod
    def candela(cls) -> "Unit":
        """Create a candela unit."""
        return cls("cd")
    
########################################################

    # Serialization

    def to_json(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_unit.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "Unit":
        """Create from JSON dictionary."""
        return cls(SimpleUnit.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_unit.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "Unit":
        """Create from HDF5 group."""
        return cls(SimpleUnit.from_hdf5(hdf5_group))
