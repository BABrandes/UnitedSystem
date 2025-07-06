from .utils import JSONable, HDF5able
from typing import Union, Tuple, Optional, TYPE_CHECKING
from .named_dimensions import NamedDimension, DimensionExponents
from .units.simple.simple_dimension import SimpleDimension
import h5py
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .unit import Unit

@dataclass(frozen=True, slots=True, init=False)
class Dimension(JSONable, HDF5able):
    """User-friendly wrapper for SimpleUnitDimension with convenient constructors."""
    
########################################################

    # Fields

    _wrapped_dimension: SimpleDimension = field(init=False)

########################################################

    # Constructor

    def __init__(
            self,
            dimension: Optional[Union[NamedDimension, DimensionExponents, SimpleDimension]] = None,
            unit: Optional["Unit"] = None,
            mass: Optional[float] = None,
            time: Optional[float] = None,
            length: Optional[float] = None,
            current: Optional[float] = None,
            temperature: Optional[float] = None,
            amount: Optional[float] = None,
            luminous_intensity: Optional[float] = None,
            angle: Optional[int] = None,
            log_level: Optional[int] = None
            ):
        """
        Create a Dimension from various input formats.
        
        Examples:
            Dimension(NamedDimension.VELOCITY)                    # Named dimension
            Dimension(unit=Unit("m/s"))                          # From unit
            Dimension(DimensionExponents(length=1, time=-1))     # Individual exponents (velocity)
            Dimension(DimensionExponents(mass=1, length=1, time=-2)) # Individual exponents (force)
        """
        from .units.simple.simple_dimension import SimpleDimension
        from .units.named_simple_dimensions import NamedSimpleDimension

        # Count specification methods
        specified_methods = sum([
             dimension is not None,
             unit is not None,
             any(exp is not None for exp in [mass, time, length, current, temperature, amount, luminous_intensity, angle, log_level])
         ])
         
        if specified_methods > 1:
            raise ValueError("Can only specify dimension using one method: dimension, unit, or individual exponents")
        
        if unit is not None:
            # Extract dimension from unit
            object.__setattr__(self, "_wrapped_dimension", unit._wrapped_unit.dimension)
        else:
            match dimension:
                case NamedDimension():
                    if isinstance(dimension.value, NamedSimpleDimension):
                        object.__setattr__(self, "_wrapped_dimension", dimension.value.simple_dimension)
                    elif isinstance(dimension.value, DimensionExponents):
                        dim_exponents = [dimension.value.mass, dimension.value.time, dimension.value.length, dimension.value.current, dimension.value.temperature, dimension.value.amount, dimension.value.luminous_intensity]
                        pseudo_exponents = [dimension.value.angle, dimension.value.log_level]
                        object.__setattr__(self, "_wrapped_dimension", SimpleDimension.create(dim_exponents, pseudo_exponents))
                    else:
                        raise ValueError(f"Invalid dimension: {dimension}")

                case DimensionExponents():
                    dim_exponents = [dimension.mass, dimension.time, dimension.length, dimension.current, dimension.temperature, dimension.amount, dimension.luminous_intensity]
                    pseudo_exponents = [dimension.angle, dimension.log_level]
                    object.__setattr__(self, "_wrapped_dimension", SimpleDimension.create(dim_exponents, pseudo_exponents))
                
                case SimpleDimension():
                    object.__setattr__(self, "_wrapped_dimension", dimension)
                
                case None:
                    # Use individual exponents or default to dimensionless
                    dim_exponents = [
                        mass or 0.0,
                        time or 0.0,
                        length or 0.0,
                        current or 0.0,
                        temperature or 0.0,
                        amount or 0.0,
                        luminous_intensity or 0.0
                    ]
                    pseudo_exponents = [
                        int(angle or 0),
                        int(log_level or 0)
                    ]
                    object.__setattr__(self, "_wrapped_dimension", SimpleDimension.create(dim_exponents, pseudo_exponents))
                case _:
                    raise ValueError(f"Invalid dimension: {dimension}")

########################################################

    # Properties for easy access

    @property
    def dimension_exponents(self) -> Tuple[float, ...]:
        """Get the dimension exponents (mass, time, length, current, temperature, amount, luminous_intensity)."""
        return tuple(self._wrapped_dimension.dimension_exponents)
    
    @property
    def pseudo_dimension_exponents(self) -> Tuple[int, ...]:
        """Get the pseudo dimension exponents (angle, log_level).""" 
        return tuple(self._wrapped_dimension.pseudo_dimension_exponents)
    
    @property
    def canonical_unit(self) -> "Unit":
        """Get the canonical unit for this dimension."""
        from .unit import Unit
        return Unit(self._wrapped_dimension.canonical_unit)
    
    def is_dimensionless(self) -> bool:
        """Check if this is a dimensionless quantity."""
        return (all(exp == 0 for exp in self.dimension_exponents) and 
                all(exp == 0 for exp in self.pseudo_dimension_exponents))
    
########################################################

    # Dimension arithmetic

    def __add__(self, other: "Dimension") -> "Dimension":
        """Add dimensions (add exponents)."""
        return Dimension(self._wrapped_dimension + other._wrapped_dimension)

    def __sub__(self, other: "Dimension") -> "Dimension":
        """Subtract dimensions (subtract exponents)."""
        return Dimension(self._wrapped_dimension - other._wrapped_dimension)

    def __mul__(self, other: float) -> "Dimension":
        """Multiply dimensions (add exponents)."""
        return Dimension(self._wrapped_dimension * other)
    
    def __truediv__(self, other: float) -> "Dimension":
        """Divide dimensions (subtract exponents)."""
        return Dimension(self._wrapped_dimension / other)
    
########################################################

    # Comparison operations
    def __eq__(self, other: "Dimension") -> bool:
        """Check equality."""
        if not isinstance(other, Dimension):
            return False
        return self._wrapped_dimension == other._wrapped_dimension
    
    def __ne__(self, other: "Dimension") -> bool:
        """Check inequality."""
        return not self.__eq__(other)
    
    def compatible_with(self, other: "Dimension") -> bool:
        """Check if this dimension is compatible with another."""
        if not isinstance(other, Dimension):
            raise TypeError(f"Can only check compatibility with Dimension, got {type(other)}")
        return self._wrapped_dimension == other._wrapped_dimension
    
########################################################

    # String representation
    
    def __str__(self) -> str:
        """String representation."""
        return str(self._wrapped_dimension)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"Dimension({self._wrapped_dimension!r})"
    
########################################################

    # Serialization
    
    def to_json(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_dimension.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "Dimension":
        """Create from JSON dictionary."""
        return cls(SimpleDimension.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_dimension.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "Dimension":
        """Create from HDF5 group."""
        return cls(SimpleDimension.from_hdf5(hdf5_group))
