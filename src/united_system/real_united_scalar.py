from .scalars.real_united_scalar import RealUnitedScalar as InternalRealUnitedScalar
from typing import Union, TYPE_CHECKING, Optional
from .utils import JSONable, HDF5able
import h5py
from .dimension import Dimension
from .unit import Unit
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .unit import Unit

@dataclass(frozen=True, slots=True, init=False)
class RealUnitedScalar(JSONable, HDF5able):

########################################################

    # Fields

    _wrapped_real_scalar: InternalRealUnitedScalar = field(init=False, hash=False, repr=False, compare=False)
    _dimension: Dimension = field(init=False, hash=False, repr=False, compare=False)
    _display_unit: Unit = field(init=False, hash=False, repr=False, compare=False)

    """User-friendly wrapper for RealUnitedScalar with convenient constructors."""
    
########################################################

    # Constructor

    def __init__(self, value: Union[float, int, str, InternalRealUnitedScalar], unit: Optional["Unit"] = None):
        """
        Create a RealUnitedScalar from various input formats.
        
        Examples:
            RealUnitedScalar("5 cm")           # String with value and unit
            RealUnitedScalar(5, Unit("cm"))    # Separate value and Unit object
            RealUnitedScalar(0.5)             # Dimensionless number
        """

        # Set the wrapped real scalar
        match value:
            case str():
                if unit is not None:
                    raise ValueError("Cannot specify unit when using string input")
                object.__setattr__(self, "_wrapped_real_scalar", InternalRealUnitedScalar.parse_string(value))
                
            case float() | int():
                if unit is None:
                    # Create dimensionless
                    object.__setattr__(self, "_wrapped_real_scalar", InternalRealUnitedScalar.create_dimensionless(float(value)))
                else:
                    object.__setattr__(self, "_wrapped_real_scalar", InternalRealUnitedScalar.create_from_value_and_unit(value, unit._wrapped_unit))

            case InternalRealUnitedScalar():
                object.__setattr__(self, "_wrapped_real_scalar", value)
        
        # Set the cached attributes
        object.__setattr__(self, "_dimension", None)
        object.__setattr__(self, "_display_unit", None)

    @classmethod
    def create_from_canonical_value(cls, canonical_value: float, display_unit_or_dimension: Unit|Dimension) -> "RealUnitedScalar":
        """Create a RealUnitedScalar from a canonical value and a unit."""
        if isinstance(display_unit_or_dimension, Unit):
            display_unit: Unit = display_unit_or_dimension
            return cls(InternalRealUnitedScalar(canonical_value, display_unit.dimension._wrapped_dimension), display_unit._wrapped_unit)
        elif isinstance(display_unit_or_dimension, Dimension):
            dimension: Dimension = display_unit_or_dimension
            return cls(InternalRealUnitedScalar(canonical_value, dimension._wrapped_dimension), None)
        else:
            raise ValueError(f"Invalid display unit or dimension: {display_unit_or_dimension}")

########################################################

    # Fields of the wrapped real scalar

    @property
    def canonical_value(self) -> float:
        return self._wrapped_real_scalar.canonical_value
    
    @property
    def dimension(self) -> Dimension:
        if self._dimension is None:
            object.__setattr__(self, "_dimension", Dimension(self._wrapped_real_scalar.dimension))
        return self._dimension
    
    @property
    def display_unit(self) -> Optional[Unit]:
        if self._display_unit is None and self._wrapped_real_scalar.display_unit is not None:
            object.__setattr__(self, "_display_unit", Unit(self._wrapped_real_scalar.display_unit))
        return self._display_unit
    
########################################################

    # Arithmetic operations

    def __add__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Add two scalars."""
        return RealUnitedScalar(self._wrapped_real_scalar + other._wrapped_real_scalar)
    
    def __radd__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Add two scalars (reverse)."""
        return other + self
    
    def __sub__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Subtract two scalars."""
        return RealUnitedScalar(self._wrapped_real_scalar - other._wrapped_real_scalar)
    
    def __rsub__(self, other: "RealUnitedScalar") -> "RealUnitedScalar":
        """Subtract scalars (reverse)."""
        return other - self
    
    def __mul__(self, other: Union["RealUnitedScalar", float, int]) -> "RealUnitedScalar":
        """Multiply scalar by another scalar or number."""
        return RealUnitedScalar(self._wrapped_real_scalar * other._wrapped_real_scalar if isinstance(other, RealUnitedScalar) else self._wrapped_real_scalar * other)
    
    def __rmul__(self, other: Union[float, int]) -> "RealUnitedScalar":
        """Multiply number by scalar (reverse)."""
        return self * other
    
    def __truediv__(self, other: Union["RealUnitedScalar", float, int]) -> "RealUnitedScalar":
        """Divide scalar by another scalar or number."""
        return RealUnitedScalar(self._wrapped_real_scalar / other._wrapped_real_scalar if isinstance(other, RealUnitedScalar) else self._wrapped_real_scalar / other)
    
    def __rtruediv__(self, other: Union[float, int]) -> "RealUnitedScalar":
        """Divide number by scalar (reverse)."""
        return RealUnitedScalar(other / self._wrapped_real_scalar)
    
    def __pow__(self, exponent: float) -> "RealUnitedScalar":
        """Raise scalar to a power."""
        return RealUnitedScalar(self._wrapped_real_scalar ** exponent)

    ########################################################

    # Comparison operations (basic equality only for now)

    def __eq__(self, other: "RealUnitedScalar") -> bool:
        """Check equality."""
        return self._wrapped_real_scalar == other._wrapped_real_scalar
    
    def __ne__(self, other: "RealUnitedScalar") -> bool:
        """Check inequality."""
        return not self == other
    
    def __lt__(self, other: "RealUnitedScalar") -> bool:
        """Check if scalar is less than another scalar."""
        return self._wrapped_real_scalar < other._wrapped_real_scalar
    
    def __le__(self, other: "RealUnitedScalar") -> bool:
        """Check if scalar is less than or equal to another scalar."""
        return self._wrapped_real_scalar <= other._wrapped_real_scalar
    
    def __gt__(self, other: "RealUnitedScalar") -> bool:
        """Check if scalar is greater than another scalar."""
        return self._wrapped_real_scalar > other._wrapped_real_scalar
    
    def __ge__(self, other: "RealUnitedScalar") -> bool:
        """Check if scalar is greater than or equal to another scalar."""
        return self._wrapped_real_scalar >= other._wrapped_real_scalar

    ########################################################

    # String representation

    def __str__(self) -> str:
        """Simple string representation."""
        return str(self._wrapped_real_scalar)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"RealScalar({self._wrapped_real_scalar!r})"
    
    def format(self, unit: Union["Unit", None] = None, decimals: int = 3) -> str:
        """
        Format the scalar with optional unit specification.
        
        Args:
            unit: Unit to display the value in. If None, uses auto-suggestion.
            decimals: Number of decimal places to show.
            
        Returns:
            Formatted string representation.
        """
        return self._wrapped_real_scalar.format(unit, decimals)
    
    ########################################################

    # Other operations

    def is_nan(self) -> bool:
        """Check if the value is NaN."""
        return self._wrapped_real_scalar.is_nan()
    
    def is_infinite(self) -> bool:
        """Check if the value is infinite."""
        return self._wrapped_real_scalar.is_infinite()
    
    def is_finite(self) -> bool:
        """Check if the value is finite."""
        return self._wrapped_real_scalar.is_finite()
    
    def compatible_with(self, other: "RealUnitedScalar") -> bool:
        """Check if this scalar is compatible (same dimension) with another."""
        return self._wrapped_real_scalar.compatible_to(other._wrapped_real_scalar)
    
    ########################################################

    # Serialization

    def to_json(self) -> dict:
        """Convert to dictionary for serialization."""
        return self._wrapped_real_scalar.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "RealUnitedScalar":
        return cls(InternalRealUnitedScalar.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_real_scalar.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "RealUnitedScalar":
        return cls(InternalRealUnitedScalar.from_hdf5(hdf5_group))