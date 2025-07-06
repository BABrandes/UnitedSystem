from .scalars.real_united_scalar import RealUnitedScalar
from typing import Union, TYPE_CHECKING, Optional
from .utils import JSONable, HDF5able
import h5py
from .dimension import Dimension
from .unit import Unit
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from .unit import Unit

@dataclass(frozen=True, slots=True, init=False)
class RealScalar(JSONable, HDF5able):

########################################################

    # Fields

    _wrapped_real_scalar: RealUnitedScalar = field(init=False, hash=False, repr=False, compare=False)
    _dimension: Dimension = field(init=False, hash=False, repr=False, compare=False)
    _display_unit: Unit = field(init=False, hash=False, repr=False, compare=False)

    """User-friendly wrapper for RealUnitedScalar with convenient constructors."""
    
########################################################

    # Constructor

    def __init__(self, value: Union[float, int, str, RealUnitedScalar], unit: Optional["Unit"] = None):
        """
        Create a RealScalar from various input formats.
        
        Examples:
            RealScalar("5 m")           # String with value and unit
            RealScalar(5, Unit("m"))    # Separate value and Unit object
            RealScalar(0.5)             # Dimensionless number
        """

        # Set the wrapped real scalar
        match value:
            case str():
                if unit is not None:
                    raise ValueError("Cannot specify unit when using string input")
                self._wrapped_real_scalar = RealUnitedScalar.parse_string(value)
                
            case float() | int():
                self._wrapped_real_scalar = RealUnitedScalar.create_from_value_and_unit(value, unit._wrapped_unit)

            case RealUnitedScalar():
                self._wrapped_real_scalar = value
        
        # Set the attributes from the wrapped real scalar
        self.canonical_value = self._wrapped_real_scalar.canonical_value
        self.dimension: Dimension = Dimension(self._wrapped_real_scalar.dimension)
        self.display_unit: Unit = Unit(self._wrapped_real_scalar.display_unit)

        self.__post_init__()

########################################################

    # Fields of the wrapped real scalar

    @property
    def canonical_value(self) -> float:
        return self._wrapped_real_scalar.canonical_value
    
    @property
    def dimension(self) -> Dimension:
        if self._dimension is None:
            self._dimension = Dimension(self._wrapped_real_scalar.dimension)
        return self._dimension
    
    @property
    def display_unit(self) -> Unit:
        if self._display_unit is None:
            self._display_unit = Unit(self._wrapped_real_scalar.display_unit)
        return self._display_unit
    
########################################################

    # Arithmetic operations

    def __add__(self, other: "RealScalar") -> "RealScalar":
        """Add two scalars."""
        return RealScalar(self._wrapped_real_scalar + other._wrapped_real_scalar)
    
    def __radd__(self, other: "RealScalar") -> "RealScalar":
        """Add two scalars (reverse)."""
        return other + self
    
    def __sub__(self, other: "RealScalar") -> "RealScalar":
        """Subtract two scalars."""
        return RealScalar(self._wrapped_real_scalar - other._wrapped_real_scalar)
    
    def __rsub__(self, other: "RealScalar") -> "RealScalar":
        """Subtract scalars (reverse)."""
        return other - self
    
    def __mul__(self, other: Union["RealScalar", float, int]) -> "RealScalar":
        """Multiply scalar by another scalar or number."""
        return RealScalar(self._wrapped_real_scalar * other._wrapped_real_scalar if isinstance(other, RealScalar) else self._wrapped_real_scalar * other)
    
    def __rmul__(self, other: Union[float, int]) -> "RealScalar":
        """Multiply number by scalar (reverse)."""
        return self * other
    
    def __truediv__(self, other: Union["RealScalar", float, int]) -> "RealScalar":
        """Divide scalar by another scalar or number."""
        return RealScalar(self._wrapped_real_scalar / other._wrapped_real_scalar if isinstance(other, RealScalar) else self._wrapped_real_scalar / other)
    
    def __rtruediv__(self, other: Union[float, int]) -> "RealScalar":
        """Divide number by scalar (reverse)."""
        return RealScalar(other / self._wrapped_real_scalar)
    
    def __pow__(self, exponent: float) -> "RealScalar":
        """Raise scalar to a power."""
        return RealScalar(self._wrapped_real_scalar ** exponent)

    ########################################################

    # Comparison operations (basic equality only for now)

    def __eq__(self, other: "RealScalar") -> bool:
        """Check equality."""
        return self._wrapped_real_scalar == other._wrapped_real_scalar
    
    def __ne__(self, other: "RealScalar") -> bool:
        """Check inequality."""
        return not self == other
    
    def __lt__(self, other: "RealScalar") -> bool:
        """Check if scalar is less than another scalar."""
        return self._wrapped_real_scalar < other._wrapped_real_scalar
    
    def __le__(self, other: "RealScalar") -> bool:
        """Check if scalar is less than or equal to another scalar."""
        return self._wrapped_real_scalar <= other._wrapped_real_scalar
    
    def __gt__(self, other: "RealScalar") -> bool:
        """Check if scalar is greater than another scalar."""
        return self._wrapped_real_scalar > other._wrapped_real_scalar
    
    def __ge__(self, other: "RealScalar") -> bool:
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
        return self._wrapped_real_scalar.is_infinite
    
    def is_finite(self) -> bool:
        """Check if the value is finite."""
        return self._wrapped_real_scalar.is_finite
    
    def compatible_with(self, other: "RealScalar") -> bool:
        """Check if this scalar is compatible (same dimension) with another."""
        return self._wrapped_real_scalar.compatible_to(other._wrapped_real_scalar)
    
    ########################################################

    # Serialization

    def to_json(self) -> dict:
        """Convert to dictionary for serialization."""
        return self._wrapped_real_scalar.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "RealScalar":
        return cls(RealUnitedScalar.from_json(data))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_real_scalar.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "RealScalar":
        return cls(RealUnitedScalar.from_hdf5(hdf5_group))