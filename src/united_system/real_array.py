from .arrays.real_united_array import RealUnitedArray
from typing import Union, Optional, TYPE_CHECKING, Any
from .utils import JSONable, HDF5able
import h5py
import numpy as np
from .dimension import Dimension
from .unit import Unit
from .real_scalar import RealScalar
from dataclasses import dataclass, field
from .scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from typing import Type
from .units.simple.simple_unit import SimpleUnit

if TYPE_CHECKING:
    from .unit import Unit

@dataclass(frozen=True, slots=True, init=False)
class RealArray(JSONable, HDF5able):
    """User-friendly wrapper for RealUnitedArray with convenient constructors."""

########################################################

    # Fields

    _wrapped_array: RealUnitedArray = field(init=False, hash=False, repr=False, compare=False)
    _dimension: Dimension = field(init=False, hash=False, repr=False, compare=False)
    _display_unit: Unit = field(init=False, hash=False, repr=False, compare=False)

########################################################

    # Constructor

    def __init__(self, values: Union[np.ndarray, list, RealUnitedArray], unit: Optional["Unit"] = None):
        """
        Create a RealArray from various input formats.
        
        Examples:
            RealArray([1, 2, 3], Unit("m"))        # List with unit
            RealArray(np.array([1, 2, 3]), Unit("m")) # Numpy array with unit
            RealArray([1, 2, 3])                   # Dimensionless array
            RealArray(existing_array)              # Wrap existing RealUnitedArray
        """

        match values:
            case RealUnitedArray():
                if unit is not None:
                    raise ValueError("Cannot specify unit when wrapping existing RealUnitedArray")
                self._wrapped_array = values
                
            case np.ndarray() | list():
                if unit is None:
                    # Create dimensionless array
                    if isinstance(values, list):
                        values = np.asarray(values, dtype=float)
                    self._wrapped_array = RealUnitedArray.create(values)
                else:
                    # Create array with specified unit
                    if isinstance(values, list):
                        values = np.asarray(values, dtype=float)
                    self._wrapped_array = RealUnitedArray.create(values, unit._wrapped_unit)
            
            case _:
                raise ValueError(f"Invalid values type: {type(values)}")

########################################################

    # Fields of the wrapped array

    @property
    def values(self) -> np.ndarray:
        """Get the raw numpy array values in the display unit."""
        if self._wrapped_array.display_unit is not None:
            return self._wrapped_array.in_unit(self._wrapped_array.display_unit)
        else:
            # Return canonical values if no display unit
            return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def canonical_values(self) -> np.ndarray:
        """Get the canonical numpy array values."""
        return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def shape(self) -> tuple:
        """Get the shape of the array."""
        return self._wrapped_array.shape
    
    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self._wrapped_array.size
    
    @property
    def dimension(self) -> Dimension:
        """Get the dimension of the array."""
        if self._dimension is None:
            self._dimension = Dimension(self._wrapped_array.dimension)
        return self._dimension
    
    @property
    def display_unit(self) -> Optional[Unit]:
        """Get the display unit of the array."""
        if self._display_unit is None and self._wrapped_array.display_unit is not None:
            self._display_unit = Unit(self._wrapped_array.display_unit)
        return self._display_unit

########################################################

    # Array access

    def __getitem__(self, index_or_slice: Union[int, slice]) -> Union[RealScalar, "RealArray"]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, RealUnitedScalar):
            return RealScalar(result)
        else:
            return RealArray(result)
    
    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)

########################################################

    # Arithmetic operations

    def __add__(self, other: Union["RealArray", RealScalar, float, int]) -> "RealArray":
        """Add arrays or scalars."""
        if isinstance(other, RealArray):
            return RealArray(self._wrapped_array + other._wrapped_array)
        elif isinstance(other, RealScalar):
            return RealArray(self._wrapped_array + other._wrapped_real_scalar)
        else:
            return RealArray(self._wrapped_array + other)
    
    def __radd__(self, other: Union[float, int, RealScalar]) -> "RealArray":
        """Reverse addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union["RealArray", RealScalar, float, int]) -> "RealArray":
        """Subtract arrays or scalars."""
        if isinstance(other, RealArray):
            return RealArray(self._wrapped_array - other._wrapped_array)
        elif isinstance(other, RealScalar):
            return RealArray(self._wrapped_array - other._wrapped_real_scalar)
        else:
            return RealArray(self._wrapped_array - other)
    
    def __rsub__(self, other: Union[float, int, RealScalar]) -> "RealArray":
        """Reverse subtraction."""
        if isinstance(other, RealScalar):
            return RealArray(other._wrapped_real_scalar - self._wrapped_array)
        else:
            return RealArray(other - self._wrapped_array)
    
    def __mul__(self, other: Union["RealArray", RealScalar, float, int]) -> "RealArray":
        """Multiply arrays or scalars."""
        if isinstance(other, RealArray):
            return RealArray(self._wrapped_array * other._wrapped_array)
        elif isinstance(other, RealScalar):
            return RealArray(self._wrapped_array * other._wrapped_real_scalar)
        else:
            return RealArray(self._wrapped_array * other)
    
    def __rmul__(self, other: Union[float, int, RealScalar]) -> "RealArray":
        """Reverse multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union["RealArray", RealScalar, float, int]) -> "RealArray":
        """Divide arrays or scalars."""
        if isinstance(other, RealArray):
            return RealArray(self._wrapped_array / other._wrapped_array)
        elif isinstance(other, RealScalar):
            return RealArray(self._wrapped_array / other._wrapped_real_scalar)
        else:
            return RealArray(self._wrapped_array / other)
    
    def __rtruediv__(self, other: Union[float, int, RealScalar]) -> "RealArray":
        """Reverse division."""
        if isinstance(other, RealScalar):
            return RealArray(other._wrapped_real_scalar / self._wrapped_array)
        else:
            return RealArray(other / self._wrapped_array)
    
    def __pow__(self, exponent: float) -> "RealArray":
        """Raise array to a power."""
        return RealArray(self._wrapped_array ** exponent)
    
    def __neg__(self) -> "RealArray":
        """Negate the array."""
        return RealArray(-self._wrapped_array)
    
    def __abs__(self) -> "RealArray":
        """Get absolute value of the array."""
        return RealArray(abs(self._wrapped_array))

########################################################

    # Array reduction methods

    def sum(self) -> RealScalar:
        """Sum of all elements."""
        return RealScalar(self._wrapped_array.sum())
    
    def mean(self) -> RealScalar:
        """Mean of all elements."""
        return RealScalar(self._wrapped_array.mean())
    
    def max(self) -> RealScalar:
        """Maximum value."""
        return RealScalar(self._wrapped_array.max())
    
    def min(self) -> RealScalar:
        """Minimum value."""
        return RealScalar(self._wrapped_array.min())
    
    def std(self) -> RealScalar:
        """Standard deviation."""
        return RealScalar(self._wrapped_array.std())
    
    def var(self) -> RealScalar:
        """Variance."""
        return RealScalar(self._wrapped_array.var())

########################################################

    # Unit conversion

    def in_unit(self, unit: Unit) -> np.ndarray:
        """Convert the array to a specific unit and return numpy array."""
        return self._wrapped_array.in_unit(unit._wrapped_unit)
    
    def with_unit(self, unit: Unit) -> "RealArray":
        """Return a new array with the specified display unit."""
        new_wrapped = self._wrapped_array.with_display_unit(unit._wrapped_unit)
        return RealArray(new_wrapped)

########################################################

    # String representation

    def __str__(self) -> str:
        """Simple string representation."""
        return str(self._wrapped_array)
    
    def __repr__(self) -> str:
        """Detailed string representation for debugging."""
        return f"RealArray({self._wrapped_array!r})"
    
    def format(self, unit: Optional[Unit] = None, decimals: int = 3) -> str:
        """
        Format the array with optional unit specification.
        
        Args:
            unit: Unit to display the values in. If None, uses auto-suggestion.
            decimals: Number of decimal places to show.
            
        Returns:
            Formatted string representation.
        """
        wrapped_unit = unit._wrapped_unit if unit is not None else None
        return self._wrapped_array.format(wrapped_unit, decimals)

########################################################

    # Comparison and utility operations

    def compatible_with(self, other: "RealArray") -> bool:
        """Check if this array is compatible (same dimension) with another."""
        return self._wrapped_array.compatible_to(other._wrapped_array)

########################################################

    # Factory methods

    @classmethod
    def from_values_and_unit(cls, values: Union[np.ndarray, list], unit: Unit) -> "RealArray":
        """Create a RealArray from values and a unit."""
        return cls(values, unit)
    
    @classmethod
    def zeros(cls, shape: Union[int, tuple], unit: Optional[Unit] = None) -> "RealArray":
        """Create an array of zeros."""
        return cls(np.zeros(shape), unit)
    
    @classmethod
    def ones(cls, shape: Union[int, tuple], unit: Optional[Unit] = None) -> "RealArray":
        """Create an array of ones."""
        return cls(np.ones(shape), unit)
    
    @classmethod
    def dimensionless(cls, values: Union[np.ndarray, list]) -> "RealArray":
        """Create a dimensionless array."""
        return cls(values)

########################################################

    # Serialization

    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict) -> "RealArray":
        """Create from JSON dictionary."""
        type_parameters: Type = {"unit_type": SimpleUnit}
        return cls(RealUnitedArray.from_json(data, type_parameters))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, type_parameters: Type) -> "RealArray":
        """Create from HDF5 group."""
        type_parameters: Type = {"unit_type": SimpleUnit}
        return cls(RealUnitedArray.from_hdf5(hdf5_group, type_parameters))