from ..united_system.real_united_array import RealUnitedArray as InternalRealUnitedArray
from typing import Union, Optional, TYPE_CHECKING, Any, overload, Sized
from ..united_system.utils.general import JSONable, HDF5able
import h5py
import numpy as np
from .dimension import Dimension
from .unit import Unit
from ..united_system.units.united import United
from .real_united_scalar_depreciated import RealUnitedScalar
from dataclasses import dataclass, field
from ..united_system.real_united_scalar import RealUnitedScalar
from ..united_system.dimension import Dimension
from typing import Type
from ..united_system.unit import Unit
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .unit import Unit
    from ..united_system.real_united_scalar import FLOAT_TYPE

@dataclass(frozen=True, slots=True, init=False)
class RealUnitedArray(JSONable, HDF5able, Sized, United[Dimension, Unit]):
    """User-friendly wrapper for RealUnitedArray with convenient constructors."""

########################################################

    # Fields

    _wrapped_array: InternalRealUnitedArray = field(init=False, hash=False, repr=False, compare=False)
    _dimension: Optional[Dimension] = field(init=False, hash=False, repr=False, compare=False)
    _display_unit: Optional[Unit] = field(init=False, hash=False, repr=False, compare=False)

########################################################

    # Constructor

    def __init__(self, values: Union[np.ndarray[Any, Any], list[FLOAT_TYPE|int], InternalRealUnitedArray], unit: Optional[Unit] = None):
        """
        Create a RealArray from various input formats.
        
        Examples:
            RealArray([1, 2, 3], Unit("m"))        # List with unit
            RealArray(np.array([1, 2, 3]), Unit("m")) # Numpy array with unit
            RealArray([1, 2, 3])                   # Dimensionless array
            RealArray(existing_array)              # Wrap existing RealUnitedArray
        """

        match values:
            case InternalRealUnitedArray():
                if unit is not None:
                    raise ValueError("Cannot specify unit when wrapping existing RealUnitedArray")
                object.__setattr__(self, "_wrapped_array", values)
                
            case np.ndarray() | list():
                if unit is None:
                    # Create dimensionless array
                    if isinstance(values, list):
                        values = np.asarray(values, dtype=float)
                    # Create a dimensionless SimpleDimension (all zeros)
                    dimensionless_dim = Dimension.create([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0])
                    object.__setattr__(self, "_wrapped_array", InternalRealUnitedArray(values, dimensionless_dim))
                else:
                    # Create array with specified unit
                    if isinstance(values, list):
                        values = np.asarray(values, dtype=float)
                    # Convert values to canonical units
                    canonical_values = unit._wrapped_unit.to_canonical_value(values) # type: ignore
                    object.__setattr__(self, "_wrapped_array", InternalRealUnitedArray(canonical_values, unit.dimension._wrapped_dimension, unit._wrapped_unit)) # type: ignore

        # Initialize cached fields
        object.__setattr__(self, "_dimension", None)
        object.__setattr__(self, "_display_unit", None)

    @classmethod
    def create_from_canonical_values(cls, canonical_values: np.ndarray, display_unit_or_dimension: Unit|Dimension) -> "RealUnitedArray":
        """Create a RealUnitedArray from canonical values and a unit."""
        if isinstance(display_unit_or_dimension, Unit):
            display_unit: Unit = display_unit_or_dimension
            return cls(InternalRealUnitedArray(canonical_values, display_unit.dimension._wrapped_dimension), display_unit._wrapped_unit) # type: ignore
        else:
            dimension: Dimension = display_unit_or_dimension
            return cls(InternalRealUnitedArray(canonical_values, dimension._wrapped_dimension), None) # type: ignore

########################################################

    # Fields of the wrapped array

    @property
    def values(self) -> NDArray[np.float64|np.float32|np.float16]:
        """Get the raw numpy array values in the display unit."""
        if self._display_unit is not None:
            # Convert from canonical to display unit
            return self._display_unit.from_canonical_value(self._wrapped_array.canonical_np_array)
        else:
            # Return canonical values if no display unit
            return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def canonical_values(self) -> NDArray[np.float64|np.float32|np.float16]:
        """Get the canonical numpy array values."""
        return self._wrapped_array.canonical_np_array.copy()
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        shape: tuple[int, ...] = self._wrapped_array.shape # type: ignore
        return shape
    
    @property
    def size(self) -> int:
        """Get the size of the array."""
        return self._wrapped_array.size
    
    @property
    def dimension(self) -> Dimension:
        """Get the dimension of the array."""
        if self._dimension is None:
            object.__setattr__(self, "_dimension", Dimension(self._wrapped_array.dimension))
        if self._dimension is None:
            raise ValueError("Dimension is None")
        return self._dimension
    
    @property
    def display_unit(self) -> Unit:
        """Get the display unit of the array."""
        if self._display_unit is None:
            object.__setattr__(self, "_display_unit", Unit(self._wrapped_array.display_unit))
        if self._display_unit is None:
            raise ValueError("Display unit is None")
        return self._display_unit

########################################################

    # Array access

    @overload
    def __getitem__(self, index_or_slice: int) -> FLOAT_TYPE:
        ...
    @overload
    def __getitem__(self, index_or_slice: slice) -> NDArray[np.float64|np.float32|np.float16]:
        ...
    def __getitem__(self, index_or_slice: int|slice) -> FLOAT_TYPE|NDArray[np.float64|np.float32|np.float16]:
        """Get array element or slice."""
        result = self._wrapped_array[index_or_slice]
        if isinstance(result, np.ndarray):
            return result.astype(np.float64|np.float32|np.float16)
        elif isinstance(result, float):
            return result
        else:
            raise ValueError(f"Invalid result type: {type(result)}")

    def __len__(self) -> int:
        """Get the length of the array."""
        return len(self._wrapped_array)

    def get_scalar(self, index: int) -> RealUnitedScalar:
        """Get a scalar value at the given index."""
        return RealUnitedScalar(self._wrapped_array[index], self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore

    def get_array(self, slice: slice) -> "RealUnitedArray":
        """Get a slice of the array."""
        return RealUnitedArray(self._wrapped_array[slice], self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore

########################################################

    # Arithmetic operations

    def __add__(self, other: Union["RealUnitedArray", RealUnitedScalar]) -> "RealUnitedArray":
        """Add arrays or scalars."""
        if isinstance(other, RealUnitedArray):
            result = self._wrapped_array + other._wrapped_array
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        else:
            result = self._wrapped_array + other
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __radd__(self, other: RealUnitedScalar) -> "RealUnitedArray":
        """Reverse addition."""
        return self.__add__(other)
    
    def __sub__(self, other: Union["RealUnitedArray", RealUnitedScalar]) -> "RealUnitedArray":
        """Subtract arrays or scalars."""
        if isinstance(other, RealUnitedArray):
            result = self._wrapped_array - other._wrapped_array
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        else:
            result = self._wrapped_array - other
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __rsub__(self, other: RealUnitedScalar) -> "RealUnitedArray":
        """Reverse subtraction."""
        return self.__sub__(other)
    
    def __mul__(self, other: Union["RealUnitedArray", RealUnitedScalar, float, int]) -> "RealUnitedArray":
        """Multiply arrays or scalars."""
        if isinstance(other, RealUnitedArray):
            result = self._wrapped_array * other._wrapped_array
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        elif isinstance(other, RealUnitedScalar):
            result = self._wrapped_array * other
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        else:
            result = self._wrapped_array * other
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __rmul__(self, other: Union[float, int, RealUnitedScalar]) -> "RealUnitedArray":
        """Reverse multiplication."""
        return self.__mul__(other)
    
    def __truediv__(self, other: Union["RealUnitedArray", RealUnitedScalar, float, int]) -> "RealUnitedArray":
        """Divide arrays or scalars."""
        if isinstance(other, RealUnitedArray):
            result = self._wrapped_array / other._wrapped_array
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        elif isinstance(other, RealUnitedScalar):
            result = self._wrapped_array / other
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        else:
            result = self._wrapped_array / other
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __rtruediv__(self, other: Union[float, int, RealUnitedScalar]) -> "RealUnitedArray":
        """Reverse division."""
        if isinstance(other, RealUnitedScalar):
            result = other / self._wrapped_array
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
        else:
            result = other / self._wrapped_array
            return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __pow__(self, exponent: float) -> "RealUnitedArray":
        """Raise array to a power."""
        result = self._wrapped_array ** exponent
        return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __neg__(self) -> "RealUnitedArray":
        """Negate the array."""
        result = -self._wrapped_array
        return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def __abs__(self) -> "RealUnitedArray":
        """Get absolute value of the array.""" 
        result = abs(self._wrapped_array)
        return RealUnitedArray(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore

########################################################

    # Array reduction methods

    def sum(self) -> RealUnitedScalar:
        """Sum of all elements."""
        result = self._wrapped_array.sum()
        return RealUnitedScalar(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def mean(self) -> RealUnitedScalar:
        """Mean of all elements."""
        result = self._wrapped_array.mean()
        return RealUnitedScalar(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def max(self) -> RealUnitedScalar:
        """Maximum value."""
        result = self._wrapped_array.max()
        return RealUnitedScalar(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def min(self) -> RealUnitedScalar:
        """Minimum value."""
        result = self._wrapped_array.min()
        return RealUnitedScalar(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def std(self) -> RealUnitedScalar:
        """Standard deviation."""
        result = self._wrapped_array.std()
        return RealUnitedScalar(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore
    
    def var(self) -> RealUnitedScalar:
        """Variance."""
        result = self._wrapped_array.var()
        return RealUnitedScalar(result, self._wrapped_array.dimension, self._wrapped_array._display_unit) # type: ignore

########################################################

    # Unit conversion

    def in_unit(self, unit: Unit) -> np.ndarray:
        """Convert the array to a specific unit and return numpy array."""
        result = self._wrapped_array.in_unit(unit._wrapped_unit) # type: ignore
        return result
    
    def with_unit(self, unit: Unit) -> "RealUnitedArray":
        """Return a new array with the specified display unit."""
        new_wrapped = self._wrapped_array.with_display_unit(unit._wrapped_unit) # type: ignore
        return RealUnitedArray(new_wrapped, self._wrapped_array.dimension, unit._wrapped_unit) # type: ignore

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
        wrapped_unit: Optional[Unit] = unit._wrapped_unit if unit is not None else None # type: ignore
        return self._wrapped_array.format(wrapped_unit, decimals)

########################################################

    # Comparison and utility operations

    def compatible_with(self, other: "RealUnitedArray") -> bool:
        """Check if this array is compatible (same dimension) with another."""
        return self._wrapped_array.compatible_to(other._wrapped_array)

########################################################

    # Factory methods

    @classmethod
    def from_values_and_unit(cls, values: Union[np.ndarray[Any, Any], list[FLOAT_TYPE|int]], unit: Unit) -> "RealUnitedArray":
        """Create a RealArray from values and a unit."""
        return cls(values, unit)
    
    @classmethod
    def zeros(cls, number_of_elements: int, unit: Optional[Unit] = None) -> "RealUnitedArray":
        """Create an array of zeros."""
        return cls(np.zeros(number_of_elements), unit)
    
    @classmethod
    def ones(cls, number_of_elements: int, unit: Optional[Unit] = None) -> "RealUnitedArray":
        """Create an array of ones."""
        return cls(np.ones(number_of_elements), unit)
    
    @classmethod
    def dimensionless(cls, values: Union[np.ndarray[Any, Any], list[FLOAT_TYPE|int]]) -> "RealUnitedArray":
        """Create a dimensionless array."""
        return cls(values)

########################################################

    # Serialization

    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self._wrapped_array.to_json()
    
    @classmethod
    def from_json(cls, data: dict[str, Any], **_: Type[Any]) -> "RealUnitedArray":
        """Create from JSON dictionary."""
        return cls(InternalRealUnitedArray.from_json(data, unit_type=Unit))
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Convert to HDF5 group for serialization."""
        self._wrapped_array.to_hdf5(hdf5_group)
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **_: Type[Any]) -> "RealUnitedArray":
        """Create from HDF5 group."""
        return cls(InternalRealUnitedArray.from_hdf5(hdf5_group, unit_type=Unit))