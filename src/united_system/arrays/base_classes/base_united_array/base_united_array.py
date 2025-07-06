import numpy as np
from dataclasses import dataclass
from typing import overload, Any, TypeVar, TypeAlias, Generic, Union, Type, Iterator, Optional
from ....units.base_classes.base_dimension import BaseDimension
from ....units.base_classes.base_unit import BaseUnit
from ....scalars.united_scalar import UnitedScalar
from ....scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ....scalars.united_scalar import UnitedScalar
from ....units.united import United
from ..base_array import BaseArray
from ..protocol_numerical_array import ProtocolNumericalArray
from abc import ABC, abstractmethod
import h5py

PrimitiveType: TypeAlias = float|int|complex

UST = TypeVar("UST", bound=UnitedScalar)
UAT = TypeVar("UAT", bound=BaseArray)
UT = TypeVar("UT", bound=BaseUnit)
DT = TypeVar("DT", bound=BaseDimension)
PT = TypeVar("PT", bound=PrimitiveType)

@dataclass(frozen=True, slots=True)
class BaseUnitedArray(BaseArray[UST, UAT], United[DT, UT], ABC, Generic[UAT, UST, UT, DT, PT]):
    
    # Required field from BaseArray inheritance
    canonical_np_array: np.ndarray
    
    # Required fields from United inheritance
    dimension: DT
    display_unit: Optional[UT] = None

    @classmethod
    def create(cls, canonical_np_array: np.ndarray, dimension_or_display_unit: Union[DT, UT]) -> "BaseUnitedArray":
        match dimension_or_display_unit:
            case BaseDimension():
                dimension: DT = dimension_or_display_unit
                display_unit: UT = None
            case BaseUnit():
                dimension: DT = dimension_or_display_unit.dimension
                display_unit: UT = dimension_or_display_unit
            case _:
                raise ValueError(f"Invalid dimension or display unit: {dimension_or_display_unit}")
        
        return cls(canonical_np_array, dimension, display_unit)

    def set_display_unit(self, display_unit: UT) -> None:
        if display_unit.compatible_to(self.dimension):
            raise ValueError(f"The display unit {display_unit} is not compatible with the canonical dimension {self.dimension}.")
        self.__setattr__("_display_unit", display_unit)

    @abstractmethod
    def get_as_united_value(self, index: int) -> UST:
        """Get a single element as a united scalar."""
        pass

    @abstractmethod
    def _get_scalar_from_value(self, value: PT) -> UST:
        """Create a scalar from a primitive value with this array's dimension and display unit."""
        pass

    def get_as_numpy_skalar(self, index: int, unit: UT) -> PT:
        value: PT = self.canonical_np_array[index]
        return unit.from_canonical_value(value)
            
    def get_as_united_array(self, slice: slice) -> UAT:
        canonical_np_array: np.ndarray = self.canonical_np_array[slice]
        return type(self)(canonical_np_array, self.dimension, self.display_unit)
    
    def get_as_numpy_array(self, slice: slice, unit: UT) -> np.ndarray:
        match self.value_type:
            case float():
                array: np.ndarray = self.canonical_np_array[slice]
                return unit.to_canonical_value(array).astype(self.value_type.value.dtype)
            case complex():
                array: np.ndarray = self.canonical_np_array[slice]
                return unit.to_canonical_value(array).astype(self.value_type.value.dtype)
            case _:
                raise ValueError(f"Invalid value type: {self.value_type}")
    
    # Iterator protocol methods
    def __iter__(self) -> Iterator[UST]:
        """Return an iterator over the array elements."""
        return BaseUnitedArrayIterator(self)
    
    def __next__(self) -> UST:
        """This method should not be called on the array itself."""
        raise TypeError("BaseUnitedArray object is not an iterator")
    
    def __contains__(self, item: UST) -> bool:
        """Check if an item is in the array."""
        if not isinstance(item, UnitedScalar):
            return False
        # Check if the dimension is compatible
        if not self.dimension.compatible_to(item.dimension):
            return False
        # Check if the canonical value is in the array
        return float(item.canonical_value) in self.canonical_np_array

    @overload
    def __getitem__(self, index_key: int) -> UST:
        ...
    @overload
    def __getitem__(self, index_key: slice) -> UAT:
        ...
    def __getitem__(self, index_key: int|slice) -> UAT:
        match index_key:
            case int():
                return self.get_as_united_value(index_key)
            case slice():
                return self.get_as_united_array(index_key)
            case _:
                raise ValueError(f"Invalid index key: {index_key}")

    @overload
    def __add__(self, other: UAT) -> UAT:
        ...
    @overload
    def __add__(self, other: UST) -> UAT:
        ...
    def __add__(self, other: UAT|UST) -> UAT:
        match other:
            case UnitedScalar():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array + other.canonical_value
                return type(self)(array, self.dimension, self.display_unit)
            case BaseUnitedArray():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array + other.canonical_np_array
                return type(self)(array, self.dimension, self.display_unit)
            case _:
                raise ValueError(f"Invalid other: {other}")

    @overload    
    def __radd__(self, other: UAT) -> UAT:
        ...
    @overload
    def __radd__(self, other: UST) -> UAT:
        ...
    def __radd__(self, other: UAT|UST) -> UAT:
        return self.__add__(other)
    
    @overload    
    def __sub__(self, other: UAT) -> UAT:
        ...
    @overload
    def __sub__(self, other: UST) -> UAT:
        ...
    def __sub__(self, other: UAT|UST) -> UAT:
        match other:
            case UnitedScalar():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array - other.canonical_value
                return type(self)(array, self.dimension, self.display_unit)
            case BaseUnitedArray():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array - other.canonical_np_array
                return type(self)(array, self.dimension, self.display_unit)
            case _:
                raise ValueError(f"Invalid other: {other}")

    @overload    
    def __rsub__(self, other: UAT) -> UAT:
        ...
    @overload
    def __rsub__(self, other: UST) -> UAT:
        ...
    def __rsub__(self, other: UAT|UST) -> UAT:
        match other:
            case UnitedScalar():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = other.canonical_value - self.canonical_np_array
                return type(self)(array, self.dimension, self.display_unit)
            case BaseUnitedArray():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = other.canonical_np_array - self.canonical_np_array
                return type(self)(array, self.dimension, self.display_unit)
            case _:
                raise ValueError(f"Invalid other: {other}")
    
    @overload    
    def __mul__(self, other: PT) -> UAT:
        ...
    @overload
    def __mul__(self, other: UST) -> UAT:
        ...
    @overload
    def __mul__(self, other: UAT) -> UAT:
        ...
    def __mul__(self, other: PT|UST|UAT) -> UAT:
        match other:
            case float() | int() | complex():
                array: np.ndarray = self.canonical_np_array * other
                return type(self)(array, self.dimension, self.display_unit)
            case UnitedScalar():
                array: np.ndarray = self.canonical_np_array * other.canonical_value
                dimension: DT = self.dimension + other.dimension
                return type(self)(array, dimension, None)
            case BaseUnitedArray():
                array: np.ndarray = self.canonical_np_array * other.canonical_np_array
                dimension: DT = self.dimension + other.dimension
                return type(self)(array, dimension, None)
            case _:
                raise ValueError(f"Invalid other: {other}")

    @overload    
    def __rmul__(self, other: PT) -> UAT:
        ...
    @overload
    def __rmul__(self, other: UST) -> UAT:
        ...
    @overload
    def __rmul__(self, other: UAT) -> UAT:
        ...
    def __rmul__(self, other: PT|UST|UAT) -> UAT:
        return self.__mul__(other)
    
    @overload
    def __truediv__(self, other: PT) -> UAT:
        ...
    @overload
    def __truediv__(self, other: UST) -> UAT:
        ...
    @overload
    def __truediv__(self, other: UAT) -> UAT:
        ...
    def __truediv__(self, other: UST|PT|UAT) -> UAT:
        match other:
            case float() | int() | complex():
                array: np.ndarray = self.canonical_np_array / other
                return type(self)(array, self.dimension, self.display_unit)
            case UnitedScalar():
                array: np.ndarray = self.canonical_np_array / other.canonical_value
                dimension: DT = self.dimension - other.dimension
                return type(self)(array, dimension, None)
            case BaseUnitedArray():
                array: np.ndarray = self.canonical_np_array / other.canonical_np_array
                dimension: DT = self.dimension - other.dimension
                return type(self)(array, dimension, None)
            case _:
                raise ValueError(f"Invalid other: {other}")
    
    @overload    
    def __rtruediv__(self, other: PT) -> UAT:
        ...
    @overload
    def __rtruediv__(self, other: UST) -> UAT:
        ...
    @overload
    def __rtruediv__(self, other: UAT) -> UAT:
        ...
    def __rtruediv__(self, other: UST|PT|UAT) -> UAT:
        match other:
            case float() | int() | complex():
                array: np.ndarray = other / self.canonical_np_array
                # For division by array, we need to invert the dimension
                from ....units.simple.simple_dimension import SimpleDimension
                if isinstance(self.dimension, SimpleDimension):
                    zero_dim = SimpleDimension.create([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0, 0])
                    dimension = zero_dim - self.dimension
                else:
                    dimension = self.dimension * (-1.0)  # Fallback for other dimension types
                return type(self)(array, dimension, None)
            case UnitedScalar():
                array: np.ndarray = other.canonical_value / self.canonical_np_array
                dimension: DT = other.dimension - self.dimension
                return type(self)(array, dimension, None)
            case BaseUnitedArray():
                array: np.ndarray = other.canonical_np_array / self.canonical_np_array
                dimension: DT = other.dimension - self.dimension
                return type(self)(array, dimension, None)
            case _:
                raise ValueError(f"Invalid other: {other}")

    def __pow__(self, exponent: float) -> UAT:
        """Raise the array to a power."""
        array: np.ndarray = self.canonical_np_array ** exponent
        dimension: DT = self.dimension * exponent
        return type(self)(array, dimension, None)

    def __neg__(self) -> UAT:
        array: np.ndarray = -1 * self.canonical_np_array
        return type(self)(array, self.dimension, self.display_unit)
    
    def __abs__(self) -> UAT:
        array: np.ndarray = np.abs(self.canonical_np_array)
        return type(self)(array, self.dimension, self.display_unit)
    
    def __len__(self) -> int:
        return len(self.canonical_np_array)
    
    def sum(self) -> UST:
        sum: PT = np.sum(self.canonical_np_array)
        return self._get_scalar_from_value(sum)
    
    def mean(self) -> UST:
        mean: PT = np.mean(self.canonical_np_array)
        return self._get_scalar_from_value(mean)
    
    def std(self) -> UST:
        std: PT = np.std(self.canonical_np_array)
        return self._get_scalar_from_value(std)
    
    def min(self) -> UST:
        min: PT = np.min(self.canonical_np_array)
        return self._get_scalar_from_value(min)
    
    def max(self) -> UST:
        max: PT = np.max(self.canonical_np_array)
        return self._get_scalar_from_value(max)
    
    def var(self) -> UST:
        var: PT = np.var(self.canonical_np_array)
        return self._get_scalar_from_value(var)
    
    def to_json(self) -> dict[str, Any]:
        canonical_dimension_as_unit: UT = self.dimension.canonical_unit
        return {
            "canonical_np_array": self.canonical_np_array.tolist(),
            "canonical_dimension_as_unit": canonical_dimension_as_unit.to_json(),
            "display_unit": self.display_unit.to_json() if self.display_unit else None
        }
    
    @classmethod
    def from_json(cls, json: dict[str, Any], type_parameters: Type) -> UAT:
        """
        type_parameters is a dictionary of type parameters for the class.

        It must contain the unit_type parameter.
        """
        if "unit_type" in type_parameters:
            unit_type: Type[UT] = type_parameters["unit_type"]
        else:
            raise ValueError("unit_type is not provided")        
        canonical_np_array: np.ndarray = np.array(json["canonical_np_array"])
        canonical_dimension_as_unit: UT = unit_type.from_json(json["canonical_dimension_as_unit"])
        dimension: DT = canonical_dimension_as_unit.dimension
        display_unit: UT|None = unit_type.from_json(json["display_unit"]) if json["display_unit"] else None

        return cls(
            canonical_np_array=canonical_np_array,
            dimension=dimension,
            display_unit=display_unit)
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        canonical_dimension_as_unit: UT = self.dimension.canonical_unit
        hdf5_group.attrs["canonical_np_array"] = self.canonical_np_array.tolist()
        hdf5_group.attrs["canonical_dimension_as_unit"] = canonical_dimension_as_unit.to_json()
        hdf5_group.attrs["display_unit"] = self.display_unit.to_json() if self.display_unit else None
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, type_parameters: Type) -> UAT:
        """
        type_parameters is a dictionary of type parameters for the class.
        
        It must contain the unit_type parameter.
        """
        if "unit_type" in type_parameters:
            unit_type: Type[UT] = type_parameters["unit_type"]
        else:
            raise ValueError("unit_type is not provided")       
        canonical_np_array: np.ndarray = np.array(hdf5_group.attrs["canonical_np_array"])
        canonical_dimension_as_unit: UT = unit_type.from_json(hdf5_group.attrs["canonical_dimension_as_unit"])
        dimension: DT = canonical_dimension_as_unit.dimension
        display_unit: UT|None = unit_type.from_json(hdf5_group.attrs["display_unit"]) if hdf5_group.attrs["display_unit"] else None

        return cls(
            canonical_np_array=canonical_np_array,
            dimension=dimension,
            display_unit=display_unit)

    def in_unit(self, unit: UT) -> np.ndarray:
        """Convert the array to a specific unit and return numpy array."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_np_array)
    
    def with_display_unit(self, unit: UT) -> UAT:
        """Return a new array with the specified display unit."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return type(self)(self.canonical_np_array, self.dimension, unit)
    
    def compatible_to(self, other: UAT) -> bool:
        """Check if this array is compatible (same dimension) with another."""
        return self.dimension.compatible_to(other.dimension)

    def format(self, unit: Optional[UT] = None, decimals: int = 3) -> str:
        """Format the array with optional unit specification."""
        if unit is not None:
            values = self.in_unit(unit)
            unit_str = unit.format_string(no_fraction=False)
        else:
            values = self.canonical_np_array
            unit_str = self.dimension.canonical_unit.format_string(no_fraction=False)
        
        # Format array values
        if len(values) <= 10:
            values_str = ", ".join([f"{v:.{decimals}f}" for v in values])
        else:
            # Show first 3 and last 3 for large arrays
            first_three = ", ".join([f"{v:.{decimals}f}" for v in values[:3]])
            last_three = ", ".join([f"{v:.{decimals}f}" for v in values[-3:]])
            values_str = f"{first_three}, ..., {last_three}"
        
        return f"[{values_str}] {unit_str}"


class BaseUnitedArrayIterator(Iterator[UST]):
    """Iterator for BaseUnitedArray that maintains separate state."""
    
    def __init__(self, array: BaseUnitedArray):
        self.array = array
        self.index = 0
    
    def __iter__(self) -> Iterator[UST]:
        return self
    
    def __next__(self) -> UST:
        if self.index >= len(self.array.canonical_np_array):
            raise StopIteration
        result = self.array.get_as_united_value(self.index)
        self.index += 1
        return result