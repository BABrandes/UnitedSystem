import numpy as np
from dataclasses import dataclass
from typing import overload, Any, TypeVar, Generic, Union, Type, Iterator, Optional
from ..units.dimension_group import BaseDimension
from ..units.unit_group import BaseUnit
from ...utils.scalars.united_scalar import UnitedScalar
from ...utils.units.united import United
from .base_array import BaseArray, PT_TYPE
from .protocol_numerical_array import ProtocolNumericalArray
from abc import ABC, abstractmethod
import h5py
import pandas as pd
from pandas._typing import Dtype

UAT = TypeVar("UAT", bound="BaseUnitedArray[Any, Any, Any, Any, Any]")
UST = TypeVar("UST", bound=UnitedScalar[Any, Any, Any, Any])
UT = TypeVar("UT", bound=BaseUnit[Any, Any])
DT = TypeVar("DT", bound=BaseDimension[Any, Any])
PT = TypeVar("PT", bound=PT_TYPE)

class ScalarIterator(Iterator[UST], Generic[UST, PT]):
    def __init__(self, array: "BaseUnitedArray[UAT, UST, UT, DT, PT]"):
        self.array = array
        self.index: int = 0
    
    def __iter__(self) -> Iterator[UST]:
        return self
    
    def __next__(self) -> UST:
        if self.index >= len(self.array):
            raise StopIteration
        result = self.array.get_scalar(self.index)
        self.index += 1
        return result
    
    def __len__(self) -> int:
        return len(self.array)

@dataclass(frozen=True, slots=True, init=False)
class BaseUnitedArray(BaseArray[PT, UST, UAT], United[DT, UT], ProtocolNumericalArray[PT], ABC, Generic[UAT, UST, UT, DT, PT]):
    
    # Required field from BaseArray inheritance
    canonical_np_array: np.ndarray
    
    # Required fields from United inheritance
    dimension: DT
    _display_unit: Optional[UT]

    def __init__(self, canonical_np_array: np.ndarray, dimension: DT, display_unit: Optional[UT] = None):

        # Check the dimension and display unit are compatible
        if display_unit is not None and not display_unit.compatible_to(dimension):
            raise ValueError(f"The display unit {display_unit} is not compatible with the dimension {dimension}.")
        
        super().__init__(canonical_np_array)

        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", display_unit)

    @classmethod
    def create(cls, canonical_np_array: np.ndarray, dimension_or_display_unit: Union[DT, UT]) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        match dimension_or_display_unit:
            case BaseDimension():
                dimension: DT = dimension_or_display_unit
                display_unit: UT = dimension_or_display_unit.canonical_unit
            case BaseUnit():
                dimension: DT = dimension_or_display_unit.dimension
                display_unit: UT = dimension_or_display_unit
        
        return cls(canonical_np_array, dimension, display_unit)
    
    @property
    def display_unit(self) -> UT:
        if self._display_unit is None:
            object.__setattr__(self, "_display_unit", self.dimension.canonical_unit)
        if self._display_unit is None:
            raise AssertionError("Display unit is None")
        return self._display_unit

    def set_display_unit(self, display_unit: UT) -> None:
        if display_unit.compatible_to(self.dimension):
            raise ValueError(f"The display unit {display_unit} is not compatible with the canonical dimension {self.dimension}.")
        self.__setattr__("_display_unit", display_unit)

    def get_as_scalar(self, index: int) -> UST:
        """Get a single element as a united scalar."""
        return self.get_scalar_from_value(self.canonical_np_array[index])
    
    @abstractmethod
    def get_scalar_from_value(self, value: PT) -> UST:
        """Create a scalar from a primitive value with this array's dimension and display unit."""
        raise NotImplementedError("This method must be implemented by the subclass.")

    def get_value(self, index: int) -> PT:
        value: PT = self.canonical_np_array[index]
        converted_value: PT = self.display_unit.from_canonical_value(value)  # type: ignore
        return converted_value # type: ignore
    
    def get_scalar(self, index: int) -> UST:
        value: PT = self.canonical_np_array[index]
        return self.get_scalar_from_value(value)
            
    def get_array(self, slice: slice) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        canonical_np_array: np.ndarray = self.canonical_np_array[slice]
        return type(self)(canonical_np_array, self.dimension, self._display_unit)
    
    def get_as_numpy_array(self, target_unit: UT|None, slice: slice|None = None) -> np.ndarray:
        """
        Get the array as a numpy array.

        If target_unit is None, the array is returned in the display unit.
        If target_unit is provided, the array is returned in the target unit.
        If slice is provided, the array is returned as a slice of the original array.
        """
        if slice is None:
            canonical_np_array: np.ndarray = self.canonical_np_array
        else:
            canonical_np_array: np.ndarray = self.canonical_np_array[slice]
        if target_unit is None:
            return self.display_unit.from_canonical_value(canonical_np_array)
        else:
            return target_unit.from_canonical_value(canonical_np_array)
    
    def get_pandas_series(self, dtype: Dtype, slice: slice|None = None, target_unit: UT|None = None) -> pd.Series: # type: ignore
        """
        Get the array as a pandas series.

        If target_unit is None, the array is returned in the display unit.
        If target_unit is provided, the array is returned in the target unit.
        If slice is provided, the array is returned as a slice of the original array.
        
        Args:
            dtype: The dtype of the series.
            slice: The slice of the array to return.
            target_unit: The unit to convert the array to.

        Returns:
            A pandas series with the array values in the target unit.
        """
        if slice is None:
            canonical_np_array: np.ndarray = self.canonical_np_array
        else:
            canonical_np_array: np.ndarray = self.canonical_np_array[slice]
        if target_unit is None:
            return pd.Series(self.display_unit.from_canonical_value(canonical_np_array), dtype=dtype)
        else:
            return pd.Series(target_unit.from_canonical_value(canonical_np_array), dtype=dtype)
    
    # Iterator protocol methods
    def __iter__(self) -> Iterator[PT]:
        """Return an iterator over the array elements."""
        return iter(self.canonical_np_array)
    
    def __next__(self) -> PT:
        return next(self.canonical_np_array) # type: ignore
    
    def __contains__(self, item: PT|UST) -> bool:
        """Check if an item is in the array."""
        if isinstance(item, float) or isinstance(item, complex) or isinstance(item, int):
            if self._display_unit is None:
                return item in self.canonical_np_array
            else:
                return self._display_unit.to_canonical_value(item) in self.canonical_np_array
        elif isinstance(item, UnitedScalar):
            if self._display_unit is None:
                return item.canonical_value in self.canonical_np_array
            else:
                return self._display_unit.to_canonical_value(item.canonical_value) in self.canonical_np_array
        else:
            raise ValueError(f"Item {item} is not a valid item for this array.")

    def scalars(self) -> ScalarIterator[UST, PT]:
        return ScalarIterator(self)

    @overload
    def __getitem__(self, index_key: int) -> UST:
        ...
    @overload
    def __getitem__(self, index_key: slice) -> UAT:
        ...
    def __getitem__(self, index_key: int|slice) -> UST|UAT:
        match index_key:
            case int():
                return self.get_scalar(index_key)
            case slice():
                return self.get_array(index_key) # type: ignore

    @overload
    def __add__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __add__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __add__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]"|UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        match other:
            case UnitedScalar():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array + other.canonical_value
                return type(self)(array, self.dimension, self._display_unit)
            case BaseUnitedArray():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array + other.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit)

    @overload    
    def __radd__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __radd__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __radd__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]"|UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        return self.__add__(other)
    
    @overload    
    def __sub__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __sub__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __sub__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]"|UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        match other:
            case UnitedScalar():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array - other.canonical_value
                return type(self)(array, self.dimension, self._display_unit)
            case BaseUnitedArray():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array - other.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit)

    @overload    
    def __rsub__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __rsub__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __rsub__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]"|UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        match other:
            case UnitedScalar():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = other.canonical_value - self.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit)
            case BaseUnitedArray():
                if not self.dimension.compatible_to(other.dimension):
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = other.canonical_np_array - self.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit)
    
    @overload    
    def __mul__(self, other: int|float|complex) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __mul__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __mul__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __mul__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        if isinstance(other, UnitedScalar):
            array: np.ndarray = self.canonical_np_array * other.canonical_value
            dimension: DT = self.dimension * other.dimension
            return type(self)(array, dimension, None)
        elif isinstance(other, BaseUnitedArray):
            array: np.ndarray = self.canonical_np_array * other.canonical_np_array
            dimension: DT = self.dimension * other.dimension
            return type(self)(array, dimension, None)
        else:
            array: np.ndarray = self.canonical_np_array.__mul__(other)
            return type(self)(array, self.dimension, self._display_unit)

    @overload    
    def __rmul__(self, other: int|float|complex) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __rmul__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __rmul__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __rmul__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        return self.__mul__(other)
    
    @overload
    def __truediv__(self, other: int|float|complex) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __truediv__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __truediv__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __truediv__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        if isinstance(other, UnitedScalar):
            array: np.ndarray = self.canonical_np_array / other.canonical_value
            dimension: DT = self.dimension / other.dimension
            return type(self)(array, dimension, None)
        elif isinstance(other, BaseUnitedArray):
            array: np.ndarray = self.canonical_np_array / other.canonical_np_array
            dimension: DT = self.dimension / other.dimension
            return type(self)(array, dimension, None)
        else:
            array: np.ndarray = self.canonical_np_array.__truediv__(other)
            return type(self)(array, self.dimension, self._display_unit)
    
    @overload    
    def __rtruediv__(self, other: int|float|complex) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __rtruediv__(self, other: UST) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    @overload
    def __rtruediv__(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        ...
    def __rtruediv__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, UT, DT, PT]") -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        if isinstance(other, UnitedScalar):
            array: np.ndarray = other.canonical_value / self.canonical_np_array
            dimension: DT = other.dimension - self.dimension
            return type(self)(array, dimension, None)
        elif isinstance(other, BaseUnitedArray):
            array: np.ndarray = other.canonical_np_array / self.canonical_np_array
            dimension: DT = other.dimension / self.dimension
            return type(self)(array, dimension, None)
        else:
            array: np.ndarray = other / self.canonical_np_array
            zero_dim: DT = (type(self.dimension)).dimensionless_dimension()
            dimension = zero_dim / self.dimension
            return type(self)(array, dimension, None)

    def __pow__(self, exponent: float|int) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        """Raise the array to a power."""
        array: np.ndarray = self.canonical_np_array.__pow__(exponent)
        dimension: DT = self.dimension ** exponent
        return type(self)(array, dimension, None)

    def __neg__(self) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        array: np.ndarray = -1 * self.canonical_np_array
        return type(self)(array, self.dimension, self._display_unit)
    
    def __abs__(self) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        array: np.ndarray = np.abs(self.canonical_np_array)
        return type(self)(array, self.dimension, self._display_unit)
    
    def __eq__(self, other: object) -> bool:
        """Compare arrays for equality."""
        if not isinstance(other, BaseUnitedArray):
            return False
        
        # Check if dimensions are compatible
        if not self.dimension.compatible_to(other.dimension): # type: ignore
            return False
        
        # Compare canonical values (they should be in the same units)
        return bool(np.array_equal(self.canonical_np_array, other.canonical_np_array))
    
    def __ne__(self, other: object) -> bool:
        """Compare arrays for inequality."""
        return not self.__eq__(other)
    
    def __len__(self) -> int:
        return len(self.canonical_np_array)
    
    def sum(self) -> PT:
        sum: PT = np.sum(self.canonical_np_array)
        return sum
    
    def sum_as_scalar(self) -> UST:
        sum: PT = np.sum(self.canonical_np_array)
        return self.get_scalar_from_value(sum)

    def mean(self) -> PT:
        mean: PT = np.mean(self.canonical_np_array) # type: ignore
        return mean
    
    def mean_as_scalar(self) -> UST:
        mean: PT = np.mean(self.canonical_np_array) # type: ignore
        return self.get_scalar_from_value(mean)
    
    def std(self) -> PT:
        std: PT = np.std(self.canonical_np_array) # type: ignore
        return std
    
    def std_as_scalar(self) -> UST:
        std: PT = np.std(self.canonical_np_array) # type: ignore
        return self.get_scalar_from_value(std)
    
    def min(self) -> PT:
        min: PT = np.min(self.canonical_np_array)
        return min
    
    def min_as_scalar(self) -> UST:
        min: PT = np.min(self.canonical_np_array)
        return self.get_scalar_from_value(min)
    
    def max(self) -> PT:
        max: PT = np.max(self.canonical_np_array)
        return max
    
    def max_as_scalar(self) -> UST:
        max: PT = np.max(self.canonical_np_array)
        return self.get_scalar_from_value(max)
    
    def var(self) -> PT:
        var: PT = np.var(self.canonical_np_array) # type: ignore
        return var
    
    def var_as_scalar(self) -> UST:
        var: PT = np.var(self.canonical_np_array) # type: ignore
        return self.get_scalar_from_value(var)
    
    def to_json(self) -> dict[str, Any]:
        canonical_dimension_as_unit: UT = self.dimension.canonical_unit
        return {
            "canonical_np_array": self.canonical_np_array.tolist(),
            "canonical_dimension_as_unit": canonical_dimension_as_unit.to_json(),
            "display_unit": self._display_unit.to_json() if self._display_unit else None
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any], **type_parameters: Type[UT]) -> UAT:
        """
        type_parameters is a dictionary of type parameters for the class.

        It must contain the unit_type parameter.
        """
        if "unit_type" in type_parameters:
            unit_type: Type[UT] = type_parameters["unit_type"]
        else:
            raise ValueError("unit_type is not provided")        
        canonical_np_array: np.ndarray = np.array(data["canonical_np_array"])
        canonical_dimension_as_unit: UT = unit_type.from_json(data["canonical_dimension_as_unit"])
        dimension: DT = canonical_dimension_as_unit.dimension
        display_unit: UT|None = unit_type.from_json(data["display_unit"]) if data["display_unit"] else None

        return type(cls)(
            canonical_np_array=canonical_np_array,
            dimension=dimension,
            display_unit=display_unit)
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        canonical_dimension_as_unit: UT = self.dimension.canonical_unit
        hdf5_group.attrs["canonical_np_array"] = self.canonical_np_array.tolist()
        hdf5_group.attrs["canonical_dimension_as_unit"] = canonical_dimension_as_unit.to_json()
        hdf5_group.attrs["display_unit"] = self._display_unit.to_json() if self._display_unit else None
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **type_parameters: Type[UT]) -> UAT:
        """
        type_parameters is a dictionary of type parameters for the class.
        
        It must contain the unit_type parameter.
        """
        if "unit_type" in type_parameters:
            unit_type: Type[UT] = type_parameters["unit_type"]
        else:
            raise ValueError("unit_type is not provided")       
        canonical_np_array: np.ndarray = np.array(hdf5_group.attrs["canonical_np_array"])
        canonical_dimension_as_unit: UT = unit_type.from_json(hdf5_group.attrs["canonical_dimension_as_unit"]) # type: ignore
        dimension: DT = canonical_dimension_as_unit.dimension
        display_unit: UT|None = unit_type.from_json(hdf5_group.attrs["display_unit"]) if hdf5_group.attrs["display_unit"] else None # type: ignore

        return type(cls)(
            canonical_np_array=canonical_np_array,
            dimension=dimension,
            display_unit=display_unit)

    def in_unit(self, unit: UT) -> np.ndarray:
        """Convert the array to a specific unit and return numpy array."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_np_array)
    
    def with_display_unit(self, unit: UT) -> "BaseUnitedArray[UAT, UST, UT, DT, PT]":
        """Return a new array with the specified display unit."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return type(self)(self.canonical_np_array, self.dimension, unit)
    
    def compatible_to(self, other: "BaseUnitedArray[UAT, UST, UT, DT, PT]") -> bool:
        """Check if this array is compatible (same dimension) with another."""
        return self.dimension.compatible_to(other.dimension)

    def format(self, unit: Optional[UT] = None, decimals: int = 3) -> str:
        """Format the array with optional unit specification."""
        if unit is not None:
            values = self.in_unit(unit)
            unit_str = unit.format_string(as_fraction=True)
        else:
            values = self.canonical_np_array
            unit_str = self.dimension.canonical_unit.format_string(as_fraction=True)
        
        # Format array values
        if len(values) <= 10:
            values_str = ", ".join([f"{v:.{decimals}f}" for v in values])
        else:
            # Show first 3 and last 3 for large arrays
            first_three = ", ".join([f"{v:.{decimals}f}" for v in values[:3]])
            last_three = ", ".join([f"{v:.{decimals}f}" for v in values[-3:]])
            values_str = f"{first_three}, ..., {last_three}"
        
        return f"[{values_str}] {unit_str}"