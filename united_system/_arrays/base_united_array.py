import numpy as np
from dataclasses import dataclass
from typing import overload, Any, TypeVar, Generic, Union, Iterator, Optional, Sequence
from .._units_and_dimension.dimension import Dimension
from .._units_and_dimension.unit import Unit
from .._scalars.united_scalar import UnitedScalar
from .._units_and_dimension.united import United
from .base_array import BaseArray, PT_TYPE
from .protocol_numerical_array import ProtocolNumericalArray
from abc import ABC, abstractmethod
import h5py
import pandas as pd
from pandas._typing import Dtype
from .._units_and_dimension.named_quantity import NamedQuantity

UAT = TypeVar("UAT", bound="BaseUnitedArray[Any, Any, Any]")
UST = TypeVar("UST", bound=UnitedScalar[Any, Any])
PT = TypeVar("PT", bound=PT_TYPE)

class ScalarIterator(Iterator[UST], Generic[UST, PT]):
    def __init__(self, array: "BaseUnitedArray[UAT, UST, PT]"):
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
class BaseUnitedArray(BaseArray[PT, UST, UAT], United, ProtocolNumericalArray[PT], ABC, Generic[UAT, UST, PT]):
    
    # Required field from BaseArray inheritance
    canonical_np_array: np.ndarray
    
    # Required fields from United inheritance
    dimension: Dimension
    _display_unit: Optional[Unit]

    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[Unit]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[str]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[Dimension]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[NamedQuantity]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[Dimension], display_unit: Optional[Unit]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[Dimension], display_unit: Optional[str]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[str], display_unit: Optional[str]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[NamedQuantity], display_unit: Optional[Unit]) -> None: # type: ignore
        ...
    @overload
    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, unit_or_dimension: Optional[NamedQuantity], display_unit: Optional[str]) -> None: # type: ignore
        ...

    def __init__(self, array: np.ndarray|Sequence[PT]|pd.Series, dimension_or_unit: Optional[Dimension|NamedQuantity|str|Unit], unit: Optional[Unit|str] = None): # type: ignore

        if isinstance(array, np.ndarray):
            np_array = array
        elif isinstance(array, pd.Series):
            np_array = array.to_numpy() # type: ignore
        else:
            np_array = np.array(array)

        if not type(self)._check_numpy_type(np_array):
            raise ValueError(f"The array is not the correct numpy array. It is of type {type(np_array)}.")
        
        # If a string is provided, it is assumed to be a dimension or unit depending on the context
        if isinstance(dimension_or_unit, str):
            if unit is not None:
                dimension_or_unit = Dimension(dimension_or_unit)
            else:
                dimension_or_unit = Unit(dimension_or_unit)

        # Check the dimension and display unit are compatible
        if unit is not None:
            if isinstance(unit, str):
                unit = Unit(unit)
            if dimension_or_unit is None:
                dimension_or_unit = unit.dimension
            elif isinstance(dimension_or_unit, NamedQuantity):
                if not unit.compatible_to(dimension_or_unit.dimension):
                    raise ValueError(f"The display unit {unit} is not compatible with the dimension {dimension_or_unit.dimension}.")
            elif isinstance(dimension_or_unit, Unit):
                raise ValueError("One cannot provide two units to the constructor.")
            else:
                if not unit.compatible_to(dimension_or_unit):
                    raise ValueError(f"The display unit {unit} is not compatible with the dimension {dimension_or_unit}.")
            canonical_np_array = unit.to_canonical_value(np_array)

        if dimension_or_unit is None:
            dimension = Dimension.dimensionless_dimension()
            canonical_np_array = np_array
        elif isinstance(dimension_or_unit, NamedQuantity):
            dimension = dimension_or_unit.dimension
            canonical_np_array = np_array
        elif isinstance(dimension_or_unit, Unit):
            dimension = dimension_or_unit.dimension
            canonical_np_array = dimension_or_unit.to_canonical_value(np_array)
        else:
            dimension = dimension_or_unit
            canonical_np_array = np_array
        
        super().__init__(canonical_np_array)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_display_unit", unit)

    @classmethod
    def create(cls, canonical_np_array: np.ndarray, dimension_or_display_unit: Union[Dimension, Unit]) -> UAT:
        match dimension_or_display_unit:
            case Dimension():
                dimension: Dimension = dimension_or_display_unit
                display_unit: Unit = dimension_or_display_unit.canonical_unit
            case Unit():
                dimension: Dimension = dimension_or_display_unit.dimension
                display_unit: Unit = dimension_or_display_unit
        
        return cls(canonical_np_array, dimension, display_unit) # type: ignore
    
    @property
    def unit(self) -> Unit:
        if self._display_unit is None:
            object.__setattr__(self, "_display_unit", self.dimension.canonical_unit)
        if self._display_unit is None:
            raise AssertionError("Display unit is None")
        return self._display_unit
    

    def set_display_unit(self, display_unit: Unit) -> None:
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
        converted_value: PT = self.unit.from_canonical_value(value)  # type: ignore
        return converted_value # type: ignore
    
    def get_scalar(self, index: int) -> UST:
        value: PT = self.canonical_np_array[index]
        return self.get_scalar_from_value(value)
            
    def get_array(self, slice: slice) -> "BaseUnitedArray[UAT, UST, PT]":
        canonical_np_array: np.ndarray = self.canonical_np_array[slice]
        return type(self)(canonical_np_array, self.dimension, self._display_unit) # type: ignore
    
    def get_as_numpy_array(self, target_unit: Unit|None, slice: slice|None = None) -> np.ndarray:
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
            return self.unit.from_canonical_value(canonical_np_array)
        else:
            return target_unit.from_canonical_value(canonical_np_array)
    
    def get_pandas_series(self, dtype: Dtype, slice: slice|None = None, target_unit: Unit|None = None) -> pd.Series: # type: ignore
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
            return pd.Series(self.unit.from_canonical_value(canonical_np_array), dtype=dtype)
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
    def __add__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    @overload
    def __add__(self, other: UST) -> UAT:
        ...
    def __add__(self, other: "BaseUnitedArray[UAT, UST, PT]"|UST) -> UAT:
        match other:
            case UnitedScalar():
                if not self.dimension == other.dimension:
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array + other.canonical_value
                return type(self)(array, self.dimension, self._display_unit) # type: ignore
            case BaseUnitedArray():
                if not self.dimension == other.dimension:
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array + other.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit) # type: ignore

    @overload    
    def __radd__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    @overload
    def __radd__(self, other: UST) -> UAT:
        ...
    def __radd__(self, other: "BaseUnitedArray[UAT, UST, PT]"|UST) -> UAT:
        return self.__add__(other)
    
    @overload    
    def __sub__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    @overload
    def __sub__(self, other: UST) -> UAT:
        ...
    def __sub__(self, other: "BaseUnitedArray[UAT, UST, PT]"|UST) -> UAT:
        match other:
            case UnitedScalar():
                if not self.dimension == other.dimension:
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array - other.canonical_value
                return type(self)(array, self.dimension, self._display_unit) # type: ignore
            case BaseUnitedArray():
                if not self.dimension == other.dimension:
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = self.canonical_np_array - other.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit) # type: ignore

    @overload    
    def __rsub__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    @overload
    def __rsub__(self, other: UST) -> UAT:
        ...
    def __rsub__(self, other: "BaseUnitedArray[UAT, UST, PT]"|UST) -> UAT:
        match other:
            case UnitedScalar():
                if not self.dimension == other.dimension:
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = other.canonical_value - self.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit) # type: ignore
            case BaseUnitedArray():
                if not self.dimension == other.dimension:
                    raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
                array: np.ndarray = other.canonical_np_array - self.canonical_np_array
                return type(self)(array, self.dimension, self._display_unit) # type: ignore
    
    @overload    
    def __mul__(self, other: int|float|complex) -> UAT:
        ...
    @overload
    def __mul__(self, other: UST) -> UAT:
        ...
    @overload
    def __mul__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    def __mul__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, PT]") -> UAT:
        if isinstance(other, UnitedScalar):
            array: np.ndarray = self.canonical_np_array * other.canonical_value
            dimension: Dimension = self.dimension * other.dimension
            return type(self)(array, dimension, None) # type: ignore
        elif isinstance(other, BaseUnitedArray):
            array: np.ndarray = self.canonical_np_array * other.canonical_np_array
            dimension: Dimension = self.dimension * other.dimension
            return type(self)(array, dimension, None) # type: ignore
        else:
            array: np.ndarray = self.canonical_np_array.__mul__(other)
            return type(self)(array, self.dimension, self._display_unit) # type: ignore

    @overload    
    def __rmul__(self, other: int|float|complex) -> UAT:
        ...
    @overload
    def __rmul__(self, other: UST) -> UAT:
        ...
    @overload
    def __rmul__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    def __rmul__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, PT]") -> UAT:
        return self.__mul__(other)
    
    @overload
    def __truediv__(self, other: int|float|complex) -> UAT:
        ...
    @overload
    def __truediv__(self, other: UST) -> UAT:
        ...
    @overload
    def __truediv__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    def __truediv__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, PT]") -> UAT:
        if isinstance(other, UnitedScalar):
            array: np.ndarray = self.canonical_np_array / other.canonical_value
            dimension: Dimension = self.dimension / other.dimension
            return type(self)(array, dimension, None) # type: ignore
        elif isinstance(other, BaseUnitedArray):
            array: np.ndarray = self.canonical_np_array / other.canonical_np_array
            dimension: Dimension = self.dimension / other.dimension
            return type(self)(array, dimension, None) # type: ignore
        else:
            array: np.ndarray = self.canonical_np_array.__truediv__(other)
            return type(self)(array, self.dimension, self._display_unit) # type: ignore
    
    @overload    
    def __rtruediv__(self, other: int|float|complex) -> UAT:
        ...
    @overload
    def __rtruediv__(self, other: UST) -> UAT:
        ...
    @overload
    def __rtruediv__(self, other: "BaseUnitedArray[UAT, UST, PT]") -> UAT:
        ...
    def __rtruediv__(self, other: int|float|complex|UST|"BaseUnitedArray[UAT, UST, PT]") -> UAT:
        if isinstance(other, UnitedScalar):
            array: np.ndarray = other.canonical_value / self.canonical_np_array
            dimension: Dimension = other.dimension / self.dimension
            return type(self)(array, dimension, None) # type: ignore
        elif isinstance(other, BaseUnitedArray):
            array: np.ndarray = other.canonical_np_array / self.canonical_np_array
            dimension: Dimension = other.dimension / self.dimension
            return type(self)(array, dimension, None) # type: ignore
        else:
            array: np.ndarray = other / self.canonical_np_array
            zero_dim: Dimension = (type(self.dimension)).dimensionless_dimension()
            dimension = zero_dim / self.dimension
            return type(self)(array, dimension, None) # type: ignore

    def __pow__(self, exponent: float|int) -> UAT:
        """Raise the array to a power."""
        array: np.ndarray = self.canonical_np_array.__pow__(exponent)
        dimension: Dimension = self.dimension ** exponent
        return type(self)(array, dimension, None) # type: ignore

    def __neg__(self) -> UAT:
        array: np.ndarray = -1 * self.canonical_np_array
        return type(self)(array, self.dimension, self._display_unit) # type: ignore
    
    def __abs__(self) -> UAT:
        array: np.ndarray = np.abs(self.canonical_np_array)
        return type(self)(array, self.dimension, self._display_unit) # type: ignore
    
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
        sum: PT = np.sum(self.get_as_numpy_array(target_unit=self.unit))
        return sum
    
    def sum_canonical(self) -> PT:
        sum: PT = np.sum(self.canonical_np_array)
        return sum
    
    def sum_in_unit(self, unit: Unit) -> PT:
        sum: PT = np.sum(self.get_as_numpy_array(target_unit=unit))
        return sum
    
    def sum_as_scalar(self) -> UST:
        sum: PT = np.sum(self.canonical_np_array)
        return self.get_scalar_from_value(sum)

    def mean(self) -> PT:
        mean: PT = np.mean(self.get_as_numpy_array(target_unit=self.unit)) # type: ignore
        return mean
    
    def mean_canonical(self) -> PT:
        mean: PT = np.mean(self.canonical_np_array) # type: ignore
        return mean
    
    def mean_in_unit(self, unit: Unit) -> PT:
        mean: PT = np.mean(self.get_as_numpy_array(target_unit=unit)) # type: ignore
        return mean
    
    def mean_as_scalar(self) -> UST:
        mean: PT = np.mean(self.canonical_np_array) # type: ignore
        return self.get_scalar_from_value(mean)
    
    def std(self) -> PT:
        std: PT = np.std(self.get_as_numpy_array(target_unit=self.unit)) # type: ignore
        return std
    
    def std_canonical(self) -> PT:
        std: PT = np.std(self.canonical_np_array) # type: ignore
        return std
    
    def std_in_unit(self, unit: Unit) -> PT:
        std: PT = np.std(self.get_as_numpy_array(target_unit=unit)) # type: ignore
        return std
    
    def std_as_scalar(self) -> UST:
        std: PT = np.std(self.canonical_np_array) # type: ignore
        return self.get_scalar_from_value(std)
    
    def min(self) -> PT:
        min: PT = np.min(self.get_as_numpy_array(target_unit=self.unit))
        return min
    
    def min_canonical(self) -> PT:
        min: PT = np.min(self.canonical_np_array)
        return min
    
    def min_in_unit(self, unit: Unit) -> PT:
        min: PT = np.min(self.get_as_numpy_array(target_unit=unit))
        return min
    def min_as_scalar(self) -> UST:
        min: PT = np.min(self.canonical_np_array)
        return self.get_scalar_from_value(min)
    
    def max(self) -> PT:
        max: PT = np.max(self.canonical_np_array)
        return max
    
    def max_canonical(self) -> PT:
        max: PT = np.max(self.canonical_np_array)
        return max
    
    def max_in_unit(self, unit: Unit) -> PT:
        max: PT = np.max(self.get_as_numpy_array(target_unit=unit))
        return max
    
    def max_as_scalar(self) -> UST:
        max: PT = np.max(self.canonical_np_array)
        return self.get_scalar_from_value(max)
    
    def var(self) -> PT:
        var: PT = np.var(self.get_as_numpy_array(target_unit=self.unit)) # type: ignore
        return var
    
    def var_canonical(self) -> PT:
        var: PT = np.var(self.canonical_np_array) # type: ignore
        return var
    
    def var_in_unit(self, unit: Unit) -> PT:
        var: PT = np.var(self.get_as_numpy_array(target_unit=unit)) # type: ignore
        return var
    
    def var_as_scalar(self) -> UST:
        var: PT = np.var(self.canonical_np_array) # type: ignore
        return self.get_scalar_from_value(var)
    
    def to_json(self) -> dict[str, Any]:
        canonical_dimension_as_unit: Unit = self.dimension.canonical_unit
        return {
            "canonical_np_array": self.canonical_np_array.tolist(),
            "canonical_dimension_as_unit": canonical_dimension_as_unit.to_json(),
            "display_unit": self._display_unit.to_json() if self._display_unit else None
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> UAT:
        canonical_np_array: np.ndarray = np.array(data["canonical_np_array"])
        canonical_dimension_as_unit: Unit = Unit.from_json(data["canonical_dimension_as_unit"])
        dimension: Dimension = canonical_dimension_as_unit.dimension
        display_unit: Unit|None = Unit.from_json(data["display_unit"]) if data["display_unit"] else None

        return type(cls)(
            canonical_np_array=canonical_np_array,
            dimension=dimension,
            display_unit=display_unit)
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        canonical_dimension_as_unit: Unit = self.dimension.canonical_unit
        hdf5_group.attrs["canonical_np_array"] = self.canonical_np_array.tolist()
        hdf5_group.attrs["canonical_dimension_as_unit"] = canonical_dimension_as_unit.to_json()
        hdf5_group.attrs["display_unit"] = self._display_unit.to_json() if self._display_unit else None
    
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> UAT:
        canonical_np_array: np.ndarray = np.array(hdf5_group.attrs["canonical_np_array"])
        canonical_dimension_as_unit: Unit = Unit.from_json(hdf5_group.attrs["canonical_dimension_as_unit"]) # type: ignore
        dimension: Dimension = canonical_dimension_as_unit.dimension
        display_unit: Unit|None = Unit.from_json(hdf5_group.attrs["display_unit"]) if hdf5_group.attrs["display_unit"] else None # type: ignore

        return type(cls)(
            canonical_np_array=canonical_np_array,
            dimension=dimension,
            display_unit=display_unit)

    def in_unit(self, unit: Unit) -> np.ndarray:
        """Convert the array to a specific unit and return numpy array."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return unit.from_canonical_value(self.canonical_np_array)
    
    def with_display_unit(self, unit: Unit) -> "BaseUnitedArray[UAT, UST, PT]":
        """Return a new array with the specified display unit."""
        if not unit.compatible_to(self.dimension):
            raise ValueError(f"Unit {unit} is not compatible with dimension {self.dimension}")
        return type(self)(self.canonical_np_array, self.dimension, unit)
    
    def compatible_to(self, other: "BaseUnitedArray[UAT, UST, PT]") -> bool:
        """Check if this array is compatible (same dimension) with another."""
        return self.dimension == other.dimension

    def format(self, unit: Optional[Unit] = None, decimals: int = 3) -> str:
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