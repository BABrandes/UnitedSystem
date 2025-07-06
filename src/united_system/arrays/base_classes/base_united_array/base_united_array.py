import numpy as np
from dataclasses import dataclass
from typing import overload, Any, TypeVar, TypeAlias, Generic, Union, Type
from ....units.base_classes.base_dimension import BaseDimension
from ....units.base_classes.base_unit import BaseUnit
from ....scalars.united_scalar import UnitedScalar
from ....scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ....scalars.united_scalar import UnitedScalar
from ....units.united import United
from ..base_array import BaseArray
from ..protocol_numerical_array import ProtocolNumericalArray
from abc import ABC
import h5py

PrimitiveType: TypeAlias = float|int|complex

UST = TypeVar("UST", bound=UnitedScalar)
UAT = TypeVar("UAT", bound=BaseArray)
UT = TypeVar("UT", bound=BaseUnit)
DT = TypeVar("DT", bound=BaseDimension)
PT = TypeVar("PT", bound=PrimitiveType)

@dataclass(frozen=True, slots=True)
class BaseUnitedArray(ABC, ProtocolNumericalArray[UST], BaseArray[UST], United[DT, UT], Generic[UAT, UST, UT, DT, PT]):

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

    def get_as_united_value(self, index: int) -> UST:
        match self.value_type:
            case float():
                value_float: float = self.canonical_np_array[index]
                return RealUnitedScalar(canonical_value=value_float, display_unit=self.display_unit, dimension=self.dimension)
            case complex():
                value_complex: complex = self.canonical_np_array[index]
                raise NotImplementedError("ComplexUnitedArrayTypes are not implemented yet.")
            case _:
                raise ValueError(f"Invalid value type: {self.value_type}")
            
    def get_as_numpy_skalar(self, index: int, unit: UT) -> PT:
        value: PT = self.canonical_np_array[index]
        return unit.from_canonical_value(value)
            
    def get_as_united_array(self, slice: slice) -> UAT:
        canonical_np_array: np.ndarray = self.canonical_np_array[slice]
        return self.create(canonical_np_array, self.dimension, self.display_unit)
    
    def get_as_numpy_array(self, slice: slice, unit: BaseUnit) -> np.ndarray:
        match self.value_type:
            case float():
                array: np.ndarray = self.canonical_np_array[slice]
                return unit.to_canonical_value(array).astype(self.value_type.value.dtype)
            case complex():
                array: np.ndarray = self.canonical_np_array[slice]
                return unit.to_canonical_value(array).astype(self.value_type.value.dtype)
            case _:
                raise ValueError(f"Invalid value type: {self.value_type}")
    
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
        if not self.dimension.compatible_to(other.dimension):
            raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
        match other:
            case UnitedScalar():
                array: np.ndarray = self.canonical_np_array + other.canonical_value
                return self.create(array, self.dimension, self.display_unit)
            case BaseUnitedArray():
                array: np.ndarray = self.canonical_np_array + other.canonical_np_array
                return self.create(array, self.dimension, self.display_unit)
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
        if not self.dimension.compatible_to(other.dimension):
            raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
        match other:
            case UnitedScalar():
                array: np.ndarray = self.canonical_np_array - other.canonical_value
                return self.create(array, self.dimension, self.display_unit)
            case BaseUnitedArray():
                array: np.ndarray = self.canonical_np_array - other.canonical_np_array
                return self.create(array, self.dimension, self.display_unit)
            case _:
                raise ValueError(f"Invalid other: {other}")

    @overload    
    def __rsub__(self, other: UAT) -> UAT:
        ...
    @overload
    def __rsub__(self, other: UST) -> UAT:
        ...
    def __rsub__(self, other: UAT|UST) -> UAT:
        if not self.dimension.compatible_to(other.dimension):
            raise ValueError(f"The unit dimension {self.dimension} is not compatible with the unit dimension {other.dimension}.")
        match other:
            case UnitedScalar():
                array: np.ndarray = other.canonical_value - self.canonical_np_array
                return self.create(array, self.dimension, self.display_unit)
            case BaseUnitedArray():
                array: np.ndarray = other.canonical_np_array - self.canonical_np_array
                return self.create(array, self.dimension, self.display_unit)
            case _:
                raise ValueError(f"Invalid other: {other}")
    
    @overload    
    def __mul__(self, other: PT) -> UAT:
        ...
    @overload
    def __mul__(self, other: UST) -> UAT:
        ...
    def __mul__(self, other: PT|UST) -> UAT:
        match other:
            case PrimitiveType():
                array: np.ndarray = self.canonical_np_array * other
                return self.create(array, self.dimension, self.display_unit)
            case UnitedScalar():
                array: np.ndarray = self.canonical_np_array * other.canonical_value
                dimension: DT = self.dimension + other.dimension
                return self.create(array, dimension, None)
            case _:
                raise ValueError(f"Invalid other: {other}")

    @overload    
    def __rmul__(self, other: PT) -> UAT:
        ...
    @overload
    def __rmul__(self, other: UST) -> UAT:
        ...
    def __rmul__(self, other: PT|UST) -> UAT:
        return self.__mul__(other)
    
    @overload
    def __truediv__(self, other: PT) -> UAT:
        ...
    @overload

    def __truediv__(self, other: UST) -> UAT:
        ...
    def __truediv__(self, other: UST|PT) -> UAT:
        match other:
            case PrimitiveType():
                array: np.ndarray = self.canonical_np_array / other
                return self.create(array, self.dimension, self.display_unit)
            case UnitedScalar():
                array: np.ndarray = self.canonical_np_array / other.canonical_value
                dimension: DT = self.dimension - other.dimension
                return self.create(array, dimension, None)
            case _:
                raise ValueError(f"Invalid other: {other}")
    
    @overload    
    def __rtruediv__(self, other: PT) -> UAT:
        ...
    @overload
    def __rtruediv__(self, other: UST) -> UAT:
        ...
    def __rtruediv__(self, other: UST|PT) -> UAT:
        match other:
            case PrimitiveType():
                array: np.ndarray = other / self.canonical_np_array
                return self.create(array, self.dimension, self.display_unit)
            case UnitedScalar():
                array: np.ndarray = other.canonical_value / self.canonical_np_array
                dimension: DT = other.dimension - self.dimension
                return self.create(array, dimension, None)

    def __neg__(self) -> UAT:
        array: np.ndarray = -1 * self.canonical_np_array
        return self.create(array, self.dimension, self.display_unit)
    
    def __abs__(self) -> UAT:
        array: np.ndarray = np.abs(self.canonical_np_array)
        return self.create(array, self.dimension, self.display_unit)
    
    def __len__(self) -> int:
        return len(self.canonical_np_array)
    
    def sum(self) -> UST:
        sum: PT = np.sum(self.canonical_np_array)
        return self.create(sum, self.dimension, self.display_unit)
    
    def mean(self) -> UST:
        mean: PT = np.mean(self.canonical_np_array)
        return self.create(mean, self.dimension, self.display_unit)
    
    def std(self) -> UST:
        std: PT = np.std(self.canonical_np_array)
        return self.create(std, self.dimension, self.display_unit)
    
    def min(self) -> UST:
        min: PT = np.min(self.canonical_np_array)
        return self.create(min, self.dimension, self.display_unit)
    
    def max(self) -> UST:
        max: PT = np.max(self.canonical_np_array)
        return self.create(max, self.dimension, self.display_unit)

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