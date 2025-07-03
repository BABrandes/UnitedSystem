import numpy as np
import pickle
from dataclasses import dataclass, field
from typing import Callable, Union, Type, Protocol, runtime_checkable, NamedTuple, overload, Tuple, cast
from enum import Enum
from typing import TypeAlias
from datetime import datetime
from .units.unit_quantity import CanonicalQuantity
from .units.unit import Unit
from .united_scalar import UnitedScalar, SimpleUnitedScalar
from .utils import JSONable
from abc import ABC, abstractmethod

SimpleUnitedArrayTypes: TypeAlias = Union[np.float64, np.float32, np.float16,]
@dataclass(frozen=True, slots=True)
class SimpleUnitedArrayValueType_Information(NamedTuple):
    name: str
    dtype: np.dtype
    simple_united_array_types: Type[SimpleUnitedArrayTypes]
    precision: int|None
class SimpleUnitedArrayValueType(Enum):
    value: SimpleUnitedArrayValueType_Information
FLOAT64 = SimpleUnitedArrayValueType_Information(name="float64", dtype=np.dtype("float64"), simple_united_array_types=np.float64, precision=64)
FLOAT32 = SimpleUnitedArrayValueType_Information(name="float32", dtype=np.dtype("float32"), simple_united_array_types=np.float32, precision=32)
FLOAT16 = SimpleUnitedArrayValueType_Information(name="float16", dtype=np.dtype("float16"), simple_united_array_types=np.float16, precision=16)

ComplexUnitedArrayTypes: TypeAlias = Union[np.complex64, np.complex128,]
@dataclass(frozen=True, slots=True)
class ComplexUnitedArrayValueType_Information(NamedTuple):
    name: str
    dtype: np.dtype
    complex_united_array_types: Type[ComplexUnitedArrayTypes]
    precision: int|None
class ComplexUnitedArrayValueType(Enum):
    value: ComplexUnitedArrayValueType_Information
COMPLEX64 =  ComplexUnitedArrayValueType_Information(name="complex64",  dtype=np.dtype("complex64"),  complex_united_array_types=np.complex64,  precision=64)
COMPLEX128 = ComplexUnitedArrayValueType_Information(name="complex128", dtype=np.dtype("complex128"), complex_united_array_types=np.complex128, precision=128)

@dataclass(frozen=True, slots=True)
class UnitedArray(ABC, JSONable):

    canonical_np_array: np.ndarray
    value_type: SimpleUnitedArrayValueType|ComplexUnitedArrayValueType
    canonical_quantity: CanonicalQuantity
    _display_unit: Unit|None = field(default=None, init=False, repr=False, compare=False, hash=False)

    @property
    def display_unit(self) -> Unit|None:
        return self._display_unit
    def set_display_unit(self, display_unit: Unit) -> None:
        if display_unit.compatible_to(self.canonical_quantity):
            raise ValueError(f"The display unit {display_unit} is not compatible with the canonical quantity {self.canonical_quantity}.")
        self.__setattr__("_display_unit", display_unit)

    def get_as_united_value(self, index: int) -> UnitedScalar:
        match self.value_type:
            case SimpleUnitedArrayValueType():
                value_float: float = self.canonical_np_array[index]
                return SimpleUnitedScalar(canonical_value=value_float, display_unit=self.display_unit, canonical_quantity=self.canonical_quantity)
            case ComplexUnitedArrayValueType():
                value_complex: complex = self.canonical_np_array[index]
                raise NotImplementedError("ComplexUnitedArrayTypes are not implemented yet.")
            case _:
                raise ValueError(f"Invalid value type: {self.value_type}")
            
    def get_as_numpy_skalar(self, index: int, unit: Unit) -> SimpleUnitedArrayTypes|ComplexUnitedArrayTypes:
        match self.value_type:
            case SimpleUnitedArrayValueType():
                value_float: SimpleUnitedArrayTypes = self.canonical_np_array[index]
                return unit.from_canonical_value(value_float)
            case ComplexUnitedArrayValueType():
                value_complex: ComplexUnitedArrayTypes = self.canonical_np_array[index]
                raise NotImplementedError("ComplexUnitedArrayTypes are not implemented yet.")
            case _:
                raise ValueError(f"Invalid value type: {self.value_type}")
            
    def get_as_united_array(self, slice: slice) -> "UnitedArray":
        if self.value_type is SimpleUnitedArrayValueType():
            canonical_np_array: np.ndarray = self.canonical_np_array[slice]
            return SimpleUnitedArray(canonical_np_array=canonical_np_array, value_type=self.value_type, canonical_quantity=self.canonical_quantity)
        elif self.value_type is ComplexUnitedArrayValueType():
            raise NotImplementedError("ComplexUnitedArrayTypes are not implemented yet.")
        else:
            raise ValueError(f"Invalid value type: {self.value_type}")
    
    def get_as_numpy_array(self, slice: slice, unit: Unit) -> np.ndarray:
        match self.value_type:
            case SimpleUnitedArrayValueType():
                array: np.ndarray = self.canonical_np_array[slice]
                return unit.to_canonical_value(array).astype(self.value_type.value.dtype)
            case ComplexUnitedArrayValueType():
                array: np.ndarray = self.canonical_np_array[slice]
                return unit.to_canonical_value(array).astype(self.value_type.value.dtype)
            case _:
                raise ValueError(f"Invalid value type: {self.value_type}")
    
    @overload
    def __getitem__(self, index_key: int) -> UnitedScalar:
        ...
    @overload
    def __getitem__(self, index_key: slice) -> "UnitedArray":
        ...
    def __getitem__(self, index_key: int|slice) -> "UnitedArray|UnitedScalar":
        match index_key:
            case int():
                return self.get_as_united_value(index_key)
            case slice():
                return self.get_as_united_array(index_key)
            case _:
                raise ValueError(f"Invalid index key: {index_key}")

    @overload
    @abstractmethod
    def __add__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __add__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    @abstractmethod
    def __add__(self, other: "UnitedArray|UnitedScalar") -> "UnitedArray":
        raise NotImplementedError("Addition of UnitedArray and UnitedValue is not implemented yet.")

    @overload    
    def __radd__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @overload
    def __radd__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    def __radd__(self, other: "UnitedArray|UnitedScalar") -> "UnitedArray":
        return self.__add__(other)
    
    @overload    
    @abstractmethod
    def __sub__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __sub__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    @abstractmethod
    def __sub__(self, other: "UnitedArray|UnitedScalar") -> "UnitedArray":
        raise NotImplementedError("Addition of UnitedArray and UnitedValue is not implemented yet.")

    @overload    
    def __rsub__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @overload
    def __rsub__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    def __rsub__(self, other: "UnitedArray|UnitedScalar") -> "UnitedArray":
        return self.__sub__(other)
    
    @overload    
    @abstractmethod
    def __mul__(self, other: float|int) -> "UnitedArray":
        ...
    @overload    
    @abstractmethod
    def __mul__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __mul__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @abstractmethod
    def __mul__(self, other: "UnitedArray|UnitedScalar|float|int") -> "UnitedArray":
        raise NotImplementedError("Multiplication of UnitedArray and UnitedValue is not implemented yet.")
    
    @overload    
    def __rmul__(self, other: float|int) -> "UnitedArray":
        ...
    @overload    
    def __rmul__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    @overload
    def __rmul__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    def __rmul__(self, other: "UnitedArray|UnitedScalar|float|int") -> "UnitedArray":
        return self.__mul__(other)
    
    @overload    
    @abstractmethod
    def __truediv__(self, other: float|int) -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __truediv__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __truediv__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @abstractmethod
    def __truediv__(self, other: "UnitedArray|UnitedScalar|float|int") -> "UnitedArray":
        raise NotImplementedError("Division of UnitedArray and UnitedValue is not implemented yet.")
    
    @overload    
    @abstractmethod 
    def __rtruediv__(self, other: float|int) -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __rtruediv__(self, other: UnitedScalar) -> "UnitedArray":
        ...
    @overload
    @abstractmethod
    def __rtruediv__(self, other: "UnitedArray") -> "UnitedArray":
        ...
    @abstractmethod
    def __rtruediv__(self, other: "UnitedArray|UnitedScalar|float|int") -> "UnitedArray":
        raise NotImplementedError("Division of UnitedArray and UnitedValue is not implemented yet.")

    @abstractmethod
    def __neg__(self) -> "UnitedArray":
        raise NotImplementedError("Negation of UnitedArray is not implemented yet.")
    
    def __abs__(self) -> "UnitedArray":
        ...
    
    def __len__(self) -> int:
        ...

    def to_json(self) -> dict[str, Any]:
        return {
            "canonical_np_array": self.canonical_np_array.tolist(),
            "value_type": self.value_type.value.name,
            "canonical_quantity": self.canonical_quantity.to_json()
        }
    
    @staticmethod
    def from_json(json: dict[str, Any]) -> "UnitedArray":
        return UnitedArray(canonical_np_array=np.array(json["canonical_np_array"]), value_type=SimpleUnitedArrayValueType(json["value_type"]), canonical_quantity=CanonicalQuantity.from_json(json["canonical_quantity"]))