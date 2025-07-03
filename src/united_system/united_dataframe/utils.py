from typing import Protocol, Type, runtime_checkable, NamedTuple, Any, Callable
from ..units.unit import Unit
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from pandas import Dtype
from ..scalars.united_scalar import UnitedScalar
from ..arrays.utils import ArrayLike
from ..scalars.real_united_scalar import RealUnitedScalar
from ..scalars.complex_united_scalar import ComplexUnitedScalar
from ..arrays.real_united_array import RealUnitedArray
from ..arrays.complex_united_array import ComplexUnitedArray
from ..arrays.string_array import StringArray
from ..arrays.int_array import IntArray
from ..arrays.float_array import FloatArray
from ..arrays.bool_array import BoolArray
from ..arrays.timestamp_array import TimestampArray
from pandas import Timestamp
import numpy as np
import math

@runtime_checkable
class Column_Key(Protocol):
    def to_string(self) -> str:
        ...
    @classmethod
    def from_string(cls: Type['Column_Key'], string: str) -> 'Column_Key':
        ...

@dataclass(frozen=True, slots=True)
class ColumnTypeInformation(NamedTuple):
    name: str
    has_unit: bool
    scalar_type: type[UnitedScalar]
    array_type: type[ArrayLike]
    dataframe_storage_type: Dtype
    array_storage_type: Dtype|type
    numpy_storage_options: np.dtype
    none_value: Any
    precision: int|None

class ColumnType(Enum):
    value: ColumnTypeInformation    
    REAL_NUMBER_64 = ColumnTypeInformation(         name="Real United Numbers (64bit)",     has_unit=True,  scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_types=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options={np.float64, np.float32, np.float16},     none_value=pd.NA,       precision=64)
    REAL_NUMBER_32 = ColumnTypeInformation(         name="Real United Numbers (32bit)",     has_unit=True,  scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_types=pd.Float32Dtype(),  array_storage_type=float,       numpy_storage_options={np.float32, np.float64, np.float16},     none_value=pd.NA,       precision=32)
    COMPLEX_NUMBER_128 = ColumnTypeInformation(     name="Complex United Numbers (128bit)", has_unit=True,  scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_types=np.complex128,      array_storage_type=complex,     numpy_storage_options={np.complex128, np.complex64},            none_value=math.nan+1j, precision=128)
    STRING = ColumnTypeInformation(                 name="Strings",                         has_unit=False, scalar_type=str,                    array_type=StringArray,         dataframe_storage_types=pd.StringDtype,     array_storage_type=str,         numpy_storage_options={np.str_},                                none_value=pd.NA,       precision=None)
    INTEGER_64 = ColumnTypeInformation(             name="Integers",                        has_unit=False, scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int64Dtype(),    array_storage_type=int,         numpy_storage_options={np.int64, np.int32, np.int16, np.int8},  none_value=pd.NA,       precision=64)
    INTEGER_32 = ColumnTypeInformation(             name="Integers",                        has_unit=False, scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int32Dtype(),    array_storage_type=int,         numpy_storage_options={np.int32, np.int64, np.int16, np.int8},  none_value=pd.NA,       precision=32)
    INTEGER_16 = ColumnTypeInformation(             name="Integers",                        has_unit=False, scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int16Dtype(),    array_storage_type=int,         numpy_storage_options={np.int16, np.int64, np.int32, np.int8},  none_value=pd.NA,       precision=16)
    INTEGER_8 = ColumnTypeInformation(              name="Integers",                        has_unit=False, scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int8Dtype(),     array_storage_type=int,         numpy_storage_options={np.int8, np.int64, np.int32, np.int16},  none_value=pd.NA,       precision=8)
    FLOAT_64 = ColumnTypeInformation(               name="Floats (64bit)",                  has_unit=True,  scalar_type=float,                  array_type=FloatArray,          dataframe_storage_types=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options={np.float64, np.float32, np.float16},     none_value=pd.NA,       precision=64)
    FLOAT_32 = ColumnTypeInformation(               name="Floats (32bit)",                  has_unit=True,  scalar_type=float,                  array_type=FloatArray,          dataframe_storage_types=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options={np.float32, np.float64, np.float16},     none_value=pd.NA,       precision=32)
    COMPLEX_128 = ColumnTypeInformation(            name="Complex (128bit)",                has_unit=True,  scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_types=np.complex128,      array_storage_type=complex,     numpy_storage_options={np.complex128, np.complex64},            none_value=math.nan+1j, precision=128)
    BOOL = ColumnTypeInformation(                   name="Bools",                           has_unit=False, scalar_type=bool,                   array_type=BoolArray,           dataframe_storage_types=pd.BooleanDtype(),  array_storage_type=bool,        numpy_storage_options={np.bool_},                               none_value=pd.NA,       precision=None)
    TIMESTAMP = ColumnTypeInformation(              name="Timestamps",                      has_unit=False, scalar_type=Timestamp,              array_type=TimestampArray,      dataframe_storage_types=pd.Timestamp(),     array_storage_type=Timestamp,   numpy_storage_options={np.datetime64},                          none_value=pd.NA,       precision=None)

class Series_With_Unit(NamedTuple):
    series: pd.Series
    unit: Unit

SIMPLE_UNITED_FORMATTER: Callable[[Column_Key|str, Unit, ColumnType], str] = lambda name, unit, _: f"{name} [{unit}]" if unit != None else f"{name} [-]"

# @dataclass(frozen=True, slots=True)
# class Value_Type_Information(NamedTuple):
#     name: str
#     corresponding_pandas_type: Dtype
#     corresponding_python_type: type
#     corresponding_numpy_dtype: np.dtype
#     corresponding_united_array_value_type: United_Array_Value_Type
#     corresponding_UnitedScalar_type: type[UnitedValueValueType]
#     corresponding_python_na_value: None|float
#     corresponding_numpy_na_value: float|np.float32|np.datetime64
#     corresponding_UnitedScalar_na_value: None|UnitedValueValueType
#     corresponding_UnitedScalar_na_value_type: None|type[UnitedValueValueType]
#     is_numeric: bool
#     is_non_numeric: bool
#     precision: int|None


# class Value_Type(Enum):
#     value: Value_Type_Information

#     FLOAT64 = Value_Type_Information(   name="float64",  corresponding_python_type=float,    corresponding_pandas_type=pd.Float64Dtype(),          corresponding_numpy_dtype=np.dtype("float64"),        corresponding_united_array_value_type=United_Array_Value_Type.FLOAT64,    corresponding_UnitedScalar_type=float,    is_numeric=True,  is_non_numeric=False, precision=64,   corresponding_python_na_value=math.nan, corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=math.nan, corresponding_UnitedScalar_na_value_type=float)
#     FLOAT32 = Value_Type_Information(   name="float32",  corresponding_python_type=float,    corresponding_pandas_type=pd.Float32Dtype(),          corresponding_numpy_dtype=np.dtype("float32"),        corresponding_united_array_value_type=United_Array_Value_Type.FLOAT32,    corresponding_UnitedScalar_type=float,    is_numeric=True,  is_non_numeric=False, precision=32,   corresponding_python_na_value=math.nan, corresponding_numpy_na_value=np.float32(np.nan),   corresponding_UnitedScalar_na_value=math.nan, corresponding_UnitedScalar_na_value_type=float)
#     INT64 = Value_Type_Information(     name="int64",    corresponding_python_type=int,      corresponding_pandas_type=pd.Int64Dtype(),            corresponding_numpy_dtype=np.dtype("int64"),          corresponding_united_array_value_type=United_Array_Value_Type.INT64,      corresponding_UnitedScalar_type=float,    is_numeric=True,  is_non_numeric=False, precision=64,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)
#     INT32 = Value_Type_Information(     name="int32",    corresponding_python_type=int,      corresponding_pandas_type=pd.Int32Dtype(),            corresponding_numpy_dtype=np.dtype("int32"),          corresponding_united_array_value_type=United_Array_Value_Type.INT32,      corresponding_UnitedScalar_type=float,    is_numeric=True,  is_non_numeric=False, precision=32,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)
#     INT16 = Value_Type_Information(     name="int16",    corresponding_python_type=int,      corresponding_pandas_type=pd.Int16Dtype(),            corresponding_numpy_dtype=np.dtype("int16"),          corresponding_united_array_value_type=United_Array_Value_Type.INT16,      corresponding_UnitedScalar_type=float,    is_numeric=True,  is_non_numeric=False, precision=16,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)
#     INT8 = Value_Type_Information(      name="int8",     corresponding_python_type=int,      corresponding_pandas_type=pd.Int8Dtype(),             corresponding_numpy_dtype=np.dtype("int8"),           corresponding_united_array_value_type=United_Array_Value_Type.INT8,       corresponding_UnitedScalar_type=float,    is_numeric=True,  is_non_numeric=False, precision=8,    corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)
#     BOOLEAN = Value_Type_Information(   name="bool",     corresponding_python_type=bool,     corresponding_pandas_type=pd.BooleanDtype(),          corresponding_numpy_dtype=np.dtype("bool"),           corresponding_united_array_value_type=United_Array_Value_Type.BOOLEAN,    corresponding_UnitedScalar_type=bool,     is_numeric=False, is_non_numeric=True,  precision=None, corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)
#     STRING = Value_Type_Information(    name="string",   corresponding_python_type=str,      corresponding_pandas_type=pd.StringDtype(),           corresponding_numpy_dtype=np.dtype("string"),         corresponding_united_array_value_type=United_Array_Value_Type.STRING,     corresponding_UnitedScalar_type=str,      is_numeric=False, is_non_numeric=True,  precision=None, corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)
#     DATETIME64 = Value_Type_Information(name="datetime", corresponding_python_type=datetime, corresponding_pandas_type=np.dtype("datetime64[ns]"), corresponding_numpy_dtype=np.dtype("datetime64[ns]"), corresponding_united_array_value_type=United_Array_Value_Type.DATETIME64, corresponding_UnitedScalar_type=datetime, is_numeric=True,  is_non_numeric=False,  precision=64,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.datetime64("NaT"), corresponding_UnitedScalar_na_value=None,     corresponding_UnitedScalar_na_value_type=None)

#     @classmethod
#     def find_value_type_by_UnitedScalar_type(cls, UnitedScalar_type: United_Array_Value_Type) -> "Value_Type":
#         """
#         Find the value type by united value type.

#         Args:
#             UnitedScalar_type (United_Array_Value_Type): The united value type to find the value type for

#         Returns:
#             Value_Type: The value type for the given united value type
#         """
#         for value_type in Value_Type:
#             if value_type.value.corresponding_united_array_value_type == UnitedScalar_type:
#                 return value_type
#         raise ValueError(f"No value type found for united value type {UnitedScalar_type}.")
    
#     @classmethod
#     def find_value_type_by_pandas_type(cls, pandas_type: Dtype) -> "Value_Type":
#         """
#         Find the value type by pandas type.

#         Args:
#             pandas_type (Dtype): The pandas type to find the value type for

#         Returns:
#             Value_Type: The value type for the given pandas type
#         """
#         for value_type in cls:
#             vtype = value_type.value.corresponding_pandas_type
#             # Use string representation matching for extension types and fallback to equality for others
#             if str(vtype) == str(pandas_type):
#                 return value_type
#         raise ValueError(f"No value type found for pandas type {pandas_type}.")
    
#     @classmethod
#     def find_value_type_by_numpy_dtype(cls, numpy_dtype: np.dtype) -> "Value_Type":
#         """
#         Find the value type by numpy dtype.

#         Args:
#             numpy_dtype (np.dtype): The numpy dtype to find the value type for

#         Returns:
#             Value_Type: The value type for the given numpy dtype
#         """
#         dtype: np.dtype = np.dtype(numpy_dtype)  # normalize dtype
#         for value_type in cls:
#             if value_type.value.corresponding_numpy_dtype == dtype:
#                 return value_type
#         raise ValueError(f"No value type found for numpy dtype {numpy_dtype}.")
    
#     def coerce_to_type(self, value: float | int | str | bool | datetime) -> float | int | str | bool | datetime:
#         try:
#             return self.value.corresponding_python_type(value)
#         except Exception as e:
#             raise TypeError(f"Cannot coerce {value!r} to {self.value.name}") from e
        
#     def has_python_na_value(self) -> bool:
#         return self.value.corresponding_python_na_value is not None
    
#     def has_numpy_na_value(self) -> bool:
#         return self.value.corresponding_numpy_na_value is not None
    
#     def has_UnitedScalar_na_value(self) -> bool:
#         return self.value.corresponding_UnitedScalar_na_value is not None
    
#     def has_UnitedScalar_na_value_type(self) -> bool:
#         return self.value.corresponding_UnitedScalar_na_value_type is not None