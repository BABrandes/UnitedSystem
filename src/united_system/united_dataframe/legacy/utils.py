from typing import Protocol, Type, runtime_checkable, NamedTuple, Any, Callable, Generic, TypeVar, TypeAlias, Literal, overload
from ...units.base_classes.base_unit import BaseUnit
import pandas as pd
from dataclasses import dataclass
from enum import Enum
from pandas._typing import Dtype
from ...scalars.united_scalar import UnitedScalar
from ...arrays.utils import ArrayLike
from ...scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ...scalars.complex_united_scalar.complex_united_scalar import ComplexUnitedScalar
from ...arrays.real_united_array.real_united_array import RealUnitedArray
from ...arrays.complex_united_array import ComplexUnitedArray
from ...arrays.string_array import StringArray
from ...arrays.int_array import IntArray
from ...arrays.float_array import FloatArray
from ...arrays.bool_array import BoolArray
from ...arrays.timestamp_array import TimestampArray
from pandas import Timestamp
import numpy as np
import math
from ...units.unit_quantity import UnitQuantity
# from ..united_dataframe.united_dataframe import UnitedDataframe  # Avoid circular import
from ...units.simple.simple_unit import SimpleUnit
from .column_type import ColumnType

@runtime_checkable
class ColumnKey(Protocol):
    def to_string(self) -> str:
        ...
    @classmethod
    def from_string(cls: Type['ColumnKey'], string: str) -> 'ColumnKey':
        ...

@dataclass(frozen=True, slots=True)
class ColumnInformation():
    unit_quantity: UnitQuantity|None
    column_type: ColumnType
    display_unit: BaseUnit|None

    def __postinit__(self):
        if self.unit_quantity is None:
            if self.display_unit is not None:
                raise ValueError(f"When the unit quantity is None, the display unit must also be None")
        else:
            if self.display_unit is None:
                self.display_unit = self.unit_quantity.canonical_unit()
            if not self.display_unit.compatible_to(self.unit_quantity):
                raise ValueError(f"Display unit {self.display_unit} is not compatible with unit quantity {self.unit_quantity}")

    def internal_dataframe_column_name(self, column_key: ColumnKey|str, internal_column_name_formatter: "InternalDataFrameNameFormatter[CK]" = "SIMPLE_INTERNAL_NAME_FORMATTER") -> str:
        return internal_column_name_formatter.create_internal_dataframe_column_name(column_key, self)

    @classmethod
    def create(
        cls,
        unit_quantity: UnitQuantity|None,
        column_type: ColumnType,
        display_unit: BaseUnit|None=None) -> "ColumnInformation":
        return cls(unit_quantity, column_type, display_unit)

CK = TypeVar("CK", bound=ColumnKey|str, default=str)
class InternalDataFrameNameFormatter(Protocol, Generic[CK]):
    def create_internal_dataframe_column_name(self, column_key: CK, column_information: ColumnInformation[CK]) -> str:
        ...
    @classmethod
    def retrieve_from_internal_dataframe_column_name(cls, internal_dataframe_column_name: str, dtype: Dtype, column_key_constructor: Callable[[str], CK]|None=None) -> tuple[CK, ColumnInformation[CK]]:
        ...

def x(internal_dataframe_column_name: str, dtype: Dtype, column_key_constructor: Callable[[str], CK]|None=None) -> tuple[CK, ColumnInformation[CK]]:
    # Find the indices of '[' and ']' in the internal_dataframe_column_name, looking from the end of the string
    internal_dataframe_column_name = internal_dataframe_column_name.strip()
    index_bracket_close: int = internal_dataframe_column_name.rfind(']')
    index_bracket_open: int = internal_dataframe_column_name[index_bracket_close:].rfind('[')
    display_unit: str = internal_dataframe_column_name[index_bracket_open+1:index_bracket_close]
    # Make sure there is a space before the '['
    if index_bracket_open > 0 and internal_dataframe_column_name[index_bracket_open-1] != ' ':
        raise ValueError(f"Invalid internal dataframe column name: {internal_dataframe_column_name}")
    # Get the rest of ths string, but without space
    column_key_str = internal_dataframe_column_name[:index_bracket_open-1]
    if display_unit == "-":
        display_unit: BaseUnit|None = None
    else:
        display_unit: BaseUnit = SimpleUnit.parse_string(display_unit)
    if CK == str:
        column_key: CK = column_key_str
    else:
        column_key: CK = column_key_constructor
    column_information: ColumnInformation = ColumnInformation.create(
        unit_quantity=display_unit.unit_quantity() if display_unit is not None else None,
        column_type=ColumnType.value,
        display_unit=display_unit if display_unit is not None else None
    )
    return column_key, column_information
SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER: InternalDataFrameNameFormatter[CK] = InternalDataFrameNameFormatter[CK](
    create_internal_dataframe_column_name=lambda column_key, column_information: f"{column_key} [{column_information.display_unit}]" if column_information.display_unit != None else f"{column_key} [-]",
    retrieve_from_internal_dataframe_column_name=lambda internal_dataframe_column_name, dtype, column_key_constructor: x(internal_dataframe_column_name, dtype, column_key_constructor))



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