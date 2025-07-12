from typing import TypeAlias, Literal, overload, Any, NamedTuple
from enum import Enum
import math

import pandas as pd
from pandas._typing import Dtype
from pandas import Timestamp
import numpy as np

from ...unit import Unit
from ...real_united_scalar import RealUnitedScalar
from ...complex_united_scalar import ComplexUnitedScalar
from ...real_united_array import RealUnitedArray
from ...complex_united_array import ComplexUnitedArray
from ...string_array import StringArray
from ...complex_array import ComplexArray
from ...int_array import IntArray
from ...float_array import FloatArray
from ...bool_array import BoolArray
from ...timestamp_array import TimestampArray
from ..scalars.base_scalar import BaseScalar
from ..units.united import United

LOWLEVEL_TYPE: TypeAlias = float|complex|str|bool|int|Timestamp
PANDAS_SERIES_TYPE: TypeAlias = float|int|complex|bool|str|Timestamp
SCALAR_TYPE: TypeAlias = LOWLEVEL_TYPE|BaseScalar|str|bool|int|Timestamp
NUMERIC_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar|int|float|complex|Timestamp
NUMERIC_ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray|IntArray|FloatArray|TimestampArray
NUMERIC_TYPE: TypeAlias = NUMERIC_SCALAR_TYPE|NUMERIC_ARRAY_TYPE
ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray|ComplexArray
UNITED_ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray
UNITED_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar
ARRAY_STORAGE_TYPE: TypeAlias = float|complex|str|int|bool|Timestamp
NUMPY_STORAGE_TYPE: TypeAlias = np.float64|np.float32|np.float16|np.complex128|np.complex64|np.int64|np.int32|np.int16|np.int8|np.bool_|np.datetime64|np.str_

class ColumnTypeInformation(NamedTuple):
    name: str
    has_unit: bool
    python_scalar_type: type[LOWLEVEL_TYPE]
    scalar_type: type[SCALAR_TYPE]
    array_type: type[ARRAY_TYPE]
    dataframe_storage_type: Dtype
    array_storage_type: type[ARRAY_STORAGE_TYPE]
    numpy_storage_options: list[type[NUMPY_STORAGE_TYPE]]
    missing_values_in_dataframe: Any
    non_values_in_scalar: Any
    precision: int|None

class ColumnType(Enum):
    value: ColumnTypeInformation # type: ignore[assignment]

    REAL_NUMBER_64 = ColumnTypeInformation(         name="Real United Numbers (64bit)",     has_unit=True,  python_scalar_type=float,         scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_type=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options=[np.float64, np.float32, np.float16],     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=64)
    REAL_NUMBER_32 = ColumnTypeInformation(         name="Real United Numbers (32bit)",     has_unit=True,  python_scalar_type=float,         scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_type=pd.Float32Dtype(),  array_storage_type=float,       numpy_storage_options=[np.float32, np.float64, np.float16],     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=32)
    COMPLEX_NUMBER_128 = ColumnTypeInformation(     name="Complex United Numbers (128bit)", has_unit=True,  python_scalar_type=complex,       scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_type=np.complex128,      array_storage_type=complex,     numpy_storage_options=[np.complex128, np.complex64],            missing_values_in_dataframe=math.nan+1j, non_values_in_scalar=math.nan+1j,   precision=128)
    STRING = ColumnTypeInformation(                 name="Strings",                         has_unit=False, python_scalar_type=str,           scalar_type=str,                    array_type=StringArray,         dataframe_storage_type=pd.StringDtype(),   array_storage_type=str,         numpy_storage_options=[np.str_],                                missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=None)
    INTEGER_64 = ColumnTypeInformation(             name="Integers (64bit)",                has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int64Dtype(),    array_storage_type=int,         numpy_storage_options=[np.int64, np.int32, np.int16, np.int8],  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=64)
    INTEGER_32 = ColumnTypeInformation(             name="Integers (32bit)",                has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int32Dtype(),    array_storage_type=int,         numpy_storage_options=[np.int32, np.int64, np.int16, np.int8],  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=32)
    INTEGER_16 = ColumnTypeInformation(             name="Integers (16bit)",                has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int16Dtype(),    array_storage_type=int,         numpy_storage_options=[np.int16, np.int64, np.int32, np.int8],  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=16)
    INTEGER_8 = ColumnTypeInformation(              name="Integers (8bit)",                 has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int8Dtype(),     array_storage_type=int,         numpy_storage_options=[np.int8, np.int64, np.int32, np.int16],  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=8)
    FLOAT_64 = ColumnTypeInformation(               name="Floats (64bit)",                  has_unit=True,  python_scalar_type=float,         scalar_type=float,                  array_type=FloatArray,          dataframe_storage_type=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options=[np.float64, np.float32, np.float16],     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=64)
    FLOAT_32 = ColumnTypeInformation(               name="Floats (32bit)",                  has_unit=True,  python_scalar_type=float,         scalar_type=float,                  array_type=FloatArray,          dataframe_storage_type=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options=[np.float32, np.float64, np.float16],     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=32)
    COMPLEX_128 = ColumnTypeInformation(            name="Complex (128bit)",                has_unit=True,  python_scalar_type=complex,       scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_type=np.complex128,      array_storage_type=complex,     numpy_storage_options=[np.complex128, np.complex64],            missing_values_in_dataframe=math.nan+1j, non_values_in_scalar=math.nan+1j,   precision=128)
    BOOL = ColumnTypeInformation(                   name="Bools",                           has_unit=False, python_scalar_type=bool,          scalar_type=bool,                   array_type=BoolArray,           dataframe_storage_type=pd.BooleanDtype(),  array_storage_type=bool,        numpy_storage_options=[np.bool_],                               missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=None)
    TIMESTAMP = ColumnTypeInformation(              name="Timestamps",                      has_unit=False, python_scalar_type=Timestamp,     scalar_type=Timestamp,              array_type=TimestampArray,      dataframe_storage_type=pd.Timestamp,       array_storage_type=Timestamp,   numpy_storage_options=[np.datetime64],                          missing_values_in_dataframe=pd.NaT,      non_values_in_scalar=None,          precision=None)

    @property
    def has_unit(self) -> bool:
        return self.value.has_unit
    
    @property
    def is_numeric(self) -> bool:
        return self in {ColumnType.REAL_NUMBER_64, ColumnType.REAL_NUMBER_32, ColumnType.FLOAT_64, ColumnType.FLOAT_32, ColumnType.COMPLEX_128, ColumnType.COMPLEX_NUMBER_128, ColumnType.INTEGER_64, ColumnType.INTEGER_32, ColumnType.INTEGER_16, ColumnType.INTEGER_8, ColumnType.TIMESTAMP}
    
    @property
    def is_not_numeric(self) -> bool:
        return not self.is_numeric
    
    # ------------ Get values for dataframe ------------

    def get_value_for_dataframe(self, value: SCALAR_TYPE|None, unit_in_dataframe: Unit|None = None) -> Any:
        """
        Get the value from a scalar for the dataframe.

        Args:
            value (SCALAR_TYPE|None): The value to get the value for the dataframe.
            unit_in_dataframe (Unit|None): The unit in the dataframe. If the column has a unit, this is required.

        Returns:
            Any: The value for the dataframe.
        """
        if value is None:
            return pd.NA
        else:
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    if not isinstance(value, RealUnitedScalar):
                        raise ValueError(f"Value {value} is not a RealUnitedScalar.")
                    if unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is required for RealUnitedScalar.")
                    return unit_in_dataframe.from_canonical_value(value.canonical_value)
                case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    if not isinstance(value, float|int):
                        raise ValueError(f"Value {value} is not a float.")
                    if not unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is not allowed for float.")
                    return float(value)
                case ColumnType.COMPLEX_NUMBER_128:
                    if not isinstance(value, ComplexUnitedScalar):
                        raise ValueError(f"Value {value} is not a ComplexUnitedScalar.")
                    if unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is required for ComplexUnitedScalar.")
                    raise NotImplementedError(f"ComplexUnitedScalar is not implemented.")
                case ColumnType.COMPLEX_128:
                    if not isinstance(value, complex):
                        raise ValueError(f"Value {value} is not a ComplexUnitedScalar.")
                    if not unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is not allowed for ComplexUnitedScalar.")
                    return complex(value)
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    if not isinstance(value, int):
                        raise ValueError(f"Value {value} is not an int.")
                    if not unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is not allowed for int.")
                    return int(value)
                case ColumnType.STRING:
                    if not isinstance(value, str):
                        raise ValueError(f"Value {value} is not a str.")
                    if not unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is not allowed for str.")
                    return str(value)
                case ColumnType.BOOL:
                    if not isinstance(value, bool):
                        raise ValueError(f"Value {value} is not a bool.")
                    if not unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is not allowed for bool.")
                    return bool(value)
                case ColumnType.TIMESTAMP:
                    if not isinstance(value, Timestamp):
                        raise ValueError(f"Value {value} is not a Timestamp.")
                    if not unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is not allowed for Timestamp.")
                    return Timestamp(value)

    def get_values_for_dataframe(self, values: ARRAY_TYPE, dataframe_unit: Unit|None) -> pd.Series: # type: ignore[no-any-return]
        """
        Get the values from an array for the dataframe.

        Args:
            values (ARRAY_TYPE): The values to get the values for the dataframe.
            dataframe_unit (Unit|None): The unit in the dataframe. If the column has a unit, this is required.

        Returns:
            pd.Series: The values for the dataframe.
        """

        dtype: Dtype = self.value.dataframe_storage_type
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if not isinstance(values, RealUnitedArray):
                    raise ValueError(f"Value {values} is not a RealUnitedArray.")
                if dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is required for RealUnitedArray.")
                np_array: np.ndarray = dataframe_unit.from_canonical_value(values.canonical_np_array)
                return pd.Series(np_array, dtype=dtype)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                if not isinstance(values, FloatArray):
                    raise ValueError(f"Value {values} is not a FloatArray.")
                if not dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for FloatArray.")
                return pd.Series(values, dtype=dtype) # type: ignore[no-any-return]
            case ColumnType.COMPLEX_NUMBER_128:
                if not isinstance(values, ComplexUnitedArray):
                    raise ValueError(f"Value {values} is not a ComplexUnitedArray.")
                if dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is required for ComplexUnitedArray.")
                raise NotImplementedError(f"ComplexUnitedArray is not implemented.")
            case ColumnType.COMPLEX_128:
                if not isinstance(values, complex):
                    raise ValueError(f"Value {values} is not a complex.")
                if not dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for complex.")
                return pd.Series(values, dtype=dtype)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                if not isinstance(values, IntArray):
                    raise ValueError(f"Value {values} is not an IntArray.")
                if not dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for IntArray.")
                return pd.Series(values, dtype=dtype) # type: ignore[no-any-return]
            case ColumnType.STRING:
                if not isinstance(values, StringArray):
                    raise ValueError(f"Value {values} is not a StringArray.")
                if not dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for StringArray.")
                return pd.Series(values, dtype=dtype) # type: ignore[no-any-return]
            case ColumnType.BOOL:
                if not isinstance(values, BoolArray):
                    raise ValueError(f"Value {values} is not a BoolArray.")
                if not dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for BoolArray.")
                return pd.Series(values, dtype=dtype) # type: ignore[no-any-return]
            case ColumnType.TIMESTAMP:
                if not isinstance(values, TimestampArray):
                    raise ValueError(f"Value {values} is not a TimestampArray.")
                if not dataframe_unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for TimestampArray.")
                return pd.Series(values, dtype=dtype) # type: ignore[no-any-return]
            
    # ------------ Get values from dataframe ------------

    def get_primitive_value_from_dataframe(self, value: Any) -> float|complex|str|bool|int|Timestamp:
        """
        Get the python values from a dataframe.

        Args:
            value (Any): The value to get the python values from.

        Returns:
            float|complex|str|bool|int|Timestamp: The python values from the dataframe.
        """

        if pd.isna(value):
            if self.value.missing_values_in_dataframe is None:
                raise ValueError(f"Missing values are not allowed for {self.name}.")
            return self.value.missing_values_in_dataframe
        
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                return float(value)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return float(value)
            case ColumnType.COMPLEX_NUMBER_128:
                return complex(value)
            case ColumnType.COMPLEX_128:
                return complex(value)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return int(value)
            case ColumnType.STRING:
                return str(value)
            case ColumnType.BOOL:
                return bool(value)
            case ColumnType.TIMESTAMP:
                return Timestamp(value)
            
    def get_scalar_value_from_dataframe(self, value: LOWLEVEL_TYPE, unit: Unit|None = None) -> SCALAR_TYPE:
        """
        Get the scalar value from a dataframe.
        """

        if pd.isna(value):
            if self.value.missing_values_in_dataframe is None:
                raise ValueError(f"Missing values are not allowed for {self.name}.")
            return self.value.missing_values_in_dataframe
        
        if self.has_unit:
            if unit is None:
                raise ValueError(f"Unit is required for {self.name}.")

        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if not isinstance(value, float|int):
                    raise ValueError(f"Value {value} is not a float or int.")
                if unit is None:
                    raise ValueError(f"Unit is required for {self.name}.")
                return RealUnitedScalar.create_from_value_and_unit(value, unit)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                if not isinstance(value, float|int):
                    raise ValueError(f"Value {value} is not a float or int.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for float or int.")
                return float(value)
            case ColumnType.COMPLEX_NUMBER_128:
                if not isinstance(value, complex):
                    raise ValueError(f"Value {value} is not a complex.")
                if unit is None:
                    raise ValueError(f"Unit is required for {self.name}.")
                raise NotImplementedError(f"ComplexUnitedScalar is not implemented.")
            case ColumnType.COMPLEX_128:
                if not isinstance(value, complex):
                    raise ValueError(f"Value {value} is not a complex.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for complex.")
                return complex(value)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                if not isinstance(value, int):
                    raise ValueError(f"Value {value} is not an int.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for int.")
                return int(value)
            case ColumnType.STRING:
                if not isinstance(value, str):
                    raise ValueError(f"Value {value} is not a str.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for str.")
                return str(value)
            case ColumnType.BOOL:
                if not isinstance(value, bool):
                    raise ValueError(f"Value {value} is not a bool.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for bool.")
                return bool(value)
            case ColumnType.TIMESTAMP:
                if not isinstance(value, Timestamp):
                    raise ValueError(f"Value {value} is not a Timestamp.")
                return Timestamp(value)
            
    def get_numpy_array_from_dataframe(self, pandas_series: pd.Series) -> np.ndarray[Any, Any]: # type: ignore[reportUnknownReturnType]
        """
        Get the numpy array from a dataframe.
        """
        if pandas_series.isna().any(): # type: ignore[reportUnknownReturnType]
            if self.value.missing_values_in_dataframe is None:
                raise ValueError(f"Missing values are not allowed for {self.name}.")
            
        pandas_series = pandas_series.fillna(self.value.missing_values_in_dataframe) # type: ignore[reportUnknownReturnType]
        
        match self:
            case ColumnType.REAL_NUMBER_64:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.float64) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.REAL_NUMBER_32:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.float32) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.FLOAT_64:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.float64) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.FLOAT_32:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.float32) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.COMPLEX_NUMBER_128:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.complex128) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.COMPLEX_128:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.complex128) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.INTEGER_64:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.int64) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.INTEGER_32:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.int32) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.INTEGER_16:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.int16) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.INTEGER_8:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.int8) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.STRING:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.str_) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.BOOL:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.bool_) # type: ignore[reportUnknownReturnType]
                return numpy_array
            case ColumnType.TIMESTAMP:
                numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=np.datetime64) # type: ignore[reportUnknownReturnType]
                return numpy_array
            
    def get_array_from_dataframe(self, pandas_series: pd.Series, unit: Unit|None = None) -> ARRAY_TYPE: # type: ignore[reportUnknownReturnType]
        """
        Get the array from a dataframe.
        """
        if pandas_series.isna().any(): # type: ignore[reportUnknownReturnType]
            if self.value.missing_values_in_dataframe is None:
                raise ValueError(f"Missing values are not allowed for {self.name}.")
            
        pandas_series = pandas_series.fillna(self.value.missing_values_in_dataframe) # type: ignore[reportUnknownReturnType]
        
        if self.has_unit:
            if unit is None:
                raise ValueError(f"Unit is required for {self.name}.")

        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                return RealUnitedArray(pandas_series.to_numpy(dtype=np.float64), unit) # type: ignore[reportUnknownReturnType]
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return FloatArray(pandas_series.to_numpy(dtype=np.float64)) # type: ignore[reportUnknownReturnType]
            case ColumnType.COMPLEX_NUMBER_128:
                raise NotImplementedError(f"ComplexUnitedArray is not implemented.")
            case ColumnType.COMPLEX_128:
                return ComplexArray(pandas_series.to_numpy(dtype=np.complex128)) # type: ignore[reportUnknownReturnType]
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return IntArray(pandas_series.to_numpy(dtype=np.int64)) # type: ignore[reportUnknownReturnType]
            case ColumnType.STRING:
                return StringArray(pandas_series.to_numpy(dtype=np.str_)) # type: ignore[reportUnknownReturnType]
            case ColumnType.BOOL:
                return BoolArray(pandas_series.to_numpy(dtype=np.bool_)) # type: ignore[reportUnknownReturnType]
            case ColumnType.TIMESTAMP:
                return TimestampArray(pandas_series.to_numpy(dtype=np.datetime64)) # type: ignore[reportUnknownReturnType]
            
    def get_list_from_dataframe(self, pandas_series: pd.Series) -> list[float]|list[complex]|list[int]|list[str]|list[bool]|list[Timestamp]: # type: ignore[reportUnknownReturnType]
        """
        Get the array from a dataframe.
        """
        if pandas_series.isna().any(): # type: ignore[reportUnknownReturnType]
            if self.value.missing_values_in_dataframe is None:
                raise ValueError(f"Missing values are not allowed for {self.name}.")
            
        pandas_series = pandas_series.fillna(self.value.missing_values_in_dataframe) # type: ignore[reportUnknownReturnType]
        
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.COMPLEX_NUMBER_128:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.COMPLEX_128:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.STRING:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.BOOL:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            case ColumnType.TIMESTAMP:
                return pandas_series.to_list() # type: ignore[reportUnknownReturnType]
            
    # ------------ Infer the column type from a pandas Series ------------
                
    @classmethod
    def from_dtype(cls, dtype: pd.Series | pd.api.extensions.ExtensionDtype | np.dtype[Any] | str, has_unit: bool = False) -> "ColumnType": # type: ignore[reportUnknownReturnType]
        """
        Infer the ColumnType from a pandas Series, dtype, or dtype string.
        Uses `has_unit` to distinguish between united and raw types.
        """
        if isinstance(dtype, pd.Series):
            dtype = dtype.dtype

        dtype_str = str(dtype).lower()

        candidates = [
            col_type
            for col_type in cls
            if col_type.value.has_unit == has_unit
        ]

        for column_type in candidates:
            expected_dtype = column_type.value.dataframe_storage_type

            if isinstance(expected_dtype, pd.api.extensions.ExtensionDtype):
                if dtype_str == str(expected_dtype).lower():
                    return column_type
            elif isinstance(expected_dtype, type):
                try:
                    if isinstance(dtype, expected_dtype):
                        return column_type
                except TypeError:
                    pass  # e.g., str vs dtype mismatch
            elif dtype_str == str(expected_dtype).lower():
                return column_type

        raise ValueError(f"No matching ColumnType for dtype: {dtype} with has_unit={has_unit}")

    @overload
    @classmethod
    def infer_approbiate_column_type(cls, scalar_or_array_type: type[SCALAR_TYPE], precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": ...
    @overload
    @classmethod
    def infer_approbiate_column_type(cls, scalar_or_array_type: type[ARRAY_TYPE], precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": ...
    @overload
    @classmethod
    def infer_approbiate_column_type(cls, scalar_or_array_type: pd.Series) -> "ColumnType": ... # type: ignore[reportUnknownReturnType]
    @classmethod
    def infer_approbiate_column_type(cls, scalar_or_array_type: type[SCALAR_TYPE]|type[ARRAY_TYPE]|pd.Series, precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": # type: ignore[reportUnknownReturnType]
        options: list[ColumnType] = []
        if isinstance(scalar_or_array_type, pd.Series):
            dtype: Dtype = scalar_or_array_type.dtype
            return cls.from_dtype(dtype) # type: ignore[reportUnknownReturnType]
        else:
            for column_type in cls:
                if column_type.value.scalar_type == scalar_or_array_type:
                    options.append(column_type)
                if column_type.value.array_type == scalar_or_array_type:
                    options.append(column_type)
            if len(options) == 0:
                raise ValueError(f"No column type found for UnitedScalar type {scalar_or_array_type}.")
            elif len(options) == 1:
                return options[0]
            else:
                if precision is None:
                    raise ValueError(f"Multiple column types found for UnitedScalar type {scalar_or_array_type}.")
                for option in options:
                    if option.value.precision == precision:
                        return option
                raise ValueError(f"No column type found for UnitedScalar type {scalar_or_array_type} with precision {precision}. Tried: {[ct.name for ct in options]}")
        
    # ------------ Check the compatibility of a scalar or array with the column type ------------

    @overload
    def check_compatibility(self, scalar_or_array_value: SCALAR_TYPE) -> bool: ...
    @overload
    def check_compatibility(self, scalar_or_array_value: ARRAY_TYPE) -> bool: ...
    def check_compatibility(self, scalar_or_array_value: SCALAR_TYPE | ARRAY_TYPE) -> bool:
        if self.has_unit:
            if not isinstance(scalar_or_array_value, United):
                return False
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if isinstance(scalar_or_array_value, RealUnitedScalar) or isinstance(scalar_or_array_value, RealUnitedArray):
                    return True
                else:
                    return False
            case ColumnType.COMPLEX_NUMBER_128:
                if isinstance(scalar_or_array_value, ComplexUnitedScalar) or isinstance(scalar_or_array_value, ComplexUnitedArray):
                    return True
                else:
                    return False
            case ColumnType.STRING:
                if isinstance(scalar_or_array_value, str) or isinstance(scalar_or_array_value, StringArray):
                    return True
                else:
                    return False
            case ColumnType.BOOL:
                if isinstance(scalar_or_array_value, bool) or isinstance(scalar_or_array_value, BoolArray):
                    return True
                else:
                    return False
            case ColumnType.TIMESTAMP:
                if isinstance(scalar_or_array_value, Timestamp) or isinstance(scalar_or_array_value, TimestampArray):
                    return True
                else:
                    return False
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                if isinstance(scalar_or_array_value, int) or isinstance(scalar_or_array_value, IntArray):
                    return True
                else:
                    return False
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                if isinstance(scalar_or_array_value, float) or isinstance(scalar_or_array_value, FloatArray):
                    return True
                else:
                    return False
            case ColumnType.COMPLEX_128:
                if isinstance(scalar_or_array_value, complex) or isinstance(scalar_or_array_value, ComplexUnitedArray):
                    return True
                else:
                    return False

    # ------------ Maybe delete? ------------

    # def create_scalar_from_value(self, value: float|complex|str|bool|int|Timestamp|None, unit: Unit|None = None) -> SCALAR_TYPE:

    #     if value is None:
    #         match self:
    #             case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
    #                 if unit is None:
    #                     raise ValueError(f"Unit is required for real number types.")
    #                 return RealUnitedScalar(np.nan, unit)
    #             case ColumnType.COMPLEX_NUMBER_128:
    #                 raise ValueError(f"Complex United Scalar is not yet supported.")
    #             case ColumnType.STRING:
    #                 raise ValueError(f"Missing values are not supported for string types.")
    #             case ColumnType.BOOL:
    #                 raise ValueError(f"Missing values are not supported for boolean types.")
    #             case ColumnType.TIMESTAMP:
    #                 raise ValueError(f"Missing values are not supported for timestamp types.")
    #             case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
    #                 raise ValueError(f"Missing values are not supported for integer types.")
    #             case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
    #                 return math.nan
    #             case ColumnType.COMPLEX_128:
    #                 return math.nan + 1j

    #     if isinstance(value, float) and self is not ColumnType.FLOAT_64 and self is not ColumnType.FLOAT_32 and self is not ColumnType.REAL_NUMBER_64 and self is not ColumnType.REAL_NUMBER_32:
    #         raise ValueError(f"Invalid value: {value} for column type {self.name}.")

    #     match self:
    #         case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
    #             if unit is None:
    #                 raise ValueError(f"Unit is required for real number types.")
    #             if not isinstance(value, (float, int)):
    #                 raise ValueError(f"Invalid value: {value} for column type {self.name}.")
    #             return RealUnitedScalar(value, unit)
    #         case ColumnType.COMPLEX_NUMBER_128:
    #             raise ValueError(f"Complex United Scalar is not yet supported.")
    #         case ColumnType.STRING:
    #             return str(value)
    #         case ColumnType.BOOL:
    #             return bool(value)
    #         case ColumnType.TIMESTAMP:
    #             if not isinstance(value, Timestamp):
    #                 raise ValueError(f"Invalid value: {value} for column type {self.name}.")
    #             return Timestamp(value)
    #         case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
    #             if not isinstance(value, (int, float)):
    #                 raise ValueError(f"Invalid value: {value} for column type {self.name}.")
    #             return int(value)
    #         case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
    #             if not isinstance(value, (float, int)):
    #                 raise ValueError(f"Invalid value: {value} for column type {self.name}.")
    #             return float(value)
    #         case ColumnType.COMPLEX_128:
    #             if not isinstance(value, (complex, float, int)):
    #                 raise ValueError(f"Invalid value: {value} for column type {self.name}.")
    #             return complex(value)
            
    # def create_array_from_values(self, values: np.ndarray[Any, Any], unit: Unit|None = None) -> ARRAY_TYPE:
    #     match self:
    #         case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
    #             if unit is None:
    #                 raise ValueError(f"Unit is required for RealUnitedArray.")
    #             return RealUnitedArray(values, unit)
    #         case ColumnType.COMPLEX_NUMBER_128:
    #             if unit is None:
    #                 raise ValueError(f"Unit is required for ComplexUnitedArray.")
    #             raise NotImplementedError(f"ComplexUnitedArray is not yet supported.")
    #         case ColumnType.STRING:
    #             if not unit is None:
    #                 raise ValueError(f"Unit is not allowed for StringArray.")
    #             return StringArray(values)
    #         case ColumnType.BOOL:
    #             if not unit is None:
    #                 raise ValueError(f"Unit is not allowed for BoolArray.")
    #             return BoolArray(values)
    #         case ColumnType.TIMESTAMP:
    #             if not unit is None:
    #                 raise ValueError(f"Unit is not allowed for TimestampArray.")
    #             return TimestampArray(values)
    #         case ColumnType.COMPLEX_128:
    #             if not unit is None:
    #                 raise ValueError(f"Unit is not allowed for ComplexUnitedArray.")
    #             raise NotImplementedError(f"ComplexUnitedArray is not yet supported.")
    #         case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
    #             if not unit is None:
    #                 raise ValueError(f"Unit is not allowed for FloatArray.")
    #             return FloatArray(values)
    #         case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
    #             if not unit is None:
    #                 raise ValueError(f"Unit is not allowed for IntArray.")
    #             return IntArray(values)

    def check_scalar_type(self, type: type) -> bool:
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                return issubclass(type, RealUnitedScalar)
            case ColumnType.COMPLEX_NUMBER_128:
                return issubclass(type, ComplexUnitedScalar)
            case ColumnType.STRING:
                return issubclass(type, str)
            case ColumnType.BOOL:
                return issubclass(type, bool)
            case ColumnType.TIMESTAMP:
                return issubclass(type, Timestamp)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return issubclass(type, int)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return issubclass(type, float)
            case ColumnType.COMPLEX_128:
                return issubclass(type, complex)
            
    def check_array_type(self, type: type) -> bool:
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                return issubclass(type, RealUnitedArray)
            case ColumnType.COMPLEX_NUMBER_128:
                return issubclass(type, ComplexUnitedArray)
            case ColumnType.STRING:
                return issubclass(type, StringArray)
            case ColumnType.BOOL:
                return issubclass(type, BoolArray)
            case ColumnType.TIMESTAMP:
                return issubclass(type, TimestampArray)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return issubclass(type, IntArray)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return issubclass(type, FloatArray)
            case ColumnType.COMPLEX_128:
                return issubclass(type, ComplexArray)