from typing import TypeAlias, Literal, overload, Any, NamedTuple
from enum import Enum
import math

import pandas as pd
from pandas._typing import Dtype
from pandas import Timestamp
import numpy as np
from typing import Optional

from .._units_and_dimension.unit import Unit
from .._scalars.real_united_scalar import RealUnitedScalar
from .._scalars.complex_united_scalar import ComplexUnitedScalar
from .._arrays.real_united_array import RealUnitedArray
from .._arrays.complex_united_array import ComplexUnitedArray
from .._arrays.string_array import StringArray
from .._arrays.complex_array import ComplexArray
from .._arrays.int_array import IntArray
from .._arrays.float_array import FloatArray
from .._arrays.bool_array import BoolArray
from .._arrays.timestamp_array import TimestampArray
from .._units_and_dimension.has_unit_protocol import HasUnit
from .._arrays.base_array import BaseArray
from .._utils.general import VALUE_TYPE, SCALAR_TYPE, ARRAY_TYPE

PANDAS_SERIES_TYPE: TypeAlias = float|int|complex|bool|str|Timestamp
NUMERIC_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar|int|float|complex|Timestamp
NUMERIC_ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray|IntArray|FloatArray|TimestampArray
NUMERIC_TYPE: TypeAlias = NUMERIC_SCALAR_TYPE|NUMERIC_ARRAY_TYPE
UNITED_ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray
UNITED_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar
NUMPY_STORAGE_TYPE: TypeAlias = np.float64|np.float32|np.float16|np.complex128|np.complex64|np.int64|np.int32|np.int16|np.int8|np.bool_|np.datetime64|np.str_

class ColumnTypeInformation(NamedTuple):
    """
    Simplified ColumnType metadata containing only the fields that are actually used.
    
    """
    has_unit: bool
    value_type: type[VALUE_TYPE]
    scalar_type: type[SCALAR_TYPE]
    array_type: type[ARRAY_TYPE]
    dataframe_storage_type: Dtype
    numpy_storage_options: list[type[NUMPY_STORAGE_TYPE]]
    missing_values_in_dataframe: Any
    precision: int|None

class ColumnType(Enum):
    value: ColumnTypeInformation # type: ignore[assignment]

    REAL_NUMBER_64 = ColumnTypeInformation(         has_unit=True,  value_type=float,       scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_type=pd.Float64Dtype(),  numpy_storage_options=[np.float64, np.float32, np.float16],     missing_values_in_dataframe=pd.NA,       precision=64)
    REAL_NUMBER_32 = ColumnTypeInformation(         has_unit=True,  value_type=float,       scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_type=pd.Float32Dtype(),  numpy_storage_options=[np.float32, np.float64, np.float16],     missing_values_in_dataframe=pd.NA,       precision=32)
    COMPLEX_NUMBER_128 = ColumnTypeInformation(     has_unit=True,  value_type=complex,     scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_type=np.complex128,      numpy_storage_options=[np.complex128, np.complex64],            missing_values_in_dataframe=math.nan+1j, precision=128)
    STRING = ColumnTypeInformation(                 has_unit=False, value_type=str,         scalar_type=str,                    array_type=StringArray,         dataframe_storage_type=pd.StringDtype(),   numpy_storage_options=[np.str_],                                missing_values_in_dataframe=pd.NA,       precision=None)
    INTEGER_64 = ColumnTypeInformation(             has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int64Dtype(),    numpy_storage_options=[np.int64, np.int32, np.int16, np.int8],  missing_values_in_dataframe=pd.NA,       precision=64)
    INTEGER_32 = ColumnTypeInformation(             has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int32Dtype(),    numpy_storage_options=[np.int32, np.int64, np.int16, np.int8],  missing_values_in_dataframe=pd.NA,       precision=32)
    INTEGER_16 = ColumnTypeInformation(             has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int16Dtype(),    numpy_storage_options=[np.int16, np.int64, np.int32, np.int8],  missing_values_in_dataframe=pd.NA,       precision=16)
    INTEGER_8 = ColumnTypeInformation(              has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int8Dtype(),     numpy_storage_options=[np.int8, np.int64, np.int32, np.int16],  missing_values_in_dataframe=pd.NA,       precision=8)
    FLOAT_64 = ColumnTypeInformation(               has_unit=False, value_type=float,       scalar_type=float,                  array_type=FloatArray,          dataframe_storage_type=pd.Float64Dtype(),  numpy_storage_options=[np.float64, np.float32, np.float16],     missing_values_in_dataframe=pd.NA,       precision=64)
    FLOAT_32 = ColumnTypeInformation(               has_unit=False, value_type=float,       scalar_type=float,                  array_type=FloatArray,          dataframe_storage_type=pd.Float64Dtype(),  numpy_storage_options=[np.float32, np.float64, np.float16],     missing_values_in_dataframe=pd.NA,       precision=32)
    COMPLEX_128 = ColumnTypeInformation(            has_unit=False, value_type=complex,     scalar_type=complex,                array_type=ComplexArray,        dataframe_storage_type=np.complex128,      numpy_storage_options=[np.complex128, np.complex64],            missing_values_in_dataframe=math.nan+1j, precision=128)
    BOOL = ColumnTypeInformation(                   has_unit=False, value_type=bool,        scalar_type=bool,                   array_type=BoolArray,           dataframe_storage_type=pd.BooleanDtype(),  numpy_storage_options=[np.bool_],                               missing_values_in_dataframe=pd.NA,       precision=None)
    TIMESTAMP = ColumnTypeInformation(              has_unit=False, value_type=Timestamp,   scalar_type=Timestamp,              array_type=TimestampArray,      dataframe_storage_type='datetime64[ns]',   numpy_storage_options=[np.datetime64],                          missing_values_in_dataframe=pd.NaT,      precision=None)

    @property
    def has_unit(self) -> bool:
        return self.value.has_unit
    
    @property
    def is_numeric(self) -> bool:
        return self in {ColumnType.REAL_NUMBER_64, ColumnType.REAL_NUMBER_32, ColumnType.FLOAT_64, ColumnType.FLOAT_32, ColumnType.COMPLEX_128, ColumnType.COMPLEX_NUMBER_128, ColumnType.INTEGER_64, ColumnType.INTEGER_32, ColumnType.INTEGER_16, ColumnType.INTEGER_8, ColumnType.TIMESTAMP}
    
    @property
    def is_not_numeric(self) -> bool:
        return not self.is_numeric
    
    @property
    def array_type(self) -> type[ARRAY_TYPE]:
        return self.value.array_type
    
    @property
    def scalar_type(self) -> type[SCALAR_TYPE]:
        return self.value.scalar_type
    
    @property
    def value_type(self) -> type[VALUE_TYPE]:
        return self.value.value_type
    
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
            
    def get_scalar_value_from_dataframe(self, value: VALUE_TYPE, unit: Unit|None = None) -> SCALAR_TYPE:
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
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Value {value} is not a float or int.")
                if unit is None:
                    raise ValueError(f"Unit is required for {self.name}.")
                return RealUnitedScalar.create_from_value_and_unit(value, unit)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                if not isinstance(value, (float, int)):
                    raise ValueError(f"Value {value} is not a float or int.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for float or int.")
                return float(value)
            case ColumnType.COMPLEX_NUMBER_128:
                if not isinstance(value, (complex, np.complex128)):
                    raise ValueError(f"Value {value} is not a complex.")
                if unit is None:
                    raise ValueError(f"Unit is required for {self.name}.")
                raise NotImplementedError(f"ComplexUnitedScalar is not implemented.")
            case ColumnType.COMPLEX_128:
                if not isinstance(value, (complex, np.complex128)):
                    raise ValueError(f"Value {value} is not a complex.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for complex.")
                return complex(value)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                if not isinstance(value, (int, np.integer)):
                    raise ValueError(f"Value {value} is not an int.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for int.")
                return int(value)
            case ColumnType.STRING:
                if not isinstance(value, (str, np.str_)):
                    raise ValueError(f"Value {value} is not a str.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for str.")
                return str(value)
            case ColumnType.BOOL:
                if not isinstance(value, (bool, np.bool_)):
                    raise ValueError(f"Value {value} is not a bool.")
                if not unit is None:
                    raise ValueError(f"Unit in dataframe is not allowed for bool.")
                return bool(value)
            case ColumnType.TIMESTAMP:
                if not isinstance(value, (Timestamp, np.datetime64)):
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
        Sophisticated ColumnType inference from dtype and has_unit flag.
        
        This method provides comprehensive mapping between pandas/numpy dtypes and ColumnTypes,
        using the has_unit flag to distinguish between united (with units) and raw types.
        
        Mapping Rules:
        - Float types: float64/float32 -> REAL_NUMBER_XX (has_unit=True) or FLOAT_XX (has_unit=False)  
        - Complex types: complex128 -> COMPLEX_NUMBER_128 (has_unit=True) or COMPLEX_128 (has_unit=False)
        - Integer types: int64/32/16/8 -> INTEGER_XX (has_unit=False only)
        - Other types: datetime64->TIMESTAMP, string->STRING, bool->BOOL (has_unit=False only)
        
        Args:
            dtype: Input dtype from pandas Series, extension dtype, numpy dtype, or string
            has_unit: Whether the column should support physical units
            
        Returns:
            Appropriate ColumnType enum value
            
        Raises:
            ValueError: If no matching ColumnType found for the dtype/has_unit combination
        """
        
        # Step 1: Normalize input to extract actual dtype
        if isinstance(dtype, pd.Series):
            dtype = dtype.dtype
        
        # Step 2: Normalize dtype to a canonical string representation
        normalized_dtype = cls._normalize_dtype_string(dtype)
        
        # Step 3: Define comprehensive dtype mappings
        # Structure: {normalized_dtype: {has_unit: ColumnType}}
        dtype_mappings = {
            # Float types - support both united and raw variants
            'float64': {
                True: cls.REAL_NUMBER_64,    # United scalars with units
                False: cls.FLOAT_64          # Raw float values
            },
            'float32': {
                True: cls.REAL_NUMBER_32,    # United scalars with units
                False: cls.FLOAT_32          # Raw float values
            },
            
            # Complex types - support both united and raw variants
            'complex128': {
                True: cls.COMPLEX_NUMBER_128, # United complex scalars with units
                False: cls.COMPLEX_128        # Raw complex values (Note: currently both have has_unit=True, this may need fixing)
            },
            'complex64': {
                True: None,                   # Not currently supported
                False: cls.COMPLEX_128        # Raw complex values (64-bit maps to 128-bit storage)
            },
            
            # Integer types - only raw variants supported (no physical units for counts)
            'int64': {
                True: None,                   # Integer units not supported
                False: cls.INTEGER_64
            },
            'int32': {
                True: None,                   # Integer units not supported
                False: cls.INTEGER_32
            },
            'int16': {
                True: None,                   # Integer units not supported
                False: cls.INTEGER_16
            },
            'int8': {
                True: None,                   # Integer units not supported
                False: cls.INTEGER_8
            },
            
            # Non-numeric types - only raw variants (no physical units)
            'datetime64': {
                True: None,                   # Timestamp units not supported
                False: cls.TIMESTAMP
            },
            'string': {
                True: None,                   # String units not supported
                False: cls.STRING
            },
            'bool': {
                True: None,                   # Boolean units not supported
                False: cls.BOOL
            }
        }
        
        # Step 4: Look up the appropriate ColumnType
        if normalized_dtype in dtype_mappings:
            column_type = dtype_mappings[normalized_dtype].get(has_unit)
            if column_type is not None:
                return column_type
            else:
                # Specific error for unsupported has_unit combination
                supported_has_unit = [k for k, v in dtype_mappings[normalized_dtype].items() if v is not None]
                raise ValueError(
                    f"ColumnType for dtype '{normalized_dtype}' does not support has_unit={has_unit}. "
                    f"Supported has_unit values: {supported_has_unit}"
                )
        
        # Step 5: No mapping found - provide helpful error
        available_dtypes = list(dtype_mappings.keys())
        raise ValueError(
            f"No matching ColumnType for dtype: {dtype} (normalized: '{normalized_dtype}') with has_unit={has_unit}. "
            f"Supported dtypes: {available_dtypes}"
        )
    
    @classmethod
    def _normalize_dtype_string(cls, dtype: Any) -> str:
        """
        Normalize various dtype representations to canonical string form.
        
        Handles pandas extension dtypes, numpy dtypes, numpy type objects, and string representations.
        
        Args:
            dtype: Input dtype in any supported format
            
        Returns:
            Canonical string representation (e.g., 'float64', 'complex128', 'string')
        """
        
        dtype_str = str(dtype).lower()
        
        # Handle pandas extension dtypes (case insensitive)
        if 'float64dtype' in dtype_str:
            return 'float64'
        elif 'float32dtype' in dtype_str:
            return 'float32'
        elif 'int64dtype' in dtype_str:
            return 'int64'
        elif 'int32dtype' in dtype_str:
            return 'int32'
        elif 'int16dtype' in dtype_str:
            return 'int16'
        elif 'int8dtype' in dtype_str:
            return 'int8'
        elif 'stringdtype' in dtype_str:
            return 'string'
        elif 'booleandtype' in dtype_str:
            return 'bool'
        elif 'complex64dtype' in dtype_str:
            return 'complex64'
        elif 'complex128dtype' in dtype_str:
            return 'complex128'
        
        # Handle datetime types (various representations)
        elif 'datetime64' in dtype_str:
            return 'datetime64'
        elif 'timestamp' in dtype_str:
            return 'datetime64'
        
        # Handle direct numpy dtype strings
        elif dtype_str in ['complex128', 'complex64', 'float64', 'float32', 
                           'int64', 'int32', 'int16', 'int8']:
            return dtype_str
            
        # Handle boolean variations
        elif dtype_str in ['bool', 'bool_', 'boolean']:
            return 'bool'
            
        # Handle string variations  
        elif dtype_str in ['str', 'str_', 'string', 'object']:
            return 'string'
        
        # Handle numpy type objects by their __name__ attribute
        elif hasattr(dtype, '__name__'):
            name = dtype.__name__.lower()
            # Map numpy type names to canonical forms
            name_mapping = {
                'complex128': 'complex128',
                'complex64': 'complex64', 
                'float64': 'float64',
                'float32': 'float32',
                'int64': 'int64', 
                'int32': 'int32',
                'int16': 'int16',
                'int8': 'int8',
                'bool_': 'bool',
                'str_': 'string',
                'datetime64': 'datetime64'
            }
            if name in name_mapping:
                return name_mapping[name]
        
        # Handle numpy type instances
        elif hasattr(dtype, 'name'):
            name = dtype.name.lower()
            if name in ['complex128', 'complex64', 'float64', 'float32',
                       'int64', 'int32', 'int16', 'int8', 'bool', 'str', 'datetime64']:
                return name if name != 'str' else 'string'
        
        # Fallback: return the string representation for debugging
        return dtype_str

    @overload
    @classmethod
    def infer_approbiate_column_type(cls, scalar_or_array_type: type[ARRAY_TYPE], precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": ...
    @overload
    @classmethod
    def infer_approbiate_column_type(cls, scalar_or_array_type: type[SCALAR_TYPE], precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": ...
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
                if issubclass(scalar_or_array_type, column_type.value.array_type):
                    options.append(column_type)
                if issubclass(scalar_or_array_type, column_type.value.scalar_type):
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
    def check_compatibility(self, scalar_or_array_value: ARRAY_TYPE, unit: Optional[Unit]=None) -> bool: ...
    @overload
    def check_compatibility(self, scalar_or_array_value: SCALAR_TYPE, unit: Optional[Unit]=None) -> bool: ...
    @overload
    def check_compatibility(self, scalar_or_array_value: BaseArray[Any, Any, Any]) -> bool: ...
    def check_compatibility(self, scalar_or_array_value: SCALAR_TYPE | ARRAY_TYPE|BaseArray[Any, Any, Any], unit: Optional[Unit]=None) -> bool:
        if self.has_unit:
            if not isinstance(scalar_or_array_value, HasUnit):
                return False
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if isinstance(scalar_or_array_value, RealUnitedScalar) or isinstance(scalar_or_array_value, RealUnitedArray):
                    if unit is not None:
                        if not scalar_or_array_value.unit.compatible_to(unit):
                            return False
                    return True
                else:
                    return False
            case ColumnType.COMPLEX_NUMBER_128:
                if isinstance(scalar_or_array_value, ComplexUnitedScalar) or isinstance(scalar_or_array_value, ComplexUnitedArray):
                    if unit is not None:
                        if not scalar_or_array_value.unit.compatible_to(unit):
                            return False
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

    def __reduce_ex__(self, _: Any):
        """Custom pickle reduction to preserve enum identity by name."""
        return (getattr, (self.__class__, self.name))