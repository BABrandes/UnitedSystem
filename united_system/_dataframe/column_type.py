"""
Column Type Management Module for UnitedSystem DataFrames.

This module provides comprehensive column type management for the UnitedSystem DataFrame implementation.
It defines column types, their metadata, and conversion functions for handling different data types
with proper unit support and missing value handling.

Key Components:
- ColumnTypeInformation: Metadata container for column type properties
- ColumnType: Enumeration of all supported column types with their configurations
- Conversion Functions: Methods for converting between different data representations
- Validation Functions: Methods for checking type compatibility and constraints

Supported Column Types:
- Numeric types with units: REAL_NUMBER_64, REAL_NUMBER_32, COMPLEX_NUMBER_128
- Numeric types without units: FLOAT_64, FLOAT_32, COMPLEX_128, INTEGER_64, INTEGER_32, INTEGER_16, INTEGER_8
- Non-numeric types: STRING, BOOL, TIMESTAMP

Each column type includes:
- Storage type information for pandas DataFrames and numpy arrays
- Missing value representations for different contexts
- Unit support configuration
- Type conversion methods

Example:
    >>> column_type = ColumnType.FLOAT_64
    >>> value = column_type.get_value_for_dataframe(3.14)
    >>> series = column_type.get_pd_series_for_dataframe([1.0, 2.0, 3.0], None, None)
    >>> retrieved_value = column_type.get_value_from_pd_series(series, 0)
"""

from typing import Literal, overload, Any, NamedTuple, Optional, Union
from enum import Enum
import math

import pandas as pd
from pandas._typing import Dtype
from pandas import Timestamp
import numpy as np
# Use Any instead of DtypeLike for compatibility with older numpy versions
from typing import Any as DtypeLike

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
from .._utils.value_type import VALUE_TYPE, VALUE_TYPE_RUNTIME
from .._utils.scalar_type import SCALAR_TYPE
from .._utils.array_type import ARRAY_TYPE

class ColumnTypeInformation(NamedTuple):
    """
    Metadata container for column type properties and configuration.
    
    This NamedTuple contains all the essential information needed to handle
    a specific column type, including storage types, missing value representations,
    and type information for different contexts.
    
    Attributes:
        has_unit (bool): Whether this column type supports units (e.g., meters, seconds).
        value_type (type[VALUE_TYPE]): The Python type for individual values (e.g., float, str).
        scalar_type (type[SCALAR_TYPE]): The type for scalar values, including united scalars.
        array_type (type[ARRAY_TYPE]): The type for array values, including united arrays.
        dataframe_storage_type (Dtype): The pandas dtype used for DataFrame storage.
        numpy_storage_options (DtypeLike): The numpy dtype used for array operations.
        missing_values_in_dataframe (Any): Representation of missing values in pandas DataFrames.
        missing_values_retrieved (Optional[VALUE_TYPE]): Representation of missing values when retrieved.
        missing_values_in_numpy_array (Any): Representation of missing values in numpy arrays.
        precision (int|None): Numeric precision in bits, or None for non-numeric types.
    
    Example:
        >>> info = ColumnTypeInformation(
        ...     has_unit=True,
        ...     value_type=float,
        ...     scalar_type=RealUnitedScalar,
        ...     array_type=RealUnitedArray,
        ...     dataframe_storage_type=pd.Float64Dtype(),
        ...     numpy_storage_options=np.float64,
        ...     missing_values_in_dataframe=pd.NA,
        ...     missing_values_retrieved=math.nan,
        ...     missing_values_in_numpy_array=np.nan,
        ...     precision=64
        ... )
    """
    has_unit: bool
    value_type: type[VALUE_TYPE]
    scalar_type: type[SCALAR_TYPE]
    array_type: type[ARRAY_TYPE]
    dataframe_storage_type: Dtype
    numpy_storage_options: DtypeLike
    missing_values_in_dataframe: Any
    missing_values_retrieved: Optional[VALUE_TYPE]
    missing_values_in_numpy_array: Any
    precision: int|None

class ColumnType(Enum):
    """
    Enumeration of all supported column types with their complete configuration.
    
    Each enum value contains a ColumnTypeInformation instance that defines
    all the properties and behavior for that specific column type, including
    storage types, missing value handling, and unit support.
    
    Supported Types:
        Numeric with Units:
        - REAL_NUMBER_64: 64-bit floating point with unit support
        - REAL_NUMBER_32: 32-bit floating point with unit support  
        - COMPLEX_NUMBER_128: 128-bit complex with unit support
        
        Numeric without Units:
        - FLOAT_64: 64-bit floating point without units
        - FLOAT_32: 32-bit floating point without units
        - COMPLEX_128: 128-bit complex without units
        - INTEGER_64: 64-bit integer
        - INTEGER_32: 32-bit integer
        - INTEGER_16: 16-bit integer
        - INTEGER_8: 8-bit integer
        
        Non-numeric:
        - STRING: String values
        - BOOL: Boolean values
        - TIMESTAMP: Timestamp values
    
    Example:
        >>> column_type = ColumnType.FLOAT_64
        >>> print(column_type.has_unit)  # False
        >>> print(column_type.is_numeric)  # True
        >>> value = column_type.get_value_for_dataframe(3.14)
    """
    value: ColumnTypeInformation # type: ignore[assignment]

    # Numeric types with unit support
    REAL_NUMBER_64 = ColumnTypeInformation(         has_unit=True,  value_type=float,       scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_type=pd.Float64Dtype(),  numpy_storage_options=np.float64,        missing_values_in_dataframe=pd.NA,       missing_values_retrieved=math.nan,              missing_values_in_numpy_array=np.nan,                  precision=64)
    """64-bit floating point with unit support (e.g., meters, seconds)."""
    
    REAL_NUMBER_32 = ColumnTypeInformation(         has_unit=True,  value_type=float,       scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_type=pd.Float32Dtype(),  numpy_storage_options=np.float32,        missing_values_in_dataframe=pd.NA,       missing_values_retrieved=math.nan,              missing_values_in_numpy_array=np.nan,                  precision=32)
    """32-bit floating point with unit support (e.g., meters, seconds)."""
    
    COMPLEX_NUMBER_128 = ColumnTypeInformation(     has_unit=True,  value_type=complex,     scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_type=np.complex128,      numpy_storage_options=np.complex128,     missing_values_in_dataframe=pd.NA,       missing_values_retrieved=math.nan+1j,           missing_values_in_numpy_array=np.nan+1j,               precision=128)
    """128-bit complex number with unit support."""
    
    # Non-numeric types
    STRING = ColumnTypeInformation(                 has_unit=False, value_type=str,         scalar_type=str,                    array_type=StringArray,         dataframe_storage_type=pd.StringDtype(),   numpy_storage_options=np.str_,           missing_values_in_dataframe=pd.NA,       missing_values_retrieved=None,                  missing_values_in_numpy_array=None,                    precision=None)
    """String values without unit support."""
    
    # Integer types without unit support
    INTEGER_64 = ColumnTypeInformation(             has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int64Dtype(),    numpy_storage_options=np.int64,          missing_values_in_dataframe=pd.NA,       missing_values_retrieved=None,                  missing_values_in_numpy_array=None,                    precision=64)
    """64-bit integer without unit support."""
    
    INTEGER_32 = ColumnTypeInformation(             has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int32Dtype(),    numpy_storage_options=np.int32,          missing_values_in_dataframe=pd.NA,       missing_values_retrieved=None,                  missing_values_in_numpy_array=None,                    precision=32)
    """32-bit integer without unit support."""
    
    INTEGER_16 = ColumnTypeInformation(             has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int16Dtype(),    numpy_storage_options=np.int16,          missing_values_in_dataframe=pd.NA,       missing_values_retrieved=None,                  missing_values_in_numpy_array=None,                    precision=16)
    """16-bit integer without unit support."""
    
    INTEGER_8 = ColumnTypeInformation(              has_unit=False, value_type=int,         scalar_type=int,                    array_type=IntArray,            dataframe_storage_type=pd.Int8Dtype(),     numpy_storage_options=np.int8,           missing_values_in_dataframe=pd.NA,       missing_values_retrieved=None,                  missing_values_in_numpy_array=None,                    precision=8)
    """8-bit integer without unit support."""
    
    # Floating point types without unit support
    FLOAT_64 = ColumnTypeInformation(               has_unit=False, value_type=float,       scalar_type=float,                  array_type=FloatArray,          dataframe_storage_type=pd.Float64Dtype(),  numpy_storage_options=np.float64,        missing_values_in_dataframe=pd.NA,       missing_values_retrieved=math.nan,              missing_values_in_numpy_array=np.nan,                  precision=64)
    """64-bit floating point without unit support."""
    
    FLOAT_32 = ColumnTypeInformation(               has_unit=False, value_type=float,       scalar_type=float,                  array_type=FloatArray,          dataframe_storage_type=pd.Float64Dtype(),  numpy_storage_options=np.float32,        missing_values_in_dataframe=pd.NA,       missing_values_retrieved=math.nan,              missing_values_in_numpy_array=np.nan,                  precision=32)
    """32-bit floating point without unit support."""
    
    COMPLEX_128 = ColumnTypeInformation(            has_unit=False, value_type=complex,     scalar_type=complex,                array_type=ComplexArray,        dataframe_storage_type=np.complex128,      numpy_storage_options=np.complex128,     missing_values_in_dataframe=pd.NA,       missing_values_retrieved=math.nan+1j,           missing_values_in_numpy_array=np.nan+1j,               precision=128)
    """128-bit complex number without unit support."""
    
    BOOL = ColumnTypeInformation(                   has_unit=False, value_type=bool,        scalar_type=bool,                   array_type=BoolArray,           dataframe_storage_type=pd.BooleanDtype(),  numpy_storage_options=np.bool_,          missing_values_in_dataframe=pd.NA,       missing_values_retrieved=None,                  missing_values_in_numpy_array=None,                    precision=None)
    """Boolean values without unit support."""
    
    TIMESTAMP = ColumnTypeInformation(              has_unit=False, value_type=Timestamp,   scalar_type=Timestamp,              array_type=TimestampArray,      dataframe_storage_type='datetime64[ns]',   numpy_storage_options=np.datetime64,     missing_values_in_dataframe=pd.NaT,      missing_values_retrieved=Timestamp("NaT"),      missing_values_in_numpy_array=np.datetime64("NaT"),    precision=None)
    """Timestamp values without unit support."""

    @property
    def has_unit(self) -> bool:
        """
        Check if this column type supports units.
        
        Returns:
            bool: True if the column type supports units (e.g., meters, seconds),
                  False otherwise.
        
        Example:
            >>> ColumnType.REAL_NUMBER_64.has_unit  # True
            >>> ColumnType.FLOAT_64.has_unit  # False
        """
        return self.value.has_unit
    
    @property
    def is_numeric(self) -> bool:
        """
        Check if this column type represents numeric data.
        
        Returns:
            bool: True if the column type is numeric (including integers, floats,
                  complex numbers, and timestamps), False otherwise.
        
        Example:
            >>> ColumnType.FLOAT_64.is_numeric  # True
            >>> ColumnType.STRING.is_numeric  # False
        """
        return self in {ColumnType.REAL_NUMBER_64, ColumnType.REAL_NUMBER_32, ColumnType.FLOAT_64, ColumnType.FLOAT_32, ColumnType.COMPLEX_128, ColumnType.COMPLEX_NUMBER_128, ColumnType.INTEGER_64, ColumnType.INTEGER_32, ColumnType.INTEGER_16, ColumnType.INTEGER_8, ColumnType.TIMESTAMP}
    
    @property
    def is_not_numeric(self) -> bool:
        """
        Check if this column type represents non-numeric data.
        
        Returns:
            bool: True if the column type is non-numeric (strings, booleans),
                  False otherwise.
        
        Example:
            >>> ColumnType.STRING.is_not_numeric  # True
            >>> ColumnType.FLOAT_64.is_not_numeric  # False
        """
        return not self.is_numeric
    
    @property
    def array_type(self) -> type[ARRAY_TYPE]:
        """
        Get the array type associated with this column type.
        
        Returns:
            type[ARRAY_TYPE]: The array type class for this column type.
        
        Example:
            >>> ColumnType.FLOAT_64.array_type  # <class 'FloatArray'>
        """
        return self.value.array_type
    
    @property
    def scalar_type(self) -> type[SCALAR_TYPE]:
        """
        Get the scalar type associated with this column type.
        
        Returns:
            type[SCALAR_TYPE]: The scalar type class for this column type.
        
        Example:
            >>> ColumnType.FLOAT_64.scalar_type  # <class 'float'>
        """
        return self.value.scalar_type
    
    @property
    def value_type(self) -> type[VALUE_TYPE]:
        """
        Get the value type associated with this column type.
        
        Returns:
            type[VALUE_TYPE]: The value type class for this column type.
        
        Example:
            >>> ColumnType.FLOAT_64.value_type  # <class 'float'>
        """
        return self.value.value_type
    
    @property
    def can_be_none(self) -> bool:
        """
        Check if this column type can have missing values.
        
        Returns:
            bool: True if the column type supports missing values, False otherwise.
        
        Example:
            >>> ColumnType.FLOAT_64.can_be_none  # True
            >>> ColumnType.STRING.can_be_none  # True
        """
        return self.value.missing_values_in_dataframe is not None
    
    @property
    def missing_value_in_dataframe(self) -> Any:
        """
        Get the missing value representation used in pandas DataFrames.
        
        Returns:
            Any: The missing value representation (e.g., pd.NA, pd.NaT).
        
        Example:
            >>> ColumnType.FLOAT_64.missing_value_in_dataframe  # pd.NA
            >>> ColumnType.TIMESTAMP.missing_value_in_dataframe  # pd.NaT
        """
        return self.value.missing_values_in_dataframe
    
    @property
    def missing_value_retrieved(self) -> Optional[VALUE_TYPE]:
        """
        Get the missing value representation when retrieved from DataFrames.
        
        Returns:
            Optional[VALUE_TYPE]: The missing value representation when retrieved,
                                 or None if not supported.
        
        Example:
            >>> ColumnType.FLOAT_64.missing_value_retrieved  # math.nan
            >>> ColumnType.STRING.missing_value_retrieved  # None
        """
        return self.value.missing_values_retrieved
    
    @property
    def numpy_storage_type(self) -> DtypeLike:
        """
        Get the numpy storage type for this column type.
        
        Returns:
            DtypeLike: The numpy dtype used for array operations.
        
        Example:
            >>> ColumnType.FLOAT_64.numpy_storage_type  # numpy.float64
            >>> ColumnType.INTEGER_32.numpy_storage_type  # numpy.int32
        """
        return self.value.numpy_storage_options
    
    @property
    def missing_values_in_numpy_array(self) -> Any:
        """
        Get the missing value representation used in numpy arrays.
        
        Returns:
            Any: The missing value representation for numpy arrays
                 (e.g., np.nan, np.datetime64("NaT"), None).
        
        Example:
            >>> ColumnType.FLOAT_64.missing_values_in_numpy_array  # np.nan
            >>> ColumnType.TIMESTAMP.missing_values_in_numpy_array  # np.datetime64("NaT")
        """
        return self.value.missing_values_in_numpy_array
    
    def validate_missing_values_allowed(self) -> bool:
        """
        Check if this column type allows missing values.
        
        Returns:
            bool: True if the column type supports missing values, False otherwise.
        
        Example:
            >>> ColumnType.FLOAT_64.validate_missing_values_allowed()  # True
            >>> ColumnType.STRING.validate_missing_values_allowed()  # True
        """
        return self.value.missing_values_in_dataframe is not None

    def get_appropriate_missing_value(self, context: Literal["dataframe", "retrieved", "numpy"]) -> Any:
        """
        Get the appropriate missing value for the given context.
        
        Args:
            context (Literal["dataframe", "retrieved", "numpy"]): The context for which
                to get the missing value representation.
                - "dataframe": Missing value in pandas DataFrames
                - "retrieved": Missing value when retrieved from DataFrames
                - "numpy": Missing value in numpy arrays
        
        Returns:
            Any: The appropriate missing value representation for the given context.
        
        Example:
            >>> ColumnType.FLOAT_64.get_appropriate_missing_value("dataframe")  # pd.NA
            >>> ColumnType.FLOAT_64.get_appropriate_missing_value("retrieved")  # math.nan
            >>> ColumnType.FLOAT_64.get_appropriate_missing_value("numpy")  # np.nan
        """
        match context:
            case "dataframe":
                return self.value.missing_values_in_dataframe
            case "retrieved":
                return self.value.missing_values_retrieved
            case "numpy":
                return self.value.missing_values_in_numpy_array

    def validate_return_type_safety(self) -> bool:
        """
        Validate that all functions return the promised types.
        
        This is a placeholder for runtime type validation.
        In a real implementation, you might want to add runtime checks.
        
        Returns:
            bool: True if type safety is validated, False otherwise.
        
        Note:
            This method is intended for future enhancement with actual runtime
            type checking capabilities.
        """
        return True

    def get_conversion_complexity(self) -> Literal["O(1)", "O(n)", "O(n log n)"]:
        """
        Return the computational complexity of converting this column type.
        
        Returns:
            Literal["O(1)", "O(n)", "O(n log n)"]: The computational complexity
                of type conversion operations for this column type.
        
        Example:
            >>> ColumnType.FLOAT_64.get_conversion_complexity()  # "O(n)"
        """
        return "O(n)"  # Most conversions are linear
    
    # ------------ Get values for dataframe ------------

    def get_value_for_dataframe(self, scalar_or_value: SCALAR_TYPE|VALUE_TYPE|None, unit_in_dataframe: Unit|None = None) -> Any:
        """
        Convert a scalar value to the appropriate format for DataFrame storage.
        
        This method handles type conversion, unit conversion, and missing value
        representation for storing values in pandas DataFrames.
        
        Args:
            scalar_or_value (SCALAR_TYPE|VALUE_TYPE|None): The scalar value to convert.
                Can be a united scalar, primitive type, or None for missing values.
            unit_in_dataframe (Unit|None): The target unit for the DataFrame.
                Required for column types with unit support, must be None for
                column types without unit support.
        
        Returns:
            Any: The value formatted for DataFrame storage with appropriate
                 type and missing value representation.
        
        Raises:
            ValueError: If the value type is incompatible with the column type,
                      or if unit requirements are not met.
            NotImplementedError: If the scalar type is not yet implemented.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> value = column_type.get_value_for_dataframe(3.14)
            >>> print(value)  # 3.14
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> unit = Unit("m")  # meters
            >>> value = column_type.get_value_for_dataframe(RealUnitedScalar(5.0, Unit("km")), unit)
            >>> print(value)  # 5000.0 (converted from km to m)
        """
        # Check if this is a united scalar (has canonical_value)
        if hasattr(scalar_or_value, 'canonical_value'):
            if scalar_or_value.canonical_value == self.value.missing_values_retrieved: # type: ignore
                if self.value.missing_values_in_dataframe is None:
                    raise ValueError(f"The column type {self} must not have missing values in dataframe.")
                return self.value.missing_values_in_dataframe

            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    if not isinstance(scalar_or_value, RealUnitedScalar):
                        raise ValueError(f"Value {scalar_or_value} is not a RealUnitedScalar.")
                    if unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is required for RealUnitedScalar.")
                    return unit_in_dataframe.from_canonical_value(scalar_or_value.canonical_value)
                
                case ColumnType.COMPLEX_NUMBER_128:
                    if not isinstance(scalar_or_value, ComplexUnitedScalar):
                        raise ValueError(f"Value {scalar_or_value} is not a ComplexUnitedScalar.")
                    if unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is required for ComplexUnitedScalar.")
                    raise NotImplementedError(f"ComplexUnitedScalar is not implemented.")
                
                case _:
                    raise NotImplementedError(f"Scalar {scalar_or_value} is not implemented.")
                    
        elif isinstance(scalar_or_value, VALUE_TYPE_RUNTIME):
            assert isinstance(scalar_or_value, VALUE_TYPE)
                    
            match self:
            
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32 | ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    if not isinstance(scalar_or_value, float|int):
                        raise ValueError(f"Value {scalar_or_value} is not a float.")
                    if self.has_unit and unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is required for this column type.")
                    elif not self.has_unit and unit_in_dataframe is not None:
                        raise ValueError(f"Unit in dataframe is not allowed for this column type.")
                    return float(scalar_or_value)                 

                case ColumnType.COMPLEX_NUMBER_128 | ColumnType.COMPLEX_128:
                    if not isinstance(scalar_or_value, complex):
                        raise ValueError(f"Value {scalar_or_value} is not a complex.")
                    if self.has_unit and unit_in_dataframe is None:
                        raise ValueError(f"Unit in dataframe is required for this column type.")
                    elif not self.has_unit and unit_in_dataframe is not None:
                        raise ValueError(f"Unit in dataframe is not allowed for this column type.")
                    return complex(scalar_or_value)
                
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    if not isinstance(scalar_or_value, int):
                        raise ValueError(f"Value {scalar_or_value} is not an int.")
                    if not self.has_unit and unit_in_dataframe is not None:
                        raise ValueError(f"Unit in dataframe is not allowed for this column type.")
                    return int(scalar_or_value)
                
                case ColumnType.STRING:
                    if not isinstance(scalar_or_value, str):
                        raise ValueError(f"Value {scalar_or_value} is not a str.")
                    if self.has_unit and unit_in_dataframe is not None:
                        raise ValueError(f"Unit in dataframe is not allowed for this column type.")
                    return str(scalar_or_value)
                
                case ColumnType.BOOL:
                    if not isinstance(scalar_or_value, bool):
                        raise ValueError(f"Value {scalar_or_value} is not a bool.")
                    if self.has_unit and unit_in_dataframe is not None:
                        raise ValueError(f"Unit in dataframe is not allowed for this column type.")
                    return bool(scalar_or_value)
                
                case ColumnType.TIMESTAMP:
                    if not isinstance(scalar_or_value, Timestamp):
                        raise ValueError(f"Value {scalar_or_value} is not a Timestamp.")
                    if self.has_unit and unit_in_dataframe is not None:
                        raise ValueError(f"Unit in dataframe is not allowed for this column type.")
                    return Timestamp(scalar_or_value)
                    
        else:
            raise ValueError(f"Invalid value: {scalar_or_value}")

    def get_pd_series_for_dataframe(self, array: Union[ARRAY_TYPE, np.ndarray, "pd.Series[Any]"], dataframe_unit: Optional[Unit], array_unit: Optional[Unit]) -> "pd.Series[Any]":
        """
        Convert an array to a pandas Series for DataFrame storage.
        
        This method handles type conversion, unit conversion, missing value
        representation, and proper dtype assignment for creating pandas Series
        that can be stored in DataFrames.
        
        Args:
            array (ARRAY_TYPE|np.ndarray|pd.Series[Any]): The array to convert.
                Can be a united array, numpy array, or pandas Series.
            dataframe_unit (Unit|None): The target unit for the DataFrame.
                Required for column types with unit support, must be None for
                column types without unit support.
            array_unit (Unit|None): The source unit of the array.
                Required if the array has units, must be None for arrays
                without units.
        
        Returns:
            pd.Series[Any]: A pandas Series with appropriate dtype and missing
                           value handling, ready for DataFrame storage.
        
        Raises:
            ValueError: If the array type is incompatible with the column type,
                      if unit requirements are not met, or if missing values
                      are not allowed for this column type.
            AssertionError: If the unit of the dataframe is not provided for a column with a unit.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> array = np.array([1.0, 2.0, 3.0])
            >>> series = column_type.get_pd_series_for_dataframe(array, None, None)
            >>> print(series.dtype)  # float64
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> array = RealUnitedArray([1.0, 2.0, 3.0], Unit("km"))
            >>> series = column_type.get_pd_series_for_dataframe(array, Unit("m"), Unit("km"))
            >>> print(series.iloc[0])  # 1000.0 (converted from km to m)
        """

        dtype: Dtype = self.value.dataframe_storage_type

        def create_series_from_np_array(np_array: np.ndarray) -> "pd.Series[Any]":
            # Only check for NaN values if the type supports it (numeric types)
            if self.is_numeric and np.isnan(np_array).any():
                if self.value.missing_values_in_dataframe is None:
                    raise ValueError(f"The column type {self} must not have missing values in dataframe.")
                np_array = np.where(np.isnan(np_array), self.value.missing_values_in_dataframe, np_array)
            return pd.Series(np_array, dtype=dtype)
        
        def create_series_from_series(series: "pd.Series[Any]") -> "pd.Series[Any]":
            if series.isna().any(): # type: ignore[reportUnknownReturnType]
                if self.value.missing_values_in_dataframe is None:
                    raise ValueError(f"The column type {self} must not have missing values in dataframe.")
                series = series.where(series.notna(), self.value.missing_values_in_dataframe)
            return series

        if isinstance(array, BaseArray):
            if array_unit is not None:
                raise ValueError(f"Unit in array is not allowed for BaseArray. It is already part of the BaseArray.")
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    if not isinstance(array, RealUnitedArray):
                        raise ValueError(f"Value {array} is not a RealUnitedArray.")
                    if dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is required for RealUnitedArray.")
                    np_array: np.ndarray = dataframe_unit.from_canonical_value(array.canonical_np_array)
                    return create_series_from_np_array(np_array)
                
                case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    if not isinstance(array, FloatArray):
                        raise ValueError(f"Value {array} is not a FloatArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for FloatArray.")
                    return create_series_from_np_array(array.canonical_np_array)
                
                case ColumnType.COMPLEX_NUMBER_128:
                    if not isinstance(array, ComplexUnitedArray):
                        raise ValueError(f"Value {array} is not a ComplexUnitedArray.")
                    if dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is required for ComplexUnitedArray.")
                    raise NotImplementedError(f"ComplexUnitedArray is not implemented.")
                
                case ColumnType.COMPLEX_128:
                    if not isinstance(array, ComplexArray):
                        raise ValueError(f"Value {array} is not a ComplexArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for complex.")
                    return create_series_from_np_array(array.canonical_np_array)
                
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    if not isinstance(array, IntArray):
                        raise ValueError(f"Value {array} is not an IntArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for IntArray.")
                    return create_series_from_np_array(array.canonical_np_array)
                
                case ColumnType.STRING:
                    if not isinstance(array, StringArray):
                        raise ValueError(f"Value {array} is not a StringArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for StringArray.")
                    return create_series_from_np_array(array.canonical_np_array)
                
                case ColumnType.BOOL:
                    if not isinstance(array, BoolArray):
                        raise ValueError(f"Value {array} is not a BoolArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for BoolArray.")
                    return create_series_from_np_array(array.canonical_np_array)
                
                case ColumnType.TIMESTAMP:
                    if not isinstance(array, TimestampArray):
                        raise ValueError(f"Value {array} is not a TimestampArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for TimestampArray.")
                    np_array = np.array(array)
                    return create_series_from_np_array(np_array)
                
        elif isinstance(array, np.ndarray):
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    if dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is required for RealUnitedArray.")
                    if array_unit is None:
                        raise ValueError(f"Unit of the numpy array is required for RealUnitedArray.")
                    np_array: np.ndarray = Unit.convert(array, array_unit, dataframe_unit) #type: ignore[reportUnknownReturnType]
                    return create_series_from_np_array(np_array)
                
                case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for FloatArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the numpy array is not allowed for FloatArray.")
                    return create_series_from_np_array(array)
                
                case ColumnType.COMPLEX_NUMBER_128:
                    if dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is required for ComplexUnitedArray.")
                    if array_unit is None:
                        raise ValueError(f"Unit of the numpy array is required for ComplexUnitedArray.")
                    np_array: np.ndarray = Unit.convert(array, array_unit, dataframe_unit) #type: ignore[reportUnknownReturnType]
                    return create_series_from_np_array(np_array)
                
                case ColumnType.COMPLEX_128:
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for complex.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the numpy array is not allowed for complex.")
                    return create_series_from_np_array(array)
                
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for IntArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the numpy array is not allowed for IntArray.")
                    return create_series_from_np_array(array)
                
                case ColumnType.STRING:
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for StringArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the numpy array is not allowed for StringArray.")
                    return create_series_from_np_array(array)
                
                case ColumnType.BOOL:
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for BoolArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the numpy array is not allowed for BoolArray.")
                    return create_series_from_np_array(array)
                
                case ColumnType.TIMESTAMP:
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for TimestampArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the numpy array is not allowed for TimestampArray.")
                    return create_series_from_np_array(array)
                
        elif isinstance(array, "pd.Series[Any]"): # type: ignore[reportUnknownReturnType]
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    if not isinstance(array, RealUnitedArray):
                        raise ValueError(f"Value {array} is not a RealUnitedArray.")
                    if dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is required for RealUnitedArray.")
                    if array_unit is None:
                        raise ValueError(f"Unit of the pandas series is required for RealUnitedArray.")
                    np_array: np.ndarray = Unit.convert(array, array_unit, dataframe_unit) #type: ignore[reportUnknownReturnType]
                    return create_series_from_np_array(np_array)
                
                case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    if not isinstance(array, FloatArray):
                        raise ValueError(f"Value {array} is not a FloatArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for FloatArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the pandas series is not allowed for FloatArray.")
                    return create_series_from_series(array)
                
                case ColumnType.COMPLEX_NUMBER_128:
                    if not isinstance(array, ComplexUnitedArray):
                        raise ValueError(f"Value {array} is not a ComplexUnitedArray.")
                    if dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is required for ComplexUnitedArray.")
                    if array_unit is None:
                        raise ValueError(f"Unit of the pandas series is required for ComplexUnitedArray.")
                    np_array: np.ndarray = Unit.convert(array, array_unit, dataframe_unit) #type: ignore[reportUnknownReturnType]
                    return create_series_from_np_array(np_array)
                
                case ColumnType.COMPLEX_128:
                    if not isinstance(array, complex):
                        raise ValueError(f"Value {array} is not a complex.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for complex.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the pandas series is not allowed for complex.")
                    return create_series_from_series(array)
                
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    if not isinstance(array, IntArray):
                        raise ValueError(f"Value {array} is not an IntArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for IntArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the pandas series is not allowed for IntArray.")
                    return create_series_from_series(array)
                
                case ColumnType.STRING:
                    if not isinstance(array, StringArray):
                        raise ValueError(f"Value {array} is not a StringArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for StringArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the pandas series is not allowed for StringArray.")
                    return create_series_from_series(array)
                
                case ColumnType.BOOL:
                    if not isinstance(array, BoolArray):
                        raise ValueError(f"Value {array} is not a BoolArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for BoolArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the pandas series is not allowed for BoolArray.")
                    return create_series_from_series(array)
                
                case ColumnType.TIMESTAMP:
                    if not isinstance(array, TimestampArray):
                        raise ValueError(f"Value {array} is not a TimestampArray.")
                    if not dataframe_unit is None:
                        raise AssertionError(f"Unit in dataframe is not allowed for TimestampArray.")
                    if array_unit is not None:
                        raise ValueError(f"Unit of the pandas series is not allowed for TimestampArray.")
                    return create_series_from_series(array)
        else:
            raise ValueError(f"Value {array} is not a valid array.")
            
    # ------------ Get values from dataframe ------------

    def get_value_from_pd_series(self, pandas_series: "pd.Series[Any]", row_index: int, source_and_target_unit: Optional[tuple[Unit, Unit]] = None) -> float|complex|str|bool|int|Timestamp:
        """
        Extract a single value from a pandas Series with proper type conversion.
        
        This method retrieves a value from a pandas Series at the specified row index,
        handles missing values appropriately, and performs unit conversion if needed.
        
        Args:
            pandas_series (pd.Series[Any]): The pandas Series to extract the value from.
            row_index (int): The row index to get the value from (0-based).
            source_and_target_unit (Optional[tuple[Unit, Unit]]): Unit conversion parameters.
                If provided, converts from source unit to target unit.
                Only applicable for column types with unit support.
        
        Returns:
            float|complex|str|bool|int|Timestamp: The extracted value with appropriate
                type and missing value representation.
        
        Raises:
            ValueError: If missing values are not allowed for this column type,
                      or if unit conversion is requested for non-unit column types.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> series = pd.Series([1.0, 2.0, 3.0])
            >>> value = column_type.get_value_from_pd_series(series, 0)
            >>> print(value)  # 1.0
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> series = pd.Series([1000.0, 2000.0, 3000.0])  # values in meters
            >>> value = column_type.get_value_from_pd_series(series, 0, (Unit("m"), Unit("km")))
            >>> print(value)  # 1.0 (converted from meters to kilometers)
        """

        value: Any = pandas_series.iloc[row_index]

        if pd.isna(value):
            if self.value.missing_values_retrieved is None:
                raise ValueError(f"The column type {self} must not have missing values in dataframe.")
            return self.value.missing_values_retrieved
        
        if not self.has_unit and source_and_target_unit is not None:
                raise ValueError(f"Unit in dataframe is not allowed for this column type.")
        
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if source_and_target_unit is None:
                    return float(value)
                else:
                    return Unit.convert(value, source_and_target_unit[0], source_and_target_unit[1]) # type: ignore[reportUnknownReturnType]
            
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return float(value)

            case ColumnType.COMPLEX_NUMBER_128:
                if source_and_target_unit is None:
                    return complex(value)
                else:
                    return Unit.convert(value, source_and_target_unit[0], source_and_target_unit[1]) # type: ignore[reportUnknownReturnType]
            
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
            
    def get_scalar_from_pd_series(self, pandas_series: "pd.Series[Any]", row_index: int, unit_in_dataframe: Optional[Unit] = None) -> SCALAR_TYPE:
        """
        Extract a scalar value from a pandas Series with proper type conversion.
        
        This method retrieves a scalar value from a pandas Series at the specified row index,
        handles missing values appropriately, and returns the value in the appropriate scalar type
        (including united scalars for column types with unit support).
        
        Args:
            pandas_series (pd.Series[Any]): The pandas Series to extract the scalar from.
            row_index (int): The row index to get the value from (0-based).
            unit_in_dataframe (Optional[Unit]): The unit context for the DataFrame.
                Required for column types with unit support, must be None for
                column types without unit support.
        
        Returns:
            SCALAR_TYPE: The extracted scalar value with appropriate type.
                For united column types, returns a united scalar.
                For primitive column types, returns the primitive type.
        
        Raises:
            ValueError: If missing values are not allowed for this column type,
                      or if unit requirements are not met.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> series = pd.Series([1.0, 2.0, 3.0])
            >>> scalar = column_type.get_scalar_from_pd_series(series, 0)
            >>> print(scalar)  # 1.0
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> series = pd.Series([1000.0, 2000.0, 3000.0])  # values in meters
            >>> scalar = column_type.get_scalar_from_pd_series(series, 0, Unit("m"))
            >>> print(scalar)  # RealUnitedScalar(1000.0, Unit("m"))
        """
        
        if self.has_unit and unit_in_dataframe is None:
            raise ValueError(f"Unit is required for {self}.")
        elif not self.has_unit and unit_in_dataframe is not None:
            raise ValueError(f"Unit in dataframe is not allowed for this column type.")
            
        value: Any = pandas_series.iloc[row_index]

        if pd.isna(value):
            if self.value.missing_values_retrieved is None:
                raise ValueError(f"The column type {self} must not have missing values in dataframe.")
            value = self.value.missing_values_retrieved

        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                assert unit_in_dataframe is not None
                return RealUnitedScalar(value, unit_in_dataframe.dimension, unit_in_dataframe)
            
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return float(value)
            
            case ColumnType.COMPLEX_NUMBER_128:
                assert unit_in_dataframe is not None
                return ComplexUnitedScalar(value, unit_in_dataframe.dimension, unit_in_dataframe)
            
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
            
    def get_numpy_array_from_pd_series(self, pandas_series: "pd.Series[Any]", source_and_target_unit: Optional[tuple[Unit, Unit]] = None) -> np.ndarray[Any, Any]: # type: ignore[reportUnknownReturnType]
        """
        Convert a pandas Series to a numpy array with proper type handling.
        
        This method converts a pandas Series to a numpy array with appropriate dtype,
        handles missing values correctly, and performs unit conversion if needed.
        
        Args:
            pandas_series (pd.Series[Any]): The pandas Series to convert to numpy array.
            source_and_target_unit (Optional[tuple[Unit, Unit]]): Unit conversion parameters.
                If provided, converts from source unit to target unit.
                Only applicable for column types with unit support.
        
        Returns:
            np.ndarray[Any, Any]: A numpy array with appropriate dtype and missing
                                 value handling.
        
        Raises:
            ValueError: If missing values are not allowed for this column type,
                      or if unit conversion is requested for non-unit column types.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> series = pd.Series([1.0, 2.0, 3.0])
            >>> array = column_type.get_numpy_array_from_pd_series(series)
            >>> print(array.dtype)  # float64
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> series = pd.Series([1000.0, 2000.0, 3000.0])  # values in meters
            >>> array = column_type.get_numpy_array_from_pd_series(series, (Unit("m"), Unit("km")))
            >>> print(array)  # [1.0, 2.0, 3.0] (converted from meters to kilometers)
        """

        if not self.has_unit and source_and_target_unit is not None:
            raise ValueError(f"Unit in dataframe is not allowed for this column type.")

        if pandas_series.isna().any() and self.value.missing_values_in_numpy_array is None: # type: ignore[reportUnknownReturnType]
            raise ValueError(f"The column type {self} must not have missing values in dataframe as missing values in the numpy array are not allowed.")
    
        numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=self.numpy_storage_type, na_value=self.value.missing_values_in_numpy_array, copy=True) # type: ignore[reportUnknownReturnType]
        
        match self:
            case ColumnType.REAL_NUMBER_64|ColumnType.REAL_NUMBER_32:
                if source_and_target_unit is not None:
                    numpy_array: np.ndarray[Any, Any] = Unit.convert(numpy_array, source_and_target_unit[0], source_and_target_unit[1]) # type: ignore[reportUnknownReturnType]
                return numpy_array
            
            case ColumnType.FLOAT_64|ColumnType.FLOAT_32:
                return numpy_array
            
            case ColumnType.COMPLEX_NUMBER_128:
                if source_and_target_unit is not None:
                    numpy_array: np.ndarray[Any, Any] = Unit.convert(numpy_array, source_and_target_unit[0], source_and_target_unit[1]) # type: ignore[reportUnknownReturnType]
                return numpy_array
            
            case ColumnType.COMPLEX_128:
                return numpy_array
            
            case ColumnType.INTEGER_64|ColumnType.INTEGER_32|ColumnType.INTEGER_16|ColumnType.INTEGER_8:
                return numpy_array

            case ColumnType.STRING:
                return numpy_array
            
            case ColumnType.BOOL:
                return numpy_array
            
            case ColumnType.TIMESTAMP:
                return numpy_array
            
    def get_array_from_pd_series(self, pandas_series: "pd.Series[Any]", unit_in_dataframe: Optional[Unit] = None) -> ARRAY_TYPE: # type: ignore[reportUnknownReturnType]
        """
        Convert a pandas Series to an appropriate array type.
        
        This method converts a pandas Series to the appropriate array type for this
        column type, including united arrays for column types with unit support.
        
        Args:
            pandas_series (pd.Series[Any]): The pandas Series to convert to array.
            unit_in_dataframe (Optional[Unit]): The unit context for the DataFrame.
                Required for column types with unit support, must be None for
                column types without unit support.
        
        Returns:
            ARRAY_TYPE: An array of the appropriate type for this column type.
                For united column types, returns a united array.
                For primitive column types, returns the corresponding array type.
        
        Raises:
            ValueError: If missing values are not allowed for this column type,
                      or if unit requirements are not met.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> series = pd.Series([1.0, 2.0, 3.0])
            >>> array = column_type.get_array_from_pd_series(series)
            >>> print(type(array))  # <class 'FloatArray'>
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> series = pd.Series([1000.0, 2000.0, 3000.0])  # values in meters
            >>> array = column_type.get_array_from_pd_series(series, Unit("m"))
            >>> print(type(array))  # <class 'RealUnitedArray'>
        """

        # Only check for missing values if the data actually contains missing values
        if pandas_series.isna().any() and self.value.missing_values_in_numpy_array is None: # type: ignore[reportUnknownReturnType]
            raise ValueError(f"The column type {self} must not have missing values in dataframe as missing values for the respective array type are not allowed.")
        
        # Only pass na_value if it's not None
        if self.value.missing_values_in_numpy_array is None:
            numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=self.numpy_storage_type, copy=True) # type: ignore[reportUnknownReturnType]
        else:
            numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=self.numpy_storage_type, na_value=self.value.missing_values_in_numpy_array, copy=True) # type: ignore[reportUnknownReturnType]
        
        if not self.has_unit and unit_in_dataframe is not None:
            raise ValueError(f"Unit in dataframe is not allowed for this column type.")

        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                return RealUnitedArray(numpy_array, unit_in_dataframe)
            
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return FloatArray(numpy_array)
            
            case ColumnType.COMPLEX_NUMBER_128:
                return ComplexUnitedArray(numpy_array, unit_in_dataframe) # type: ignore[reportUnknownReturnType]
            
            case ColumnType.COMPLEX_128:
                return ComplexArray(numpy_array)
            
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return IntArray(numpy_array)
            
            case ColumnType.STRING:
                return StringArray(numpy_array)
            
            case ColumnType.BOOL:
                return BoolArray(numpy_array)
            
            case ColumnType.TIMESTAMP:
                return TimestampArray(numpy_array)
            
    def get_list_from_pd_series(self, pandas_series: "pd.Series[Any]", source_and_target_unit: Optional[tuple[Unit, Unit]] = None) -> list[float]|list[complex]|list[int]|list[str]|list[bool]|list[Timestamp]: # type: ignore[reportUnknownReturnType]
        """
        Convert a pandas Series to a Python list with proper type handling.
        
        This method converts a pandas Series to a Python list with appropriate types,
        handles missing values correctly, and performs unit conversion if needed.
        
        Args:
            pandas_series (pd.Series[Any]): The pandas Series to convert to list.
            source_and_target_unit (Optional[tuple[Unit, Unit]]): Unit conversion parameters.
                If provided, converts from source unit to target unit.
                Only applicable for column types with unit support.
        
        Returns:
            list[float]|list[complex]|list[int]|list[str]|list[bool]|list[Timestamp]: 
                A Python list with appropriate types and missing value handling.
        
        Raises:
            ValueError: If missing values are not allowed for this column type,
                      or if unit conversion is requested for non-unit column types.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> series = pd.Series([1.0, 2.0, 3.0])
            >>> lst = column_type.get_list_from_pd_series(series)
            >>> print(lst)  # [1.0, 2.0, 3.0]
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> series = pd.Series([1000.0, 2000.0, 3000.0])  # values in meters
            >>> lst = column_type.get_list_from_pd_series(series, (Unit("m"), Unit("km")))
            >>> print(lst)  # [1.0, 2.0, 3.0] (converted from meters to kilometers)
        """

        if pandas_series.isna().any() and self.value.missing_values_retrieved is None: # type: ignore[reportUnknownReturnType]
            raise ValueError(f"The column type {self} must not have missing values in dataframe as missing values for the respective list type are not allowed.")
        
        if not self.has_unit and source_and_target_unit is not None:
                raise ValueError(f"Unit in dataframe is not allowed for this column type.")
        
        numpy_array: np.ndarray[Any, Any] = pandas_series.to_numpy(dtype=self.numpy_storage_type, na_value=self.value.missing_values_in_numpy_array, copy=True) # type: ignore[reportUnknownReturnType]
        
        match self:
            case ColumnType.REAL_NUMBER_64|ColumnType.REAL_NUMBER_32:
                if source_and_target_unit is not None:
                    numpy_array: np.ndarray[Any, Any] = Unit.convert(numpy_array, source_and_target_unit[0], source_and_target_unit[1]) # type: ignore[reportUnknownReturnType]
                return numpy_array.tolist()
            
            case ColumnType.FLOAT_64|ColumnType.FLOAT_32:
                return numpy_array.tolist()
            
            case ColumnType.COMPLEX_NUMBER_128:
                if source_and_target_unit is not None:
                    numpy_array: np.ndarray[Any, Any] = Unit.convert(numpy_array, source_and_target_unit[0], source_and_target_unit[1]) # type: ignore[reportUnknownReturnType]
                return numpy_array.tolist()
            
            case ColumnType.COMPLEX_128:
                return numpy_array.tolist()
            
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return numpy_array.tolist()
            
            case ColumnType.STRING:
                return numpy_array.tolist()
            
            case ColumnType.BOOL:
                return numpy_array.tolist()
            
            case ColumnType.TIMESTAMP:
                missing_value: Timestamp = ColumnType.TIMESTAMP.value.missing_values_retrieved # type: ignore[reportUnknownReturnType]
                assert isinstance(missing_value, Timestamp)
                return [Timestamp(item) if item is not None else missing_value for item in numpy_array.tolist()]
            
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
        """
        Infer the appropriate ColumnType from a scalar/array type or pandas Series.
        
        This method determines the most suitable ColumnType based on the input type
        or pandas Series dtype. It handles both scalar types and array types,
        and can resolve ambiguity using precision specifications.
        
        Args:
            scalar_or_array_type (type[SCALAR_TYPE]|type[ARRAY_TYPE]|pd.Series): 
                The type to infer from, or a pandas Series to infer from its dtype.
            precision (Literal[8, 16, 32, 64, 128, 256]|None): 
                Optional precision specification to resolve ambiguity when multiple
                column types match the input type.
        
        Returns:
            ColumnType: The inferred column type that best matches the input.
        
        Raises:
            ValueError: If no matching column type is found, or if multiple matches
                      exist without precision specification.
        
        Example:
            >>> # Infer from pandas Series
            >>> series = pd.Series([1.0, 2.0, 3.0], dtype=float)
            >>> column_type = ColumnType.infer_approbiate_column_type(series)
            >>> print(column_type)  # ColumnType.FLOAT_64
            
            >>> # Infer from scalar type
            >>> column_type = ColumnType.infer_approbiate_column_type(float)
            >>> print(column_type)  # ColumnType.FLOAT_64
            
            >>> # Infer from array type with precision
            >>> column_type = ColumnType.infer_approbiate_column_type(RealUnitedArray, precision=32)
            >>> print(column_type)  # ColumnType.REAL_NUMBER_32
        """
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
        
    # ------------ Check the compatibility of a value, scalar or array with the column type ------------

    def check_item_compatibility(self, item: VALUE_TYPE | SCALAR_TYPE | ARRAY_TYPE, column_unit: Optional[Unit]) -> bool:
        """
        Check if a value, scalar or array is compatible with this column type.
        
        This method verifies whether a given value can be stored in a column of this type,
        including unit compatibility checks for united types.
        
        Args:
            item (VALUE_TYPE | SCALAR_TYPE | ARRAY_TYPE): 
                The value to check for compatibility.
            column_unit (Optional[Unit]): 
                Optional unit to check against for united types.
                If provided, the value's unit must be compatible with this unit.
        
        Returns:
            bool: True if the value is compatible with this column type, False otherwise.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> is_compatible = column_type.check_item_compatibility(3.14)
            >>> print(is_compatible)  # True
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> scalar = RealUnitedScalar(5.0, Unit("m"))
            >>> is_compatible = column_type.check_item_compatibility(scalar, Unit("km"))
            >>> print(is_compatible)  # True (meters are compatible with kilometers)
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> scalar = RealUnitedScalar(5.0, Unit("m"))
            >>> is_compatible = column_type.check_item_compatibility(scalar)
            >>> print(is_compatible)  # True (no unit validation)
        """
        if self.has_unit:
            if not isinstance(item, HasUnit):
                return False
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if isinstance(item, RealUnitedScalar) or isinstance(item, RealUnitedArray):
                    if column_unit is not None:
                        if not item.unit.compatible_to(column_unit):
                            return False
                    return True
                else:
                    return False
            case ColumnType.COMPLEX_NUMBER_128:
                if isinstance(item, ComplexUnitedScalar) or isinstance(item, ComplexUnitedArray):
                    if column_unit is not None:
                        if not item.unit.compatible_to(column_unit):
                            return False
                    return True
                else:
                    return False
            case ColumnType.STRING:
                if isinstance(item, str) or isinstance(item, StringArray):
                    return True
                else:
                    return False
            case ColumnType.BOOL:
                if isinstance(item, bool) or isinstance(item, BoolArray):
                    return True
                else:
                    return False
            case ColumnType.TIMESTAMP:
                if isinstance(item, Timestamp) or isinstance(item, TimestampArray):
                    return True
                else:
                    return False
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                if isinstance(item, int) or isinstance(item, IntArray):
                    return True
                else:
                    return False
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                if isinstance(item, float) or isinstance(item, FloatArray):
                    return True
                else:
                    return False
            case ColumnType.COMPLEX_128:
                if isinstance(item, complex) or isinstance(item, ComplexUnitedArray):
                    return True
                else:
                    return False
                
    def check_type_compatibility(self, value_type: type, check_kind: Literal["value", "scalar", "array"]) -> bool:
        """
        Check if a type is compatible with this column type.
        
        This method verifies whether a given type matches the expected
        type for this column type based on the specified check kind.
        
        Args:
            value_type (type): The type to check for compatibility.
            check_kind (Literal["value", "scalar", "array"]): The kind of type to check:
                - "value": Check against the column's value type (e.g., float, str)
                - "scalar": Check against the column's scalar type (e.g., RealUnitedScalar, float)
                - "array": Check against the column's array type (e.g., RealUnitedArray, FloatArray)
        
        Returns:
            bool: True if the type is compatible with this column type, False otherwise.
        
        Example:
            >>> column_type = ColumnType.FLOAT_64
            >>> is_compatible = column_type.check_type_compatibility(float, "value")
            >>> print(is_compatible)  # True
            
            >>> column_type = ColumnType.REAL_NUMBER_64
            >>> is_compatible = column_type.check_type_compatibility(RealUnitedScalar, "scalar")
            >>> print(is_compatible)  # True
            
            >>> column_type = ColumnType.FLOAT_64
            >>> is_compatible = column_type.check_type_compatibility(FloatArray, "array")
            >>> print(is_compatible)  # True
        """
        
        match check_kind:
            case "value":
                match self:
                    case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                        return value_type == float
                    case ColumnType.COMPLEX_NUMBER_128:
                        return value_type == complex
                    case ColumnType.STRING:
                        return value_type == str
                    case ColumnType.BOOL:
                        return value_type == bool
                    case ColumnType.TIMESTAMP:
                        return value_type == Timestamp
                    case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                        return value_type == int
                    case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                        return value_type == float
                    case ColumnType.COMPLEX_128:
                        return value_type == complex
            case "scalar":
                match self:
                    case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                        return issubclass(value_type, RealUnitedScalar)
                    case ColumnType.COMPLEX_NUMBER_128:
                        return issubclass(value_type, ComplexUnitedScalar)
                    case ColumnType.STRING:
                        return issubclass(value_type, str)
                    case ColumnType.BOOL:
                        return issubclass(value_type, bool)
                    case ColumnType.TIMESTAMP:
                        return issubclass(value_type, Timestamp)
                    case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                        return issubclass(value_type, int)
                    case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                        return issubclass(value_type, float)
                    case ColumnType.COMPLEX_128:
                        return issubclass(value_type, complex)
            case "array":
                match self:
                    case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                        return issubclass(value_type, RealUnitedArray)
                    case ColumnType.COMPLEX_NUMBER_128:
                        return issubclass(value_type, ComplexUnitedArray)
                    case ColumnType.STRING:
                        return issubclass(value_type, StringArray)
                    case ColumnType.BOOL:
                        return issubclass(value_type, BoolArray)
                    case ColumnType.TIMESTAMP:
                        return issubclass(value_type, TimestampArray)
                    case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                        return issubclass(value_type, IntArray)
                    case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                        return issubclass(value_type, FloatArray)
                    case ColumnType.COMPLEX_128:
                        return issubclass(value_type, ComplexArray)
            case _: # type: ignore[unreachable]
                raise ValueError(f"Invalid check kind: {check_kind}")

    def __reduce_ex__(self, _: Any):
        """
        Support for pickle serialization of ColumnType enum values.
        
        This method enables proper serialization and deserialization of ColumnType
        instances when using Python's pickle module.
        
        Args:
            _: Protocol version (unused, required by pickle interface).
        
        Returns:
            tuple: Pickle-compatible representation of the enum value.
        """
        """Custom pickle reduction to preserve enum identity by name."""
        return (getattr, (self.__class__, self.name))