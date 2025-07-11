"""
Column operations mixin for UnitedDataframe.

Contains all operations related to column CRUD operations, including
getting the column data as a pandas Series, numpy array, or array.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import Any, TypeVar, overload
import pandas as pd
import numpy as np
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..column_type import ColumnType, ARRAY_TYPE
from ..accessors._column_accessor import ColumnAccessor

AT = TypeVar("AT", bound=ARRAY_TYPE)

class ColumnAccessMixin(UnitedDataframeProtocol[CK]):
    """
    Column access mixin for UnitedDataframe.
    
    Provides all functionality related to column access operations, including
    getting the column data as a pandas Series, numpy array, or array.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    def _column_get_as_pd_series(self, column_key: CK, slice: slice|None = None) -> pd.Series[Any]:
        """
        Internal: Get the column data for a column as a pandas Series. (no lock)
        """
        if not self._colkey_exists(column_key):
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        internal_column_name: str = self._internal_dataframe_column_names[column_key]
        if slice is not None:
            pd_series: pd.Series[Any] = self._internal_dataframe[internal_column_name].iloc[slice] # type: ignore[no-any-return]
        else:
            pd_series: pd.Series[Any] = self._internal_dataframe[internal_column_name]
        return pd_series
    
    def column_get_as_pd_series(self, column_key: CK, slice: slice|None = None) -> pd.Series[Any]:
        """
        Get the column data for a column as a pandas Series.
        """
        with self._rlock:
            return self._column_get_as_pd_series(column_key, slice)

    def _column_get_as_numpy_array(self, column_key: CK, slice: slice|None = None) -> np.ndarray[Any, Any]:
        """
        Internal: Get the column data for a column as a numpy array. (no lock)
        """
        if not self._colkey_exists(column_key):
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        internal_column_name: str = self._internal_dataframe_column_names[column_key]
        if slice is not None:
            pd_series: pd.Series[Any] = self._internal_dataframe[internal_column_name].iloc[slice] # type: ignore[no-any-return]
        else:
            pd_series: pd.Series[Any] = self._internal_dataframe[internal_column_name]
        return pd_series.to_numpy() # type: ignore[no-any-return]

    def column_get_as_numpy_array(self, column_key: CK, slice: slice|None = None) -> np.ndarray:
        """
        Get the column data for a column as a numpy array.
        """
        with self._rlock:
            return self._column_get_as_numpy_array(column_key, slice)
        
    @overload
    def column_get_as_array(self, column_key: CK, expected_column_type: None = None, slice: slice|None = None) -> ARRAY_TYPE: ...
    @overload
    def column_get_as_array(self, column_key: CK, expected_column_type: type[AT], slice: slice|None = None) -> AT: ...
    def column_get_as_array(self, column_key: CK, expected_column_type: type[AT]|None = None, slice: slice|None = None) -> AT|ARRAY_TYPE:
        """
        Get the column data for a column.
        """
        with self._rlock:
            if expected_column_type is not None:
                if not self._column_types[column_key].check_array_type(expected_column_type):
                    raise ValueError(f"Column {column_key} is not a {expected_column_type} column.")
                result: AT = self._column_get_as_array(column_key, expected_column_type, slice) # type: ignore
                return result
            else:
                result: ARRAY_TYPE = self._column_get_as_array(column_key, slice) # type: ignore
                return result # type: ignore[no-any-return]

    @overload
    def _column_get_as_array(self, column_key: CK, expected_column_type: None = None, slice: slice|None = None) -> ARRAY_TYPE: ...
    @overload
    def _column_get_as_array(self, column_key: CK, expected_column_type: type[AT], slice: slice|None = None) -> AT: ...
    def _column_get_as_array(self, column_key: CK, expected_column_type: type[AT]|None = None, slice: slice|None = None) -> AT|ARRAY_TYPE:
        """
        Internal: Get the column data for a column. (no lock, no read-only check)
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BaseArray: The column data
        """

        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name: str = self._internal_dataframe_column_names[column_key]
        column_type: ColumnType = self._column_types[column_key]
        if slice is not None:
            pandas_series: pd.Series[Any] = self._internal_dataframe[internal_column_name].iloc[slice] # type: ignore[no-any-return]
        else:
            pandas_series: pd.Series[Any] = self._internal_dataframe[internal_column_name]

        array: ARRAY_TYPE = column_type.get_array_from_dataframe(pandas_series, self._column_units[column_key])
        return array
    
    def _column_get_as_column_accessor(self, column_key: CK, slice: slice|None = None) -> ColumnAccessor[CK]:
        """
        Internal: Get the column data for a column as a column accessor. (no lock, no read-only check)
        """
        return ColumnAccessor[CK](self, column_key, slice) # type: ignore[no-any-return]

    def _columns_get_with_units(self) -> list[CK]:
        """
        Internal: Get the column keys for all columns with units. (no lock, no read-only check)
        """
        return [column_key for column_key in self._column_keys if self._column_units[column_key] != None]
    
    def column_get_with_units(self) -> list[CK]:
        """
        Get the column keys for all columns with units.
        """
        with self._rlock:
            return self._columns_get_with_units()

    def _columns_get_without_units(self) -> list[CK]:
        """
        Internal: Get the column keys for all columns without units. (no lock, no read-only check)
        """
        return [column_key for column_key in self._column_keys if self._column_units[column_key] == None]
    
    def column_get_without_units(self) -> list[CK]:
        """
        Get the column keys for all columns without units.
        """
        with self._rlock:
            return self._columns_get_without_units()