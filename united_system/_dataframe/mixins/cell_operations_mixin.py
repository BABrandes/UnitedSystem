"""
Cell operations mixin for UnitedDataframe.

Contains all operations related to individual cell operations,
including getting and setting individual cell values.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, Any, TypeVar, overload, Sequence, Optional
from ..._units_and_dimension.unit import Unit
from ..._dataframe.column_type import ColumnType
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._utils.general import VALUE_TYPE, SCALAR_TYPE, ARRAY_TYPE
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

VT = TypeVar("VT", bound=VALUE_TYPE)
ST = TypeVar("ST", bound=SCALAR_TYPE)
AT = TypeVar("AT", bound=ARRAY_TYPE)

class CellOperationsMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Cell operations mixin for UnitedDataframe.
    
    Provides all functionality related to individual cell operations,
    including getting and setting individual cell values.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    ########################################################
    # Cell Operations: Get
    ########################################################

    @overload
    def cell_get_scalar(self, row_index: int, column_key: CK) -> SCALAR_TYPE: ...
    @overload
    def cell_get_scalar(self, row_index: int, column_key: CK, expected_type: type[ST]) -> ST: ...
    def cell_get_scalar(self, row_index: int, column_key: CK, expected_type: type[ST]|None = None) -> ST|SCALAR_TYPE:
        """
        Get the value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
        """
        
        with self._rlock:
            if expected_type is not None:
                if not self._column_types[column_key].check_scalar_type(expected_type):
                    raise ValueError(f"Column {column_key} is not a {expected_type} column.")
                result: ST = self._cell_get_scalar(row_index, column_key) # type: ignore
                return result
            else:
                result: SCALAR_TYPE = self._cell_get_scalar(row_index, column_key) # type: ignore
                return result

    def _cell_get_scalar(self, row_index: int, column_key: CK) -> SCALAR_TYPE:
        """
        Internal: Get the value of a specific cell. (no lock)
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Returns:
            SCALAR_TYPE: The cell value
        """
        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        raw_value: VALUE_TYPE = self._internal_dataframe.loc[row_index, self._internal_dataframe_column_names[column_key]] # type: ignore
        result: SCALAR_TYPE = self._column_types[column_key].get_scalar_value_from_dataframe(raw_value, self._column_units[column_key]) # type: ignore
        return result
    
    @overload
    def cell_get_value(self, row_index: int, column_key: CK) -> VALUE_TYPE: ...
    @overload
    def cell_get_value(self, row_index: int, column_key: CK, expected_type: type[VT]) -> VT: ...
    def cell_get_value(self, row_index: int, column_key: CK, expected_type: type[VT]|None = None) -> VT|VALUE_TYPE:
        """
        Get the lowlevel value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
        """
        
        with self._rlock:
            if expected_type is not None:
                if not self._column_types[column_key].check_scalar_type(expected_type):
                    raise ValueError(f"Column {column_key} is not a {expected_type} column.")
                result: VT = self._cell_get_value(row_index, column_key) # type: ignore
                return result
            else:
                result: VALUE_TYPE = self._cell_get_value(row_index, column_key) # type: ignore
                return result

    def _cell_get_value(self, row_index: int, column_key: CK) -> VALUE_TYPE:
        """
        Internal: Get the lowlevel value of a specific cell. (no lock)
        """
        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        raw_value: VALUE_TYPE = self._internal_dataframe.loc[row_index, self._internal_dataframe_column_names[column_key]] # type: ignore
        return raw_value
    
    @overload
    def cell_get_scalars(self, row_indices: Sequence[int], column_key: CK) -> list[SCALAR_TYPE]: ...
    @overload
    def cell_get_scalars(self, row_indices: Sequence[int], column_key: CK, expected_type: type[ST]) -> list[ST]: ...
    def cell_get_scalars(self, row_indices: Sequence[int], column_key: CK, expected_type: type[ST]|None = None) -> list[ST]|list[SCALAR_TYPE]:
        """
        Get multiple cell values from a single column.

        Args:
            row_indices (Sequence[int]): List of row indices
            column_key (CK): The column key
            expected_type (type[ST]|None): The expected type of the scalars

        Returns:
            list[ST]: List of cell values
        """

        with self._rlock:
            if expected_type is not None:
                if not self._column_types[column_key].check_scalar_type(expected_type):
                    raise ValueError(f"Column {column_key} is not a {expected_type} column.")
                result: list[ST] = self._cell_get_scalars(row_indices, column_key) # type: ignore
                return result
            else:
                result: list[SCALAR_TYPE] = self._cell_get_scalars(row_indices, column_key) # type: ignore
                return result

    def _cell_get_scalars(self, row_indices: list[int], column_key: CK) -> list[SCALAR_TYPE]:
        """
        Internal: Get multiple cell values from a single column. (no lock)
        
        Args:
            row_indices (list[int]): List of row indices
            column_key (CK): The column key
            
        Returns:
            list[SCALAR_TYPE]: List of cell values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Check all row indices are valid
        for row_index in row_indices:
            if row_index < 0 or row_index >= len(self._internal_dataframe):
                raise ValueError(f"Row index {row_index} is out of bounds.")
            
        list_of_values: list[SCALAR_TYPE] = []
        for row_index in row_indices:
            list_of_values.append(self._cell_get_scalar(row_index, column_key))
        return list_of_values

    def _cell_get_array(self, first_row_index: int, last_row_index: int, column_key: CK, expected_type: type[AT]|None = None) -> AT|ARRAY_TYPE:
        """
        Internal: Get an array of values from a single column. (no lock)

        Args:
            first_row_index (int): The first row index
            last_row_index (int): The last row index
            column_key (CK): The column key
            expected_type (type[AT]|None): The expected type of the array

        Returns:
            AT: The array of values
            
        Raises:
            ValueError: If the parameters are invalid
        """
        if first_row_index < 0 or first_row_index >= len(self._internal_dataframe):
            raise ValueError(f"First row index {first_row_index} is out of bounds.")
        if last_row_index < 0 or last_row_index >= len(self._internal_dataframe):
            raise ValueError(f"Last row index {last_row_index} is out of bounds.")
        if first_row_index > last_row_index:
            raise ValueError(f"First row index {first_row_index} cannot be greater than last row index {last_row_index}.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        if expected_type is not None:
            if not self._column_types[column_key].check_array_type(expected_type):
                raise ValueError(f"Column {column_key} is not a {expected_type} column.")
            result: AT = self._column_types[column_key].get_array_from_dataframe(self._internal_dataframe.loc[first_row_index:last_row_index, self._internal_dataframe_column_names[column_key]], self._column_units[column_key]) # type: ignore
            return result
        else:
            result: ARRAY_TYPE = self._column_types[column_key].get_array_from_dataframe(self._internal_dataframe.loc[first_row_index:last_row_index, self._internal_dataframe_column_names[column_key]], self._column_units[column_key]) # type: ignore
            return result
        
    @overload
    def cell_get_array(self, first_row_index: int, last_row_index: int, column_key: CK) -> ARRAY_TYPE: ...
    @overload
    def cell_get_array(self, first_row_index: int, last_row_index: int, column_key: CK, expected_type: type[AT]) -> AT: ...
    def cell_get_array(self, first_row_index: int, last_row_index: int, column_key: CK, expected_type: type[AT]|None = None) -> AT|ARRAY_TYPE:
        """
        Get an array of values from a single column.

        Args:
            first_row_index (int): The first row index
            last_row_index (int): The last row index
            column_key (CK): The column key
            expected_type (type[AT]|None): The expected type of the array

        Returns:
            AT: The array of values

        Raises:
            ValueError: If the parameters are invalid
        """
        with self._rlock:
            if expected_type is not None:
                if not self._column_types[column_key].check_array_type(expected_type):
                    raise ValueError(f"Column {column_key} is not a {expected_type} column.")
                result: AT = self._cell_get_array(first_row_index, last_row_index, column_key, expected_type) # type: ignore
                return result
            else:
                result: ARRAY_TYPE = self._cell_get_array(first_row_index, last_row_index, column_key) # type: ignore
                return result

    ########################################################
    # Cell Operations: Set
    ########################################################

    def _cell_set_scalar(self, row_index: int, column_key: CK, value: SCALAR_TYPE) -> None:
        """
        Internal: Set the value of a specific cell. (no lock, no read-only check)

        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (SCALAR_TYPE): The new value
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """

        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        column_index: int = self._column_keys.index(column_key)        
        column_unit: Optional[Unit] = self._column_units[column_key]
        self._internal_dataframe.iloc[row_index, column_index] = self._column_types[column_key].get_value_for_dataframe(value, column_unit)

    def cell_set_scalar(self, row_index: int, column_key: CK, value: SCALAR_TYPE) -> None:
        """
        Set the value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (Any): The new value
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._cell_set_scalar(row_index, column_key, value)

    def _cell_set_value(self, row_index: int, column_key: CK, value: VALUE_TYPE, unit: Unit|None=None) -> None:
        """
        Internal: Set the value of a specific cell. (no lock, no read-only check)

        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (VALUE_TYPE): The new value
            unit (Unit|None): The unit of the value
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """

        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")

        column_index: int = self._column_keys.index(column_key)
        column_unit: Optional[Unit] = self._column_units[column_key]
        column_type: ColumnType = self._column_types[column_key]
        if not column_type.check_value_type(value):
            raise ValueError(f"Value {value} is not a {column_type.name}.")
        if column_type.has_unit:
            if unit is None:
                self._internal_dataframe.iloc[row_index, column_index] = value
            else:
                self._internal_dataframe.iloc[row_index, column_index] = Unit.convert(value, unit, column_unit) # type: ignore
        else:
            self._internal_dataframe.iloc[row_index, column_index] = value

    def cell_set_value(self, row_index: int, column_key: CK, value: VALUE_TYPE, unit: Unit|None=None) -> None:
        """
        Set the value of a specific cell.

        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (VALUE_TYPE): The new value
            unit (Unit|None): The unit of the value
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._cell_set_value(row_index, column_key, value, unit)

    def cell_set_scalars(self, row_indices: Sequence[int], column_key: CK, scalars: Sequence[SCALAR_TYPE]) -> None:
        """
        Set multiple cell values in a single column.
        
        Args:
            row_indices (Sequence[int]): List of row indices
            column_key (CK): The column key
            scalars (Sequence[SCALAR_TYPE]): List of new values
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if len(row_indices) != len(scalars):
                raise ValueError(f"Length of row_indices ({len(row_indices)}) does not match length of values ({len(scalars)}).")
            
            # Check all row indices are valid
            for row_index in row_indices:
                if row_index < 0 or row_index >= len(self._internal_dataframe):
                    raise ValueError(f"Row index {row_index} is out of bounds.")
            
            for row_index, value in zip(row_indices, scalars):
                self._cell_set_scalar(row_index, column_key, value)

    def _cell_set_array(self, first_row_index: int, column_key: CK, array: ARRAY_TYPE) -> None:
        """
        Internal: Set an array of values at a specific row index. (no lock, no read-only check)

        Args:
            first_row_index (int): The first row index
            column_key (CK): The column key
            array (ARRAY_TYPE): The array to set
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        if first_row_index < 0 or first_row_index >= len(self._internal_dataframe) + len(array):
            raise ValueError(f"Row index {first_row_index} is out of bounds or the array is too large.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._internal_dataframe_column_names[column_key]
        internal_dataframe_column_type = self._column_types[column_key]
        internal_dataframe_column_unit = self._column_units[column_key]

        self._internal_dataframe.loc[first_row_index:first_row_index + len(array), internal_column_name] = internal_dataframe_column_type.get_pd_series_for_dataframe(array, internal_dataframe_column_unit) # type: ignore[reportUnknownReturnType]

    def cell_set_array(self, first_row_index: int, column_key: CK, array: ARRAY_TYPE) -> None:
        """
        Set an array of values at a specific row index.

        Args:
            first_row_index (int): The first row index
            column_key (CK): The column key
            array (ARRAY_TYPE): The array to set
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._cell_set_array(first_row_index, column_key, array)

    def _cell_set_numpy_array_or_series(self, first_row_index: int, column_key: CK, numpy_array_or_series: np.ndarray|pd.Series[Any], array_unit: Unit|None=None) -> None: # type: ignore
        """
        Internal: Set a numpy array or series at a specific row index. (no lock, no read-only check)
        """
        if first_row_index < 0 or first_row_index >= len(self._internal_dataframe) + len(numpy_array_or_series):
            raise ValueError(f"Row index {first_row_index} is out of bounds or the array is too large.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._internal_dataframe_column_names[column_key]
        internal_dataframe_column_type = self._column_types[column_key]
        internal_dataframe_column_unit = self._column_units[column_key]

        self._internal_dataframe.loc[first_row_index:first_row_index + len(numpy_array_or_series), internal_column_name] = internal_dataframe_column_type.get_pd_series_for_dataframe(numpy_array_or_series, internal_dataframe_column_unit, array_unit) # type: ignore[reportUnknownReturnType]
        
    def cell_set_numpy_array_or_series(self, first_row_index: int, column_key: CK, numpy_array_or_series: np.ndarray|pd.Series[Any], array_unit: Unit|None=None) -> None:
        """
        Set a numpy array or series at a specific row index.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._cell_set_numpy_array_or_series(first_row_index, column_key, numpy_array_or_series, array_unit)

    ########################################################
    # Cell Operations: Other
    ########################################################

    def cell_is_missing(self, row_index: int, column_key: CK) -> bool:
        """
        Check if a cell value is missing (NaN).
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Returns:
            bool: True if the cell value is missing, False otherwise
        """
        with self._rlock:
            if row_index < 0 or row_index >= len(self._internal_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Check if cell exists:
            if self._internal_dataframe.iloc[row_index][self._internal_dataframe_column_names[column_key]] is pd.NA:
                return True
            else:
                return False

    def _cell_set_missing(self, row_index: int, column_key: CK) -> None:
        """
        Internal: Set a cell value to missing (NaN). (no lock, no read-only check)
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Raises:
            ValueError: If the column type cannot be set to missing
        """
        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise IndexError(f"Row index {row_index} is out of bounds")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name: str = self._internal_dataframe_column_names[column_key]
        if self._column_types[column_key].can_be_none:
            self._internal_dataframe.loc[row_index, internal_column_name] = self._column_types[column_key].missing_value
        else:
            raise ValueError(f"Column '{column_key}' of type {self._column_types[column_key]} cannot be set to missing.")
            
    def cell_set_missing(self, row_index: int, column_key: CK) -> None:
        """
        Set a cell value to missing (NaN). Some column types cannot be set to missing.

        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._cell_set_missing(row_index, column_key)