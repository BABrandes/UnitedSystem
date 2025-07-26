"""
Cell operations mixin for UnitedDataframe.

Contains all operations related to individual cell operations,
including getting and setting individual cell values.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, Any, TypeVar, overload
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ....column_type import SCALAR_TYPE, LOWLEVEL_TYPE
import pandas as pd

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

ST = TypeVar("ST", bound=SCALAR_TYPE)
LT = TypeVar("LT", bound=LOWLEVEL_TYPE)

class CellOperationsMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Cell operations mixin for UnitedDataframe.
    
    Provides all functionality related to individual cell operations,
    including getting and setting individual cell values.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Cell Operations: Get/Set ------------

    @overload
    def cell_get_value(self, row_index: int, column_key: CK) -> SCALAR_TYPE: ...
    @overload
    def cell_get_value(self, row_index: int, column_key: CK, expected_type: type[ST]) -> ST: ...
    def cell_get_value(self, row_index: int, column_key: CK, expected_type: type[ST]|None = None) -> ST|SCALAR_TYPE:
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
                result: ST = self._cell_get_value(row_index, column_key) # type: ignore
                return result
            else:
                result: SCALAR_TYPE = self._cell_get_value(row_index, column_key) # type: ignore
                return result

    def _cell_get_value(self, row_index: int, column_key: CK) -> SCALAR_TYPE:
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
        
        raw_value: LOWLEVEL_TYPE = self._internal_dataframe.loc[row_index, self._internal_dataframe_column_names[column_key]] # type: ignore
        result: SCALAR_TYPE = self._column_types[column_key].get_scalar_value_from_dataframe(raw_value, self._column_units[column_key])
        return result
    
    @overload
    def cell_get_lowlevel_value(self, row_index: int, column_key: CK) -> LOWLEVEL_TYPE: ...
    @overload
    def cell_get_lowlevel_value(self, row_index: int, column_key: CK, expected_type: type[LT]) -> LT: ...
    def cell_get_lowlevel_value(self, row_index: int, column_key: CK, expected_type: type[LT]|None = None) -> LT|LOWLEVEL_TYPE:
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
                result: LT = self._cell_get_lowlevel_value(row_index, column_key) # type: ignore
                return result
            else:
                result: LOWLEVEL_TYPE = self._cell_get_lowlevel_value(row_index, column_key) # type: ignore
                return result

    def _cell_get_lowlevel_value(self, row_index: int, column_key: CK) -> LOWLEVEL_TYPE:
        """
        Internal: Get the lowlevel value of a specific cell. (no lock)
        """
        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        raw_value: LOWLEVEL_TYPE = self._internal_dataframe.loc[row_index, self._internal_dataframe_column_names[column_key]] # type: ignore
        return raw_value

    def _cell_set_value(self, row_index: int, column_key: CK, value: Any) -> None:
        """
        Internal: Set the value of a specific cell. (no lock, no read-only check)

        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (Any): The new value
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """

        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds.")
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._internal_dataframe_column_names[column_key]
        column_unit = self._column_units[column_key]
        self._internal_dataframe.loc[row_index, internal_column_name] = self._column_types[column_key].get_value_for_dataframe(value, column_unit)
        

    def cell_set_value(self, row_index: int, column_key: CK, value: Any) -> None:
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
            self._cell_set_value(row_index, column_key, value)

    # ----------- Cell Operations: Batch ------------

    @overload
    def cells_get_values(self, row_indices: list[int], column_key: CK) -> list[SCALAR_TYPE]: ...
    @overload
    def cells_get_values(self, row_indices: list[int], column_key: CK, expected_type: type[ST]) -> list[ST]: ...
    def cells_get_values(self, row_indices: list[int], column_key: CK, expected_type: type[ST]|None = None) -> list[ST]|list[SCALAR_TYPE]:
        """
        Get multiple cell values from a single column.
        """

        with self._rlock:
            if expected_type is not None:
                if not self._column_types[column_key].check_scalar_type(expected_type):
                    raise ValueError(f"Column {column_key} is not a {expected_type} column.")
                result: list[ST] = self._cells_get_values(row_indices, column_key) # type: ignore
                return result
            else:
                result: list[SCALAR_TYPE] = self._cells_get_values(row_indices, column_key) # type: ignore
                return result

    def _cells_get_values(self, row_indices: list[int], column_key: CK, expected_type: type[ST]|None = None) -> list[ST]|list[SCALAR_TYPE]:
        """
        Internal: Get multiple cell values from a single column. (no lock)
        
        Args:
            row_indices (list[int]): List of row indices
            column_key (CK): The column key
            
        Returns:
            list[Any]: List of cell values
        """

        if expected_type is not None:
            if not self._column_types[column_key].check_scalar_type(expected_type):
                raise ValueError(f"Column {column_key} is not a {expected_type} column.")
            result: list[ST] = self._cells_get_values(row_indices, column_key) # type: ignore
            return result
        else:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Check all row indices are valid
            for row_index in row_indices:
                if row_index < 0 or row_index >= len(self._internal_dataframe):
                    raise ValueError(f"Row index {row_index} is out of bounds.")
                
            list_of_values: list[Any] = []
            for row_index in row_indices:
                list_of_values.append(self._cell_get_value(row_index, column_key))
            return list_of_values

    def cells_set_values(self, row_indices: list[int], column_key: CK, values: list[Any]) -> None:
        """
        Set multiple cell values in a single column.
        
        Args:
            row_indices (list[int]): List of row indices
            column_key (CK): The column key
            values (list[Any]): List of new values
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if len(row_indices) != len(values):
                raise ValueError(f"Length of row_indices ({len(row_indices)}) does not match length of values ({len(values)}).")
            
            # Check all row indices are valid
            for row_index in row_indices:
                if row_index < 0 or row_index >= len(self._internal_dataframe):
                    raise ValueError(f"Row index {row_index} is out of bounds.")
            
            for row_index, value in zip(row_indices, values):
                self._cell_set_value(row_index, column_key, value)

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

    def cell_set_missing(self, row_index: int, column_key: CK) -> None:
        """
        Set a cell value to missing (NaN).
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Raises:
            ValueError: If the dataframe is read-only
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row_index < 0 or row_index >= len(self._internal_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            self._cell_set_value(row_index, column_key, pd.NA)