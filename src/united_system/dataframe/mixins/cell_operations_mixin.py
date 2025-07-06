"""
Cell operations mixin for UnitedDataframe.

Contains all operations related to individual cell operations,
including getting and setting individual cell values.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Any
from .dataframe_protocol import UnitedDataframeMixin, CK

class CellOperationsMixin(UnitedDataframeMixin[CK]):
    """
    Cell operations mixin for UnitedDataframe.
    
    Provides all functionality related to individual cell operations,
    including getting and setting individual cell values.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Cell Operations: Get/Set ------------

    def cell_get_value(self, row_index: int, column_key: CK) -> Any:
        """
        Get the value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Returns:
            Any: The cell value
        """
        with self._rlock:  # Full IDE support!
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise ValueError(f"Row index {row_index} is out of bounds.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            return self._internal_canonical_dataframe.loc[row_index, internal_column_name]

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
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise ValueError(f"Row index {row_index} is out of bounds.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            self._internal_canonical_dataframe.loc[row_index, internal_column_name] = value

    # ----------- Cell Operations: Batch ------------

    def cells_get_values(self, row_indices: list[int], column_key: CK) -> list[Any]:
        """
        Get multiple cell values from a single column.
        
        Args:
            row_indices (list[int]): List of row indices
            column_key (CK): The column key
            
        Returns:
            list[Any]: List of cell values
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Check all row indices are valid
            for row_index in row_indices:
                if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                    raise ValueError(f"Row index {row_index} is out of bounds.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            return [self._internal_canonical_dataframe.loc[row_index, internal_column_name] for row_index in row_indices]

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
                if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                    raise ValueError(f"Row index {row_index} is out of bounds.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            for row_index, value in zip(row_indices, values):
                self._internal_canonical_dataframe.loc[row_index, internal_column_name] = value

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
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_string = self._internal_dataframe_column_strings[column_key]
            raw_value = self._internal_canonical_dataframe.iloc[row_index][internal_column_string]
            
            # Check if value is NaN
            import pandas as pd
            return pd.isna(raw_value)

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
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_string = self._internal_dataframe_column_strings[column_key]
            
            # Set the value to NaN
            import numpy as np
            self._internal_canonical_dataframe.iloc[row_index, self._internal_canonical_dataframe.columns.get_loc(internal_column_string)] = np.nan 