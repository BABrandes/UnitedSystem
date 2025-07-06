"""
Cell operations mixin for UnitedDataframe.

Contains all operations related to individual cell access and manipulation,
including getting and setting cell values by row and column.
"""

from typing import Generic, TypeVar

from ..column_type import SCALAR_TYPE

CK = TypeVar("CK", bound=str, default=str)

class CellOperationsMixin(Generic[CK]):
    """
    Cell operations mixin for UnitedDataframe.
    
    Provides all functionality related to individual cell operations,
    including getting and setting cell values by row and column.
    """

    # ----------- Cell operations ------------

    def cell_get_value(self, row_index: int, column_key: CK) -> SCALAR_TYPE:
        """
        Get a single cell value.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Returns:
            SCALAR_TYPE: The cell value as a UnitedScalar
        """
        with self._rlock:
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_string = self._internal_dataframe_column_strings[column_key]
            raw_value = self._internal_canonical_dataframe.iloc[row_index][internal_column_string]
            column_type = self._column_types[column_key]
            display_unit = self._display_units[column_key]
            dimension = self._dimensions[column_key]
            
            # Convert raw value to appropriate scalar type
            return column_type.raw_value_to_united_scalar(raw_value, display_unit, dimension)

    def cell_set_value(self, row_index: int, column_key: CK, value: SCALAR_TYPE) -> None:
        """
        Set a single cell value.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (SCALAR_TYPE): The value to set
            
        Raises:
            ValueError: If the dataframe is read-only or the value is incompatible
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not self.compatible_with_column(column_key, value):
                raise ValueError(f"Value {value} is not compatible with column {column_key}.")
            
            internal_column_string = self._internal_dataframe_column_strings[column_key]
            column_type = self._column_types[column_key]
            raw_value = column_type.united_scalar_to_raw_value(value)
            
            # Set the value
            self._internal_canonical_dataframe.iloc[row_index, self._internal_canonical_dataframe.columns.get_loc(internal_column_string)] = raw_value

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