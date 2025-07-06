"""
Row operations mixin for UnitedDataframe.

Contains all operations related to row CRUD (Create, Read, Update, Delete),
including getting, setting, adding, and removing rows.
"""

from typing import Generic, TypeVar, Dict, Any
import pandas as pd
import numpy as np

from ..column_type import SCALAR_TYPE

CK = TypeVar("CK", bound=str, default=str)

class RowOperationsMixin(Generic[CK]):
    """
    Row operations mixin for UnitedDataframe.
    
    Provides all functionality related to row CRUD operations,
    including getting, setting, adding, and removing rows.
    """

    # ----------- Row operations ------------



    def row_set_values(self, row_index: int, values: Dict[CK, SCALAR_TYPE]) -> None:
        """
        Set row values from a dictionary.
        
        Args:
            row_index (int): The row index
            values (Dict[CK, SCALAR_TYPE]): Dictionary mapping column keys to scalar values
            
        Raises:
            ValueError: If the dataframe is read-only or values are incompatible
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            
            # Validate all values before setting any
            for column_key, value in values.items():
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
                if not self.compatible_with_column(column_key, value):
                    raise ValueError(f"Value {value} is not compatible with column {column_key}.")
            
            # Set all values
            for column_key, value in values.items():
                internal_column_string = self._internal_dataframe_column_strings[column_key]
                column_type = self._column_types[column_key]
                raw_value = column_type.united_scalar_to_raw_value(value)
                self._internal_canonical_dataframe.iloc[row_index, self._internal_canonical_dataframe.columns.get_loc(internal_column_string)] = raw_value

    def row_add_values(self, values: Dict[CK, SCALAR_TYPE]) -> None:
        """
        Add a new row with the specified values.
        
        Args:
            values (Dict[CK, SCALAR_TYPE]): Dictionary mapping column keys to scalar values
            
        Raises:
            ValueError: If the dataframe is read-only or values are incompatible
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Validate all values before adding
            for column_key, value in values.items():
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
                if not self.compatible_with_column(column_key, value):
                    raise ValueError(f"Value {value} is not compatible with column {column_key}.")
            
            # Prepare row data
            row_data = {}
            for column_key in self._column_keys:
                internal_column_string = self._internal_dataframe_column_strings[column_key]
                if column_key in values:
                    column_type = self._column_types[column_key]
                    raw_value = column_type.united_scalar_to_raw_value(values[column_key])
                    row_data[internal_column_string] = raw_value
                else:
                    # Use NaN for missing values
                    row_data[internal_column_string] = np.nan
            
            # Add the row
            new_row = pd.DataFrame([row_data])
            self._internal_canonical_dataframe = pd.concat([self._internal_canonical_dataframe, new_row], ignore_index=True)

    def row_remove_by_index(self, row_index: int) -> None:
        """
        Remove a row by index.
        
        Args:
            row_index (int): The row index to remove
            
        Raises:
            ValueError: If the dataframe is read-only
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            
            self._internal_canonical_dataframe.drop(index=row_index, inplace=True)
            self._internal_canonical_dataframe.reset_index(drop=True, inplace=True)

    def rows_remove_by_indices(self, row_indices: list[int]) -> None:
        """
        Remove multiple rows by indices.
        
        Args:
            row_indices (list[int]): List of row indices to remove
            
        Raises:
            ValueError: If the dataframe is read-only
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Validate all indices
            for row_index in row_indices:
                if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                    raise IndexError(f"Row index {row_index} is out of bounds")
            
            # Remove rows
            self._internal_canonical_dataframe.drop(index=row_indices, inplace=True)
            self._internal_canonical_dataframe.reset_index(drop=True, inplace=True)

    def rows_add_from_dict_list(self, rows_data: list[Dict[CK, SCALAR_TYPE]]) -> None:
        """
        Add multiple rows from a list of dictionaries.
        
        Args:
            rows_data (list[Dict[CK, SCALAR_TYPE]]): List of dictionaries, each mapping column keys to scalar values
            
        Raises:
            ValueError: If the dataframe is read-only or values are incompatible
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Validate all values before adding any rows
            for row_values in rows_data:
                for column_key, value in row_values.items():
                    if column_key not in self._column_keys:
                        raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
                    if not self.compatible_with_column(column_key, value):
                        raise ValueError(f"Value {value} is not compatible with column {column_key}.")
            
            # Prepare all row data
            rows_to_add = []
            for row_values in rows_data:
                row_data = {}
                for column_key in self._column_keys:
                    internal_column_string = self._internal_dataframe_column_strings[column_key]
                    if column_key in row_values:
                        column_type = self._column_types[column_key]
                        raw_value = column_type.united_scalar_to_raw_value(row_values[column_key])
                        row_data[internal_column_string] = raw_value
                    else:
                        # Use NaN for missing values
                        row_data[internal_column_string] = np.nan
                rows_to_add.append(row_data)
            
            # Add all rows
            new_rows = pd.DataFrame(rows_to_add)
            self._internal_canonical_dataframe = pd.concat([self._internal_canonical_dataframe, new_rows], ignore_index=True)

    def rows_clear_all(self) -> None:
        """
        Remove all rows from the dataframe, keeping column structure.
        
        Raises:
            ValueError: If the dataframe is read-only
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Create empty dataframe with same columns and dtypes
            empty_data = {col: pd.Series([], dtype=self._internal_canonical_dataframe[col].dtype) 
                         for col in self._internal_canonical_dataframe.columns}
            self._internal_canonical_dataframe = pd.DataFrame(empty_data) 