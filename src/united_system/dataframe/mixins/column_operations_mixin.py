"""
Column operations mixin for UnitedDataframe.

Contains all operations related to column CRUD (Create, Read, Update, Delete),
including adding, removing, and setting column values.
"""

from typing import Generic, TypeVar, Literal
import pandas as pd
import numpy as np

from ..column_type import ColumnType, SCALAR_TYPE, ARRAY_TYPE
from ...unit import Unit
from ...dimension import Dimension

CK = TypeVar("CK", bound=str, default=str)

class ColumnOperationsMixin(Generic[CK]):
    """
    Column operations mixin for UnitedDataframe.
    
    Provides all functionality related to column CRUD operations,
    including adding, removing, and setting column values.
    """

    # ----------- Retrievals: Column values ------------

    def column(self, column_key: CK) -> ARRAY_TYPE:
        """
        Get the column as a UnitedArray.
        
        Args:
            column_key (CK): The column key to get the column of
            
        Returns:
            ARRAY_TYPE: The column as a UnitedArray
        """
        with self._rlock:
            internal_column_string: str = self._internal_dataframe_column_strings[column_key]
            column_type: ColumnType = self._column_types[column_key]
            display_unit: Unit = self._display_units[column_key]
            dimension: Dimension = self._dimensions[column_key]
            internal_column_values: pd.Series = self._internal_canonical_dataframe[internal_column_string]
            return column_type.pandas_series_to_united_array(internal_column_values, display_unit, dimension)

    # ----------- Setters: Column values ------------

    def set_column(self, column_key: CK, column_values: ARRAY_TYPE) -> None:
        """
        Set the column values.
        
        Args:
            column_key (CK): The column key to set the column values of
            column_values (ARRAY_TYPE): The column values to set
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist,
                       or the column values are not compatible with the column type
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not self.compatible_with_column(column_key, column_values):
                raise ValueError(f"Column values are not compatible with column key {column_key}.")
            internal_column_string: str = self._internal_dataframe_column_strings[column_key]
            column_type: ColumnType = self._column_types[column_key]
            self._internal_canonical_dataframe[internal_column_string] = column_type.united_array_to_pandas_series(column_values)

    def add_column(self, column_key: CK, column_values: ARRAY_TYPE|None = None, column_type: ColumnType|None = None, display_unit: Unit|None = None, dimension: Dimension|None = None) -> None:
        """
        Add a new column to the dataframe.
        
        Args:
            column_key (CK): The column key to add
            column_values (ARRAY_TYPE|None): The column values to add. If None, the column will be filled with NaN
            column_type (ColumnType|None): The column type. If None, inferred from column_values
            display_unit (Unit|None): The display unit. If None, inferred from column_values
            dimension (Dimension|None): The dimension. If None, inferred from column_values
            
        Raises:
            ValueError: If the dataframe is read-only, the column already exists,
                       or the column values are not compatible with the column type
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key in self._column_keys:
                raise ValueError(f"Column key {column_key} already exists in the dataframe.")
            
            # Infer column properties from column_values if not provided
            if column_values is not None:
                if column_type is None:
                    column_type = ColumnType.infer_from_value(column_values)
                if display_unit is None:
                    display_unit = getattr(column_values, 'unit', None)
                if dimension is None:
                    dimension = getattr(column_values, 'dimension', None)
            else:
                # Default to float column if no values provided
                if column_type is None:
                    column_type = ColumnType.FLOAT
                if display_unit is None:
                    display_unit = None
                if dimension is None:
                    dimension = None
            
            # Create internal column name
            internal_column_string: str = self.create_internal_dataframe_column_name(column_key)
            
            # Add to internal structures
            self._column_keys.append(column_key)
            self._column_types[column_key] = column_type
            self._display_units[column_key] = display_unit
            self._dimensions[column_key] = dimension
            self._internal_dataframe_column_strings[column_key] = internal_column_string
            
            # Add to internal dataframe
            if column_values is not None:
                self._internal_canonical_dataframe[internal_column_string] = column_type.united_array_to_pandas_series(column_values)
            else:
                # Create empty column with NaN values
                self._internal_canonical_dataframe[internal_column_string] = pd.Series([np.nan] * len(self._internal_canonical_dataframe), dtype=column_type.value.corresponding_pandas_type)

    def remove_column(self, column_key: CK) -> None:
        """
        Remove a column from the dataframe.
        
        Args:
            column_key (CK): The column key to remove
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Remove from internal structures
            internal_column_string: str = self._internal_dataframe_column_strings[column_key]
            self._column_keys.remove(column_key)
            del self._column_types[column_key]
            del self._display_units[column_key]
            del self._dimensions[column_key]
            del self._internal_dataframe_column_strings[column_key]
            
            # Remove from internal dataframe
            self._internal_canonical_dataframe.drop(columns=[internal_column_string], inplace=True)

    def column_clamp_values(self, column_key: CK, min_val: SCALAR_TYPE, max_val: SCALAR_TYPE, inclusive: Literal["both", "neither", "left", "right"] = "both") -> None:
        """
        Clamp all values in a column to the specified range.
        
        Args:
            column_key (CK): The column key to clamp
            min_val (SCALAR_TYPE): The minimum value
            max_val (SCALAR_TYPE): The maximum value
            inclusive (str): Which bounds to include ("both", "neither", "left", "right")
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist,
                       or the column is not numeric
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            # Get the column values
            column_values = self.column(column_key)
            
            # Clamp the values
            clamped_values = column_values.clip(min_val, max_val, inclusive=inclusive)
            
            # Set the clamped values back
            self.set_column(column_key, clamped_values) 