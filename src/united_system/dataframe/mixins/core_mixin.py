"""
Core functionality mixin for UnitedDataframe.

Contains basic properties, initialization helpers, and core utility methods.
"""

from typing import TypeVar, Literal
import pandas as pd
from pandas._typing import Dtype
import numpy as np

from ...utils import JSONable, HDF5able
from .dataframe_protocol import UnitedDataframeMixin, UnitedDataframeProtocol, CK

class CoreMixin(UnitedDataframeMixin[CK], JSONable, HDF5able):
    """
    Core functionality mixin for UnitedDataframe.
    
    Provides basic properties, initialization helpers, and core utility methods
    that are used throughout the dataframe implementation.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """
    
    # Basic properties and information
    def __len__(self) -> int:
        """
        Return the number of rows in the dataframe.
        
        Returns:
            int: The number of rows in the dataframe
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe)
    
    def is_empty(self) -> bool:
        """
        Check if the dataframe is empty.
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe) == 0
        
    def is_numeric(self, column_key: CK) -> bool:
        """
        Check if a column is numeric.
        """
        with self._rlock:
            return self.column_type(column_key).is_numeric()
        
    def has_unit(self, column_key: CK) -> bool:
        """
        Check if a column has a unit.
        """
        with self._rlock:
            return self.column_type(column_key).has_unit

    @property
    def cols(self) -> int:
        """
        Return the number of columns in the dataframe.
        """
        with self._rlock:
            return len(self._column_keys)
        
    @property
    def rows(self) -> int:
        """
        Return the number of rows in the dataframe.
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the dataframe as (rows, columns).
        
        Returns:
            tuple[int, int]: A tuple containing (number_of_rows, number_of_columns)
        """
        with self._rlock:
            return self._internal_canonical_dataframe.shape

    @property
    def size(self) -> int:
        """
        Get the total number of elements in the dataframe.
        
        Returns:
            int: Total number of elements (rows × columns)
        """
        with self._rlock:
            return self._internal_canonical_dataframe.size

    @property
    def empty(self) -> bool:
        """
        Check if the dataframe is empty.
        
        Returns:
            bool: True if the dataframe has no rows, False otherwise
        """
        with self._rlock:
            return self._internal_canonical_dataframe.empty

    # Internal utilities
    def create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Create the internal dataframe column name for a column.

        Args:
            column_key (CK): The column key to create the internal dataframe column name for

        Returns:
            str: The internal dataframe column name for the column
        """
        column_information = self._column_information[column_key]
        return self._internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_information)

    @staticmethod
    def column_key_to_string(column_key: CK) -> str:
        """Convert a column key to string."""
        if hasattr(column_key, 'to_string'):
            return column_key.to_string()
        else:
            return str(column_key)

    def compatible_with_column(self, column_key: CK, value) -> bool:
        """
        Check if a value is compatible with a column.
        
        Args:
            column_key (CK): The column key to check compatibility with
            value: The value to check
            
        Returns:
            bool: True if the value is compatible with the column type
        """
        column_type = self._column_types[column_key]
        try:
            # Try to convert the value using the column type
            column_type.cast_for_dataframe(value)
            return True
        except (TypeError, ValueError):
            return False

    @property
    def internal_dataframe_deepcopy(self) -> pd.DataFrame:
        """
        Get a deep copy of the internal pandas DataFrame.
        
        Returns:
            pd.DataFrame: A deep copy of the underlying pandas DataFrame
        """
        with self._rlock:
            return self._internal_canonical_dataframe.copy(deep=True)

    def copy(self, deep: bool = True) -> UnitedDataframeProtocol[CK]:
        """
        Create a copy of the UnitedDataframe.
        
        Returns:
            UnitedDataframeProtocol[CK]: A new instance with copied data and metadata
        """
        with self._rlock:
            # Import here to avoid circular imports
            from ...united_dataframe import UnitedDataframe
            new_df = UnitedDataframe(
                self._internal_canonical_dataframe.copy(deep=deep),
                self._column_information,
                self._internal_dataframe_column_name_formatter)
            return new_df
    
    # Read-only state management
    def is_read_only(self) -> bool:
        """
        Check if the dataframe is in read-only mode.
        
        Returns:
            bool: True if the dataframe is read-only, False otherwise
        """
        with self._rlock:
            return self._read_only

    def set_read_only(self, read_only: bool) -> None:
        """
        Set the read-only status of the dataframe.
        
        Args:
            read_only (bool): True to make the dataframe read-only, False to allow modifications
        """
        with self._wlock:
            self._read_only = read_only

    # Utility methods
    def get_numeric_column_keys(self) -> list[CK]:
        """
        Get a list of column keys for numeric columns only.
        
        Returns:
            list[CK]: List of column keys for numeric columns
        """
        with self._rlock:
            return [column_key for column_key in self._column_keys if self.is_numeric(column_key)] 