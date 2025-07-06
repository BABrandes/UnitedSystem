"""
Row access mixin for UnitedDataframe.

Contains all operations related to accessing rows and row ranges,
including head, tail, first, last, and row iteration patterns.
"""

from typing import Generic, TypeVar, Dict, Iterator, Union, Tuple
import pandas as pd

from ..column_type import SCALAR_TYPE
from ..column_key import ColumnKey

CK = TypeVar("CK", bound=Union[ColumnKey, str])

class RowAccessMixin(Generic[CK]):
    """
    Row access mixin for UnitedDataframe.
    
    Provides all functionality related to accessing rows and row ranges,
    including head, tail, first, last, and row iteration patterns.
    """

    def rows_get_head(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the first n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 1)
            
        Returns:
            UnitedDataframe: A new dataframe containing the first n rows
            
        Raises:
            ValueError: If n is negative
            
        Examples:
            # Get first 1 rows (default)
            df.rows_get_head()
            
            # Get first 10 rows
            df.rows_get_head(10)
            
            # Get all rows if n is larger than dataframe size
            df.rows_get_head(100)  # Returns all rows if dataframe has fewer than 100 rows
        """
        with self._rlock:
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            
            # If n is larger than the dataframe size, return all rows
            actual_n = min(n, len(self._internal_canonical_dataframe))
            head_df = self._internal_canonical_dataframe.head(actual_n)
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                head_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def rows_get_tail(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the last n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 1)
            
        Returns:
            UnitedDataframe: A new dataframe containing the last n rows
            
        Raises:
            ValueError: If n is negative
            
        Examples:
            # Get last 1 rows (default)
            df.rows_get_tail()
            
            # Get last 10 rows
            df.rows_get_tail(10)
            
            # Get all rows if n is larger than dataframe size
            df.rows_get_tail(100)  # Returns all rows if dataframe has fewer than 100 rows
        """
        with self._rlock:
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            
            # If n is larger than the dataframe size, return all rows
            actual_n = min(n, len(self._internal_canonical_dataframe))
            tail_df = self._internal_canonical_dataframe.tail(actual_n)
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                tail_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def rows_get_first(self) -> "UnitedDataframe[CK]":
        """
        Get the first row of the dataframe.
        
        Returns:
            UnitedDataframe: A new dataframe containing only the first row
            
        Raises:
            ValueError: If the dataframe is empty
            
        Examples:
            # Get the first row
            first_row = df.rows_get_first()
            
            # Access the first row's values
            first_row[0, 'column_name']
        """
        with self._rlock:
            if self.empty:
                raise ValueError("Cannot get first row from an empty dataframe")
            
            first_df = self._internal_canonical_dataframe.head(1)
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                first_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def rows_get_last(self) -> "UnitedDataframe[CK]":
        """
        Get the last row of the dataframe.
        
        Returns:
            UnitedDataframe: A new dataframe containing only the last row
            
        Raises:
            ValueError: If the dataframe is empty
            
        Examples:
            # Get the last row
            last_row = df.rows_get_last()
            
            # Access the last row's values
            last_row[0, 'column_name']
        """
        with self._rlock:
            if self.empty:
                raise ValueError("Cannot get last row from an empty dataframe")
            
            last_df = self._internal_canonical_dataframe.tail(1)
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                last_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def row_get_values(self, row_index: int) -> Dict[CK, SCALAR_TYPE]:
        """
        Get a row from the dataframe as a dictionary.

        Args:
            row_index (int): The index of the row to get

        Returns:
            Dict[CK, SCALAR_TYPE]: A dictionary of column keys and values for the row
        """
        with self._rlock:
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            
            return {column_key: self.cell_value_get(row_index, column_key) for column_key in self._column_keys}

    def iterrows(self) -> Iterator[Tuple[int, Dict[CK, SCALAR_TYPE]]]:
        """
        Iterate over dataframe rows as (row_index, row_dict) pairs.
        
        Returns:
            Iterator[Tuple[int, Dict[CK, SCALAR_TYPE]]]: Iterator yielding (index, row_dict) pairs
            
        Examples:
            # Iterate over all rows
            for row_index, row_values in df.iterrows():
                print(f"Row {row_index}: {row_values}")
        """
        with self._rlock:
            for row_index in range(len(self._internal_canonical_dataframe)):
                yield (row_index, self.row_get_values(row_index))

    def get_iterator_for_row(self, row_index: int) -> Iterator[SCALAR_TYPE]:
        """
        Get an iterator over the values of a specific row.

        Args:
            row_index (int): The index of the row to get the iterator for

        Returns:
            Iterator[SCALAR_TYPE]: An iterator over the values of the row
            
        Examples:
            # Iterate over values in row 0
            for value in df.get_iterator_for_row(0):
                print(value)
        """
        with self._rlock:
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise IndexError(f"Row index {row_index} is out of bounds")
            
            return (self.cell_value_get(row_index, column_key) for column_key in self._column_keys) 