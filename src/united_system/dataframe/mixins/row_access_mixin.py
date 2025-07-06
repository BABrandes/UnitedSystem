"""
Row access mixin for UnitedDataframe.

Contains all operations related to row access patterns, including
head, tail, first, last, and row iteration.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Any, Dict, Iterator, List
from .dataframe_protocol import UnitedDataframeMixin, CK

class RowAccessMixin(UnitedDataframeMixin[CK]):
    """
    Row access mixin for UnitedDataframe.
    
    Provides all functionality related to row access patterns, including
    head, tail, first, last, and row iteration.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Row Access: Head/Tail ------------

    def rows_get_head(self, n: int = 5) -> List[Dict[CK, Any]]:
        """
        Get the first n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 5)
            
        Returns:
            List[Dict[CK, Any]]: List of row dictionaries
        """
        with self._rlock:  # Full IDE support!
            if n < 0:
                raise ValueError("n must be non-negative.")
            
            n = min(n, len(self._internal_canonical_dataframe))
            rows = []
            
            for row_index in range(n):
                row_dict = {}
                for column_key in self._column_keys:
                    internal_column_name = self._internal_dataframe_column_strings[column_key]
                    row_dict[column_key] = self._internal_canonical_dataframe.loc[row_index, internal_column_name]
                rows.append(row_dict)
            
            return rows

    def rows_get_tail(self, n: int = 5) -> List[Dict[CK, Any]]:
        """
        Get the last n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 5)
            
        Returns:
            List[Dict[CK, Any]]: List of row dictionaries
        """
        with self._rlock:
            if n < 0:
                raise ValueError("n must be non-negative.")
            
            total_rows = len(self._internal_canonical_dataframe)
            n = min(n, total_rows)
            rows = []
            
            for row_index in range(total_rows - n, total_rows):
                row_dict = {}
                for column_key in self._column_keys:
                    internal_column_name = self._internal_dataframe_column_strings[column_key]
                    row_dict[column_key] = self._internal_canonical_dataframe.loc[row_index, internal_column_name]
                rows.append(row_dict)
            
            return rows

    def rows_get_first(self) -> Dict[CK, Any]:
        """
        Get the first row of the dataframe.
        
        Returns:
            Dict[CK, Any]: Dictionary representing the first row
        """
        with self._rlock:
            if len(self._internal_canonical_dataframe) == 0:
                raise ValueError("Dataframe is empty.")
            
            return self.rows_get_head(1)[0]

    def rows_get_last(self) -> Dict[CK, Any]:
        """
        Get the last row of the dataframe.
        
        Returns:
            Dict[CK, Any]: Dictionary representing the last row
        """
        with self._rlock:
            if len(self._internal_canonical_dataframe) == 0:
                raise ValueError("Dataframe is empty.")
            
            return self.rows_get_tail(1)[0]

    # ----------- Row Access: Single Row ------------

    def row_get_values(self, row_index: int) -> Dict[CK, Any]:
        """
        Get values for a specific row.
        
        Args:
            row_index (int): The row index
            
        Returns:
            Dict[CK, Any]: Dictionary mapping column keys to values
        """
        with self._rlock:
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise ValueError(f"Row index {row_index} is out of bounds.")
            
            row_dict = {}
            for column_key in self._column_keys:
                internal_column_name = self._internal_dataframe_column_strings[column_key]
                row_dict[column_key] = self._internal_canonical_dataframe.loc[row_index, internal_column_name]
            
            return row_dict

    # ----------- Row Access: Iteration ------------

    def iterrows(self) -> Iterator[tuple[int, Dict[CK, Any]]]:
        """
        Iterate over rows as (index, row_dict) pairs.
        
        Yields:
            tuple[int, Dict[CK, Any]]: Row index and row dictionary
        """
        with self._rlock:
            for row_index in range(len(self._internal_canonical_dataframe)):
                row_dict = {}
                for column_key in self._column_keys:
                    internal_column_name = self._internal_dataframe_column_strings[column_key]
                    row_dict[column_key] = self._internal_canonical_dataframe.loc[row_index, internal_column_name]
                yield row_index, row_dict

    def get_iterator_for_row(self, row_index: int) -> Iterator[tuple[CK, Any]]:
        """
        Get an iterator for a specific row's (column_key, value) pairs.
        
        Args:
            row_index (int): The row index
            
        Yields:
            tuple[CK, Any]: Column key and value pairs
        """
        with self._rlock:
            if row_index < 0 or row_index >= len(self._internal_canonical_dataframe):
                raise ValueError(f"Row index {row_index} is out of bounds.")
            
            for column_key in self._column_keys:
                internal_column_name = self._internal_dataframe_column_strings[column_key]
                value = self._internal_canonical_dataframe.loc[row_index, internal_column_name]
                yield column_key, value 