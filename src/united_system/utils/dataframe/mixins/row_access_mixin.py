"""
Row access mixin for UnitedDataframe.

Contains all operations related to row access patterns, including
head, tail, first, last, and row iteration.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Sequence
from .dataframe_protocol import UnitedDataframeProtocol, CK, SCALAR_TYPE
from ..accessors._row_accessor import RowAccessor

class RowAccessMixin(UnitedDataframeProtocol[CK]):
    """
    Row access mixin for UnitedDataframe.
    
    Provides all functionality related to row access patterns, including
    head, tail, first, last, and row iteration.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    def _row_get_as_dict(self, row_index: int, column_keys: Sequence[CK]|None = None) -> dict[CK, SCALAR_TYPE]:
        """
        Internal: Get a row as a dictionary. (no lock, no read-only check)
        """
        if column_keys is None:
            column_keys = self._column_keys
        return {column_key: self._cell_get_value(row_index, column_key) for column_key in column_keys}

    def row_get_as_dict(self, row_index: int, column_keys: list[CK]|None = None) -> dict[CK, SCALAR_TYPE]:
        """
        Get a row as a dictionary with a subset of column keys.
        """
        with self._rlock:
            return self._row_get_as_dict(row_index, column_keys)
        
    def _row_get_as_row_accessor(self, row_index: int, column_keys: Sequence[CK]|None = None) -> RowAccessor[CK]:
        """
        Internal: Get a row as a row accessor. (no lock, no read-only check)
        """
        return RowAccessor[CK](self, row_index, column_keys) # type: ignore

    # ----------- Row Access: Head/Tail ------------

    def row_get_head(self, n: int = 5) -> list[dict[CK, SCALAR_TYPE]]:
        """
        Get the first n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 5)
            
        Returns:
            List[Dict[CK, Any]]: List of row dictionaries
        """
        with self._rlock:
            if n < 0:
                raise ValueError("n must be non-negative.")
            n = min(n, self._number_of_rows())
            rows: list[dict[CK, SCALAR_TYPE]] = []
            for row_index in range(n):
                rows.append(self._row_get_as_dict(row_index))
            return rows

    def row_get_tail(self, n: int = 5) -> list[dict[CK, SCALAR_TYPE]]:
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
            n = min(n, self._number_of_rows())
            rows: list[dict[CK, SCALAR_TYPE]] = []
            for row_index in range(self._number_of_rows() - n, self._number_of_rows()):
                rows.append(self._row_get_as_dict(row_index))
            return rows

    def row_get_first(self) -> dict[CK, SCALAR_TYPE]:
        """
        Get the first row of the dataframe.
        
        Returns:
            Dict[CK, Any]: Dictionary representing the first row
        """
        with self._rlock:
            if self._number_of_rows() == 0:
                raise ValueError("Dataframe is empty.")
            return self._row_get_as_dict(0)

    def row_get_last(self) -> dict[CK, SCALAR_TYPE]:
        """
        Get the last row of the dataframe.
        
        Returns:
            Dict[CK, Any]: Dictionary representing the last row
        """
        with self._rlock:
            if self._number_of_rows() == 0:
                raise ValueError("Dataframe is empty.")
            return self._row_get_as_dict(self._number_of_rows() - 1)