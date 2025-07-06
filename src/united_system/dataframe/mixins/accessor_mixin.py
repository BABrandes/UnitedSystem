"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the core access patterns for the dataframe including
iteration, indexing, and item assignment through magic methods.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Iterator, Union, overload, TYPE_CHECKING
from .dataframe_protocol import UnitedDataframeMixin, UnitedDataframeProtocol, CK
from ..column_type import SCALAR_TYPE
from ..accessors._row_accessor import _RowAccessor
from ..accessors._column_accessor import _ColumnAccessor

if TYPE_CHECKING:
    from ...united_dataframe import UnitedDataframe
    from ..accessors._row_accessor import _RowAccessor
    from ..accessors._column_accessor import _ColumnAccessor

class AccessorMixin(UnitedDataframeMixin[CK]):
    """
    Mixin providing magic methods for dataframe access patterns.
    
    This mixin implements:
    - __iter__: Iterate over rows using _RowAccessor
    - __getitem__: Comprehensive indexing (columns, rows, cells, slices)
    - __setitem__: Cell value assignment using tuple syntax
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """
    
    def __iter__(self) -> Iterator[_RowAccessor[CK]]:
        """
        Iterate over the rows of the dataframe using _RowAccessor.
        
        Returns:
            Iterator[_RowAccessor[CK]]: An iterator over row accessors
        """
        with self._rlock:  # Now IDE knows _rlock exists!
            for row_index in range(len(self._internal_canonical_dataframe)):  # And _internal_canonical_dataframe!
                yield _RowAccessor(self, row_index)

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: CK) -> _ColumnAccessor[CK]:
        """
        Get a column accessor for pandas-like column access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK): The column index or column key
            
        Returns:
            ColumnAccessor[CK]: An accessor object for the specified column
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int) -> _RowAccessor[CK]:
        """
        Get a row accessor for pandas-like row access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK): The column index or column key
            
        Returns:
            RowAccessor[CK]: An accessor object for the specified row
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[CK]|set[CK]) -> UnitedDataframeProtocol[CK]:
        """
        Get a new dataframe with the selected columns.
        
        Args:
            index_or_column_key_or_list_of_keys (list[int|CK]): The column indices or column keys
            
        Returns:
            UnitedDataframeProtocol[CK]: A new dataframe with the selected columns as a shallow copy
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[int]|set[int]|slice) -> UnitedDataframeProtocol[CK]:
        """
        Get a new dataframe with the selected rows.
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: tuple[int, CK]|tuple[CK, int]) -> SCALAR_TYPE:
        """
        Get a cell value for pandas-like cell access.
        """
        ...

    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int|CK|list[int]|set[int]|slice|list[CK]|set[CK]|tuple[int, CK]|tuple[CK, int]) -> _ColumnAccessor[CK] | _RowAccessor[CK] | UnitedDataframeProtocol[CK] | SCALAR_TYPE:
        """
        Get a column accessor, row accessor, subset dataframe, or cell value.
        
        Args:
            column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: The access pattern
            
        Returns:
            ColumnAccessor[CK] | RowAccessor[CK] | UnitedDataframeProtocol[CK] | SCALAR_TYPE: The result based on access pattern
            
        Raises:
            ValueError: If the column doesn't exist or invalid access pattern
        """
        with self._rlock:  # Full IDE support here too!
            match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position:
                case int():
                    return _RowAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case str() if hasattr(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position, '__class__'):  # Handle ColumnKey too
                    return _ColumnAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case slice():
                    new_united_dataframe: UnitedDataframe[CK] = self.copy(deep=True)
                    # Convert slice to list of indices
                    start, stop, step = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position.indices(len(self._internal_canonical_dataframe))
                    indices_to_remove = list(range(start, stop, step))
                    if indices_to_remove:
                        new_united_dataframe.rows_remove_by_indices(indices_to_remove)  # Protocol knows this exists!
                    return new_united_dataframe
                case list() | set():
                    if len(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) == 0:
                        # Import here to avoid circular imports
                        from ...united_dataframe import UnitedDataframe
                        return UnitedDataframe.create_empty([], [])
                    if isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), int):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.rows_remove_by_indices(list(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position))
                        return new_united_dataframe
                    else:  # Column keys
                        new_united_dataframe = self.copy(deep=True)
                        # Remove columns one by one (since remove_column is singular)
                        for column_key in column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position:
                            if new_united_dataframe.has_column(column_key):  # Protocol knows has_column exists!
                                new_united_dataframe.remove_column(column_key)  # And remove_column!
                        return new_united_dataframe
                case tuple():
                    if len(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) == 2:
                        pos1, pos2 = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position
                        if isinstance(pos1, int):
                            # (row_index, column_key)
                            return self.cell_get_value(pos1, pos2)  # Protocol knows cell_get_value exists!
                        else:
                            # (column_key, row_index)
                            return self.cell_get_value(pos2, pos1)
                    else:
                        raise ValueError(f"Invalid tuple length: {len(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)}")
                case _:
                    raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}")

    def __setitem__(self, cell_position: tuple[int, CK]|tuple[CK, int], value: SCALAR_TYPE):
        """
        Set a cell value using pandas-like syntax.
        
        Args:
            cell_position (tuple[int, CK]|tuple[CK, int]): The cell position
            value (UnitedScalar): The value to set
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist, or the row is out of bounds
        """
        with self._wlock:  # Full IDE support for _wlock too!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if len(cell_position) != 2:
                raise ValueError(f"Cell position must be a tuple of length 2, got {len(cell_position)}")
            
            pos1, pos2 = cell_position
            if isinstance(pos1, int):
                # (row_index, column_key)
                row_index, column_key = pos1, pos2
            else:
                # (column_key, row_index)
                column_key, row_index = pos1, pos2
                
            if not self.has_column(column_key):  # Protocol provides has_column!
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not (0 <= row_index < len(self._internal_canonical_dataframe)):  # And _internal_canonical_dataframe!
                raise ValueError(f"The row index {row_index} does not exist. The dataframe has {len(self)} rows.")
            
            self.cell_set_value(row_index, column_key, value)  # And cell_set_value! 