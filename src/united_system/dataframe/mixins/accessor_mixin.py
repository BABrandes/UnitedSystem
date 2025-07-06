"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the core access patterns for the dataframe including
iteration, indexing, and item assignment through magic methods.
"""

from typing import TypeVar, Generic, Iterator, Union, overload, TYPE_CHECKING
from ..column_key import ColumnKey
from ..column_type import SCALAR_TYPE
from ..accessors._row_accessor import _RowAccessor
from ..accessors._column_accessor import _ColumnAccessor

if TYPE_CHECKING:
    from ...united_dataframe import UnitedDataframe

CK = TypeVar("CK", bound=Union[ColumnKey, str])


class AccessorMixin(Generic[CK]):
    """
    Mixin providing magic methods for dataframe access patterns.
    
    This mixin implements:
    - __iter__: Iterate over rows using _RowAccessor
    - __getitem__: Comprehensive indexing (columns, rows, cells, slices)
    - __setitem__: Cell value assignment using tuple syntax
    """
    
    def __iter__(self) -> Iterator[_RowAccessor[CK]]:
        """
        Iterate over the rows of the dataframe using _RowAccessor.
        
        Returns:
            Iterator[_RowAccessor[CK]]: An iterator over row accessors
        """
        with self._rlock:
            for row_index in range(len(self._internal_canonical_dataframe)):
                yield _RowAccessor(self, row_index)

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: CK) -> "_ColumnAccessor[CK]":
        """
        Get a column accessor for pandas-like column access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK): The column index or column key
            
        Returns:
            ColumnAccessor[CK]: An accessor object for the specified column
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int) -> "_RowAccessor[CK]":
        """
        Get a row accessor for pandas-like row access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK): The column index or column key
            
        Returns:
            RowAccessor[CK]: An accessor object for the specified row
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[CK]|set[CK]) -> "UnitedDataframe[CK]":
        """
        Get a new dataframe with the selected columns.
        
        Args:
            index_or_column_key_or_list_of_keys (list[int|CK]): The column indices or column keys
            
        Returns:
            UnitedDataframe[CK]: A new dataframe with the selected columns as a shallow copy
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[int]|set[int]|slice) -> "UnitedDataframe[CK]":
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

    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int|CK|list[int]|set[int]|slice|list[CK]|set[CK]|tuple[int, CK]|tuple[CK, int]) -> "_ColumnAccessor[CK] | _RowAccessor[CK] | UnitedDataframe[CK] | SCALAR_TYPE":
        """
        Get a column accessor, row accessor, subset dataframe, or cell value.
        
        Args:
            column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: The access pattern
            
        Returns:
            ColumnAccessor[CK] | RowAccessor[CK] | UnitedDataframe[CK] | SCALAR_TYPE: The result based on access pattern
            
        Raises:
            ValueError: If the column doesn't exist or invalid access pattern
        """
        with self._rlock:
            match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position:
                case int():
                    return _RowAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case ColumnKey()|str():
                    return _ColumnAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case slice():
                    new_united_dataframe = self.copy(deep=True)
                    new_united_dataframe.remove_rows(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                    return new_united_dataframe
                case list() | set():
                    if len(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) == 0:
                        from ...united_dataframe import UnitedDataframe
                        return UnitedDataframe[CK].create_empty([], [], [], 0, self._internal_dataframe_column_name_formatter)
                    if isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), int):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.remove_rows(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) # type: ignore[arg-type]
                        return new_united_dataframe
                    elif isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), ColumnKey|str):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.remove_columns(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) # type: ignore[arg-type]
                        return new_united_dataframe
                    else:
                        raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                case tuple():
                    match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0], column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1]:
                        case int(), ColumnKey()|str():
                            row_index: int = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0] # type: ignore[assignment]
                            column_key: CK = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1] # type: ignore[assignment]
                            return self.cell_value_get(row_index, column_key)
                        case ColumnKey()|str(), int():
                            column_key: CK = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0] # type: ignore[assignment]
                            row_index: int = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1] # type: ignore[assignment]
                            return self.cell_value_get(row_index, column_key)
                        case _:
                            raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                case _:
                    raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")

    def __setitem__(self, cell_position: tuple[int, CK]|tuple[CK, int], value: SCALAR_TYPE):
        """
        Set a cell value using pandas-like syntax.
        
        Args:
            cell_position (tuple[int, CK]|tuple[CK, int]): The cell position
            value (UnitedScalar): The value to set
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist, or the row is out of bounds
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            match cell_position:
                case int(), ColumnKey()|str():
                    row_index: int = cell_position[0]
                    column_key: CK = cell_position[1]
                case ColumnKey()|str(), int():
                    column_key: CK = cell_position[0]
                    row_index: int = cell_position[1]
                case _:
                    raise ValueError(f"Invalid key: {cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not (0 <= row_index < len(self._internal_canonical_dataframe)):
                raise ValueError(f"The row index {row_index} does not exist. The dataframe has {len(self)} rows.")
            self.cell_value_set(row_index, column_key, value) 