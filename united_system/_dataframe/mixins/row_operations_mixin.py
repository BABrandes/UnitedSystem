"""
Row operations mixin for UnitedDataframe.

Contains all operations related to row CRUD operations, including
addition, removal, and row data manipulation.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""
import pandas as pd
from typing import TYPE_CHECKING, Sequence

from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._utils.value_type import VALUE_TYPE, VALUE_TYPE_RUNTIME
from ..._utils.scalar_type import SCALAR_TYPE, SCALAR_TYPE_RUNTIME

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class RowOperationsMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Row operations mixin for UnitedDataframe.
    
    Provides all functionality related to row CRUD operations, including
    addition, removal, and row data manipulation.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Row Operations: CRUD ------------

    ##### Base row operations start #####

    def _row_insert_empty(self, row_index: int) -> None:
        """
        Internal: Insert a row at a specific index. (no lock)
        
        Args:
            row_index (int): The index where to insert the row
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        if self._read_only:
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
        
        if row_index < 0 or row_index >= len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds for insertion.")
        
        # Create empty rows and insert them at the correct index
        empty_rows = pd.DataFrame(index=range(row_index, row_index + 1), columns=self._internal_dataframe.columns)
        self._internal_dataframe = pd.concat([self._internal_dataframe.iloc[:row_index], empty_rows, self._internal_dataframe.iloc[row_index:]], ignore_index=True) 

    def _row_add_empty(self, number_of_rows: int = 1) -> None:
        """
        Internal: Add empty rows to the end of the dataframe. (no lock)
        
        Args:
            number_of_rows (int): Number of empty rows to add
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        if self._read_only:
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
        
        empty_rows = pd.DataFrame(index=range(len(self._internal_dataframe), len(self._internal_dataframe) + number_of_rows), columns=self._internal_dataframe.columns)  
        self._internal_dataframe = pd.concat([self._internal_dataframe, empty_rows], ignore_index=True) 

    def _row_set_items(self, row_index: int, values: dict[CK, VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Internal: Set multiple row values in the dataframe from a dictionary mapping column keys to lists of values. (no lock)

        Args:
            row_index (int): Row index (start) or slice object.
            values (dict[CK, VALUE_TYPE|SCALAR_TYPE]): Dictionary mapping column keys to lists of values.

        Raises:
            ValueError: If dataframe is read-only, values are inconsistent, or column types mismatch.
        """
        if self._read_only:
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
        
        for column_key, value in values.items():
            if isinstance(value, SCALAR_TYPE_RUNTIME):
                self._cell_set_scalar(row_index, column_key, value)
            else:
                self._cell_set_value(row_index, column_key, value) # type: ignore

    def _row_set_ordered_items(self, row_index: int, values: Sequence[VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Internal: Set rows with values at a specific index. (no lock)
        The values are expected to be correctly ordered according to the column keys.

        Args:
            row_index (int): The index where to set the values
            values (Sequence[Any]): Sequence of values to set

        Raises:
            ValueError: If the dataframe is read-only or values are invalid
        """

        if not len(values) == len(self._column_keys):
            raise ValueError(f"Number of values {len(values)} does not match number of columns {len(self._column_keys)}.")
        
        # Direct cell operations for performance
        for col_index, column_key in enumerate(self._column_keys):
            if isinstance(values[col_index], SCALAR_TYPE_RUNTIME):
                self._cell_set_scalar(row_index, column_key, values[col_index])
            else:
                self._cell_set_value(row_index, column_key, values[col_index]) # type: ignore

    def _row_remove(self, row_index_start_inclusive: int, row_index_stop_exclusive: int) -> None:
        """
        Internal: Remove rows from the dataframe. (no lock)
        """
        if self._read_only:
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
        
        if row_index_start_inclusive < 0 or row_index_start_inclusive >= row_index_stop_exclusive:
            raise ValueError(f"Row index start {row_index_start_inclusive} is out of bounds.")
        
        if row_index_stop_exclusive < 0 or row_index_stop_exclusive > len(self._internal_dataframe):
            raise ValueError(f"Row index stop {row_index_stop_exclusive} is out of bounds.")
        
        self._internal_dataframe.drop(index=range(row_index_start_inclusive, row_index_stop_exclusive), inplace=True)
        self._internal_dataframe.reset_index(drop=True, inplace=True)

    ##### Base row operations end #####

    def row_add_empty(self, number_of_rows: int = 1) -> None:
        """
        Add empty rows to the end of the dataframe.
        
        Args:
            number_of_rows (int): Number of empty rows to add
            
        Raises:
            ValueError: If the dataframe is read-only or number_of_rows is invalid
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_add_empty(number_of_rows)

    def row_set_ordered_items(self, row_index: int, values: Sequence[VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Set rows with values at a specific index. The values are expected to be correctly ordered according to the column keys.

        Args:
            row_index (int): The index where to set the values
            values (Sequence[VALUE_TYPE]): Sequence of values to set

        Raises:
            ValueError: If the dataframe is read-only or values are invalid
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_set_ordered_items(row_index, values)

    def row_add_ordered_items(self, values: Sequence[VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Add rows with values to the end of the dataframe. The values are expected to be correctly ordered according to the column keys.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_add_empty(1)
            self._row_set_ordered_items(self._number_of_rows()-1, values)


    def row_add_items(self, values: dict[CK, VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Add rows with values to the end of the dataframe.
        
        Args:
            values (dict[CK, VALUE_TYPE|SCALAR_TYPE]): Dictionary mapping column keys to lists of values or single values
            
        Raises:
            ValueError: If the dataframe is read-only or values are invalid
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")

            self._row_add_empty(1)
            self._row_set_items(self._number_of_rows()-1, values)

    def row_insert_empty(self, row_index: int) -> None:
        """
        Insert empty rows at a specific index.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")

            self._row_insert_empty(row_index)

    def row_insert_items(self, row_index: int, values: dict[CK, VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Insert rows with values at a specific index.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_insert_empty(row_index)
            self._row_set_items(row_index, values)

    def row_set_items(self, row_index: int, values: dict[CK, VALUE_TYPE|SCALAR_TYPE]) -> None:
        """
        Replace rows with values at a specific index.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_set_items(row_index, values)

    def row_clear(self, row_index_start_inclusive: int, number_of_rows: int) -> None:
        """
        Clear rows at a specific index.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if row_index_start_inclusive < 0 or row_index_start_inclusive >= self._number_of_rows():
                raise ValueError(f"Start row index {row_index_start_inclusive} is out of bounds for insertion.")
            
            if number_of_rows < 0 or row_index_start_inclusive + number_of_rows > self._number_of_rows():
                raise ValueError(f"Number of rows {number_of_rows} is out of bounds for insertion.")
            
            # Clear each row using proper missing value handling
            for i in range(number_of_rows):
                row_index = row_index_start_inclusive + i
                for column_key in self._column_keys:
                    self._cell_set_missing(row_index, column_key)
            
    def row_remove(self, row_index_start_inclusive: int, row_index_stop_exclusive: int) -> None:
        """
        Remove rows from the dataframe.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if row_index_start_inclusive < 0 or row_index_start_inclusive >= row_index_stop_exclusive:
                raise ValueError(f"Row index start {row_index_start_inclusive} is out of bounds.")
            
            if row_index_stop_exclusive < 0 or row_index_stop_exclusive > self._number_of_rows():
                raise ValueError(f"Row index stop {row_index_stop_exclusive} is out of bounds.")
            
            self._row_remove(row_index_start_inclusive, row_index_stop_exclusive)

    def row_clear_all(self) -> None:
        """
        Clear all rows from the dataframe.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Clear all rows using proper missing value handling
            for row_index in range(self._number_of_rows()):
                for column_key in self._column_keys:
                    self._cell_set_missing(row_index, column_key)

    def row_remove_all(self) -> None:
        """
        Remove all rows from the dataframe.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_remove(0, self._number_of_rows())

    def _row_find_index_by_item(self, column_key: CK, value: VALUE_TYPE|SCALAR_TYPE) -> list[int]:
        """
        Internal: Find all row indices where the value in the specified column matches the given value. (no lock)

        Args:
            column_key (CK): The column key to search in
            value (VALUE_TYPE|SCALAR_TYPE): The value to search for

        Returns:
            list[int]: List of row indices where the value matches
        """
              
        if isinstance(value, SCALAR_TYPE_RUNTIME):
            _value: VALUE_TYPE = self._column_types[column_key].get_value_for_dataframe(value, self._column_units[column_key])
        elif isinstance(value, VALUE_TYPE_RUNTIME):
            assert isinstance(value, VALUE_TYPE)
            _value: VALUE_TYPE = value
        else:
            raise ValueError(f"Invalid value type: {type(value)}")

        _row_indices: list[int] = self._internal_dataframe[self._internal_dataframe[self._internal_dataframe_column_names[column_key]] == _value].index.tolist() # type: ignore

        return _row_indices
    
    def row_find_index_by_item(self, column_key: CK, value: VALUE_TYPE|SCALAR_TYPE) -> list[int]:
        """
        Find all row indices where the value in the specified column matches the given value.
        """
        with self._rlock:
            return self._row_find_index_by_item(column_key, value)