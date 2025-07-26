"""
Row operations mixin for UnitedDataframe.

Contains all operations related to row CRUD operations, including
addition, removal, and row data manipulation.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""
import pandas as pd
from typing import Any, Dict, TYPE_CHECKING

from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..column_type import ColumnType
from united_system._units_and_dimension.unit import Unit
from united_system._scalars.united_scalar import UnitedScalar
from united_system._arrays.base_array import PT_TYPE

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

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

    def _row_insert_empty(self, row_index: int, number_of_rows: int) -> None:
        """
        Internal: Insert a row at a specific index. (no lock)
        
        Args:
            row_index (int): The index where to insert the row
            row_data (Dict[CK, Any]): Dictionary mapping column keys to values
            
        Raises:
            ValueError: If the dataframe is read-only or parameters are invalid
        """
        if self._read_only:
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
        
        if row_index < 0 or row_index > len(self._internal_dataframe):
            raise ValueError(f"Row index {row_index} is out of bounds for insertion.")
        
        # Create empty rows
        empty_rows = pd.DataFrame(index=range(row_index, row_index + number_of_rows), columns=self._internal_dataframe.columns)
        self._internal_dataframe = pd.concat([self._internal_dataframe.iloc[:row_index], empty_rows, self._internal_dataframe.iloc[row_index:]], ignore_index=True)      

    def _row_set_values(self, row_index_or_slice: int | slice, values: dict[CK, list[UnitedScalar[Any, Any]|PT_TYPE|None]]) -> None:
        """
        Internal: Set multiple row values in the dataframe from a dictionary mapping column keys to lists of values. (no lock)

        Args:
            row_index_or_slice (int | slice): Row index (start) or slice object.
            values (dict[CK, list[Any]]): Dictionary mapping column keys to lists of values.

        Raises:
            ValueError: If dataframe is read-only, values are inconsistent, or column types mismatch.
        """
        if self._read_only:
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
        
        if not values:
            return

        num_rows: int = len(next(iter(values.values())))

        # Validate all inputs and convert values
        converted_values: dict[str, list[Any]] = {}
        for column_key, value_list in values.items():
            if len(value_list) != num_rows:
                raise ValueError(f"Column '{column_key}' has {len(value_list)} values, expected {num_rows}.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key '{column_key}' does not exist in the dataframe.")
            
            column_type: ColumnType = self._column_types[column_key]
            column_unit: Unit | None = self._column_units[column_key]
            internal_column_name = self._internal_dataframe_column_names[column_key]
            converted_column: list[Any] = []

            for value in value_list:
                if column_type.has_unit and value is not None:
                    if not isinstance(value, UnitedScalar):
                        raise ValueError(f"Value '{value}' in column '{column_key}' is not a UnitedScalar.")
                    if value.unit != column_unit:
                        raise ValueError(
                            f"Value '{value}' has unit {value.unit}, "
                            f"but expected {column_unit} for column '{column_key}'."
                        )
                converted_column.append(column_type.get_value_for_dataframe(value, column_unit))
            
            converted_values[internal_column_name] = converted_column

        # Determine row range
        if isinstance(row_index_or_slice, int):
            row_start = row_index_or_slice
            row_stop = row_start + num_rows
        else:
            if row_index_or_slice.step not in (None, 1):
                raise ValueError("Only slices with step=None or step=1 are supported.")
            if row_index_or_slice.start is None or row_index_or_slice.stop is None:
                raise ValueError("Slice must have start and stop defined.")
            row_start = row_index_or_slice.start
            row_stop = row_index_or_slice.stop
            if row_stop - row_start != num_rows:
                raise ValueError(
                    f"Slice range {row_stop - row_start} does not match number of rows in values: {num_rows}."
                )

        # Assign using .loc with new DataFrame
        self._internal_dataframe.loc[row_start:row_stop - 1, list(converted_values.keys())] = pd.DataFrame(converted_values, index=range(row_start, row_stop))

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

    def row_add_empty(self, number_of_rows: int) -> None:
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
            
            self._row_insert_empty(len(self._internal_dataframe), number_of_rows)

    def row_add_values(self, values: Dict[CK, list[Any] | Dict[CK, Any]]) -> None:
        """
        Add rows with values to the end of the dataframe.
        
        Args:
            values (Dict[CK, list[Any] | Dict[CK, Any]]): Dictionary mapping column keys to lists of values or single values
            
        Raises:
            ValueError: If the dataframe is read-only or values are invalid
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")

            if len(values) == 0:
                raise ValueError("No values provided. Use 'add_empty_rows' instead!")

            # Transform a dict of any to a dict of lists
            if isinstance(next(iter(values.values())), (list, tuple)):
                # Already a list of values
                pass
            else:
                # Single value per column, convert to list
                values = {column_key: [values[column_key]] for column_key in values}

            number_of_rows_to_add = len(next(iter(values.values())))
            start_index = len(self._internal_dataframe)

            self._row_insert_empty(start_index, number_of_rows_to_add)
            self._row_set_values(slice(start_index, start_index + number_of_rows_to_add), values) # type: ignore

    def row_insert_empty(self, row_index: int|slice, number_of_rows: int) -> None:
        """
        Insert empty rows at a specific index.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")

            if isinstance(row_index, slice):
                if row_index.step not in (None, 1):
                    raise ValueError("Only slices with step=None or step=1 are supported.")
                if row_index.start is None or row_index.stop is None:
                    raise ValueError("Slice must have start and stop defined.")
                insert_index = row_index.start
            else:
                insert_index = row_index
            
            self._row_insert_empty(insert_index, number_of_rows)

    def row_insert_values(self, row_index: int|slice, values: Dict[CK, list[Any]|Dict[CK, Any]]) -> None:
        """
        Insert rows with values at a specific index.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if len(values) == 0:
                raise ValueError("No values provided. Use 'insert_empty_rows' instead!")
            
            # Transform a dict of any to a dict of lists
            if isinstance(next(iter(values.values())), (list, tuple)):
                # Already a list of values
                pass
            else:
                # Single value per column, convert to list
                values = {column_key: [values[column_key]] for column_key in values}
            
            number_of_rows_to_add = len(next(iter(values.values())))
            
            # Handle slice vs int for row_index_or_slice
            if isinstance(row_index, slice):
                if row_index.step not in (None, 1):
                    raise ValueError("Only slices with step=None or step=1 are supported.")
                if row_index.start is None or row_index.stop is None:
                    raise ValueError("Slice must have start and stop defined.")
                insert_index = row_index.start
            else:
                insert_index = row_index
            
            self._row_insert_empty(insert_index, number_of_rows_to_add)
            self._row_set_values(slice(insert_index, insert_index + number_of_rows_to_add), values) # type: ignore

    def row_set_values(self, row_index: int|slice, values: Dict[CK, list[Any]|Dict[CK, Any]]) -> None:
        """
        Replace rows with values at a specific index.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if len(values) == 0:
                raise ValueError("No values provided!")
            
            # Transform a dict of any to a dict of lists
            if isinstance(next(iter(values.values())), (list, tuple)):
                # Already a list of values
                pass
            else:
                # Single value per column, convert to list
                values = {column_key: [values[column_key]] for column_key in values}
            
            number_of_rows_to_replace = len(next(iter(values.values())))
            
            # Handle slice vs int for row_index_or_slice
            if isinstance(row_index, slice):
                if row_index.step not in (None, 1):
                    raise ValueError("Only slices with step=None or step=1 are supported.")
                if row_index.start is None or row_index.stop is None:
                    raise ValueError("Slice must have start and stop defined.")
                row_start = row_index.start
                row_stop = row_index.stop
            else:
                row_start = row_index
                row_stop = row_start + number_of_rows_to_replace
            
            self._row_remove(row_start, row_stop)
            self._row_insert_empty(row_start, number_of_rows_to_replace)
            self._row_set_values(slice(row_start, row_start + number_of_rows_to_replace), values) # type: ignore

    def row_clear(self, row_index_start_inclusive: int, number_of_rows: int) -> None:
        """
        Clear rows at a specific index.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if row_index_start_inclusive < 0 or row_index_start_inclusive >= len(self._internal_dataframe):
                raise ValueError(f"Start row index {row_index_start_inclusive} is out of bounds for insertion.")
            
            if number_of_rows < 0 or row_index_start_inclusive + number_of_rows > len(self._internal_dataframe):
                raise ValueError(f"Number of rows {number_of_rows} is out of bounds for insertion.")
            
            self._row_set_values(slice(row_index_start_inclusive, row_index_start_inclusive + number_of_rows), {column_key: [None] * number_of_rows for column_key in self._column_keys}) # type: ignore
            
    def row_remove(self, row_index_start_inclusive: int, row_index_stop_exclusive: int) -> None:
        """
        Remove rows from the dataframe.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if row_index_start_inclusive < 0 or row_index_start_inclusive >= row_index_stop_exclusive:
                raise ValueError(f"Row index start {row_index_start_inclusive} is out of bounds.")
            
            if row_index_stop_exclusive < 0 or row_index_stop_exclusive > len(self._internal_dataframe):
                raise ValueError(f"Row index stop {row_index_stop_exclusive} is out of bounds.")
            
            self._row_remove(row_index_start_inclusive, row_index_stop_exclusive)

    def row_clear_all(self) -> None:
        """
        Clear all rows from the dataframe.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_set_values(slice(0, len(self._internal_dataframe)), {column_key: [None] * len(self._internal_dataframe) for column_key in self._column_keys}) # type: ignore

    def row_remove_all(self) -> None:
        """
        Remove all rows from the dataframe.
        """
        with self._wlock:

            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            self._row_remove(0, len(self._internal_dataframe))