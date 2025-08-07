"""
Column operations mixin for UnitedDataframe.

Contains all operations related to column CRUD operations, including
addition, removal, and column data manipulation.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, TypeVar, Optional
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._dataframe.column_type import ColumnType, ARRAY_TYPE
from ..._units_and_dimension.unit import Unit
from ..._units_and_dimension.has_unit_protocol import HasUnit
import numpy as np

AT = TypeVar("AT", bound=ARRAY_TYPE)

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class ColumnOperationsMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Column operations mixin for UnitedDataframe.
    
    Provides all functionality related to column CRUD operations, including
    addition, removal, and column data manipulation.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Column Operations: CRUD ------------

    def column_add_empty(self, column_key: CK, column_type: ColumnType, unit: Optional[Unit]) -> None:
        """
        Add a new empty column to the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._column_add_empty(column_key, column_type, unit)

    def _column_add_empty(self, column_key: CK, column_type: ColumnType, unit: Optional[Unit]) -> None:
        """
        Internal: Add a new empty column to the dataframe. (no lock, no read-only check)
        
        Args:
            column_key (CK): The column key
            column_type (ColumnType): The column type
            unit (Optional[Unit]): The unit. Can be None only if the column type does not have a unit.
            
        Raises:
            ValueError: If the dataframe is read-only or the column already exists
        """
    
        if column_key in self._column_keys:
            raise ValueError(f"Column key {column_key} already exists in the dataframe.")
        
        # Add to internal structures
        self._column_keys.append(column_key)
        self._column_types[column_key] = column_type
        self._column_units[column_key] = unit
        
        # Add to internal dataframe
        internal_column_name: str = self._create_internal_dataframe_column_name(column_key)
        self._internal_dataframe_column_names[column_key] = internal_column_name
        self._internal_dataframe[internal_column_name] = column_type.get_pd_series_for_dataframe(np.empty(len(self._internal_dataframe)), unit) # type: ignore

    def column_add(self, column_key: CK, array: ARRAY_TYPE, column_type: ColumnType, unit: Optional[Unit] = None) -> None:
        """
        Add a new column to the dataframe.

        Args:
            column_key (CK): The column key
            array (ARRAY_TYPE): The column data
            column_type (ColumnType): The column type
            unit (Optional[Unit]): The unit. If None, the unit is set according to the array.
            
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._column_add(column_key, array, column_type, unit)

    def _column_add(self, column_key: CK, array: ARRAY_TYPE, column_type: ColumnType, unit: Optional[Unit]) -> None:
        """
        Internal: Add a new column to the dataframe. (no lock, no read-only check)
        
        Args:
            column_key (CK): The column key
            array (ARRAY_TYPE): The column data
            column_type (ColumnType): The column type
            unit (Optional[Unit]): The unit. If None, the unit is set according to the array.
            
        Raises:
            ValueError: If the dataframe is read-only or the column already exists
        """
    
        if column_key in self._column_keys:
            raise ValueError(f"Column key {column_key} already exists in the dataframe.")
        
        # Check that the column data has the right length
        if len(array) != len(self._internal_dataframe):
            raise ValueError(f"Column data length ({len(array)}) does not match dataframe length ({len(self._internal_dataframe)}).")

        # Add to internal structures
        self._column_keys.append(column_key)
        self._column_types[column_key] = column_type

        # Use the provided unit if provided, otherwise infer the unit from the array
        if column_type.has_unit:
            if unit is None:
                if isinstance(array, HasUnit):
                    unit = array.unit
                else:
                    raise ValueError(f"Unit is required for column type {column_type}.")
            else:
                if not isinstance(array, HasUnit):
                    raise ValueError(f"Unit is required for column type {column_type}.")
        
        # Add to internal dataframe
        internal_column_name: str = self._create_internal_dataframe_column_name(column_key)
        self._internal_dataframe_column_names[column_key] = internal_column_name
        self._internal_dataframe[internal_column_name] = column_type.get_pd_series_for_dataframe(array, unit) # type: ignore

    def column_remove(self, column_key: CK) -> None:
        """
        Remove a column from the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._column_remove(column_key)

    def _column_remove(self, column_key: CK) -> None:
        """
        Internal: Remove a column from the dataframe. (no lock, no read-only check)
        
        Args:
            column_key (CK): The column key to remove
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """

        
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Remove from internal structures
        self._column_keys.remove(column_key)
        del self._column_types[column_key]
        del self._column_units[column_key]
        
        # Remove from internal dataframe
        internal_column_name = self._internal_dataframe_column_names.pop(column_key)
        self._internal_dataframe.drop(columns=[internal_column_name], inplace=True)

    def column_set_values(self, column_key: CK, array: ARRAY_TYPE) -> None:
        """
        Replace the data in a column.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._column_set_values(column_key, array)

    def _column_set_values(self, column_key: CK, array: ARRAY_TYPE) -> None:
        """
        Replace the data in a column.
        
        Args:
            column_key (CK): The column key
            array (ARRAY_TYPE): The new column data
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """

        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Check that the column data has the right length
        if len(array) != len(self._internal_dataframe):
            raise ValueError(f"Column data length ({len(array)}) does not match dataframe length ({len(self._internal_dataframe)}).")
        
        if not self._is_compatible_with_column(column_key, array):
            raise ValueError(f"The provided array is not compatible with the column {column_key}.")
        
        # Replace in internal dataframe
        internal_column_name: str = self._internal_dataframe_column_names[column_key]
        self._internal_dataframe[internal_column_name] = self._column_types[column_key].get_pd_series_for_dataframe(array, self._column_units[column_key], None) # type: ignore

    def column_rename(self, current_column_key: CK, new_column_key: CK):
        """
        Set the column key for a column at the specified index.
        
        Args:
            current_column_key (CK): The current column key
            new_column_key (CK): The new column key
            
        Raises:
            ValueError: If the dataframe is read-only, the name already exists,
                                       or the new column name conflicts with existing columns
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if not self._colkey_exists(current_column_key):
                raise ValueError(f"Column key {current_column_key} does not exist in the dataframe.")
            if self._colkey_exists(new_column_key):
                raise ValueError(f"Column key {new_column_key} already exists in the dataframe.")
            
            current_and_new_unit: Optional[Unit] = self._column_units[current_column_key]
            current_dataframe_column_name: str = self._internal_dataframe_column_names[current_column_key]
            new_dataframe_column_name: str = self._internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(new_column_key, current_and_new_unit)
            if new_dataframe_column_name in self._internal_dataframe.columns:
                raise ValueError(f"Column name {new_dataframe_column_name} already exists in the dataframe.")
            
            self._internal_dataframe.rename(columns={current_dataframe_column_name: new_dataframe_column_name}, inplace=True)
            self._column_keys[self._column_keys.index(current_column_key)] = new_column_key
            self._internal_dataframe_column_names.pop(current_column_key)
            self._internal_dataframe_column_names[new_column_key] = new_dataframe_column_name
            self._column_types[new_column_key] = self._column_types.pop(current_column_key)
            self._column_units[new_column_key] = self._column_units.pop(current_column_key) 