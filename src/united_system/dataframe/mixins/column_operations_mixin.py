"""
Column operations mixin for UnitedDataframe.

Contains all operations related to column CRUD operations, including
addition, removal, and column data manipulation.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Union
from .dataframe_protocol import UnitedDataframeMixin, CK
from ..column_type import ColumnType
from ...units.base_classes.base_dimension import BaseDimension
from ...units.base_classes.base_unit import BaseUnit
from ...units.united import United
from ...arrays.base_classes.base_array import BaseArray

class ColumnOperationsMixin(UnitedDataframeMixin[CK]):
    """
    Column operations mixin for UnitedDataframe.
    
    Provides all functionality related to column CRUD operations, including
    addition, removal, and column data manipulation.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Column Operations: CRUD ------------

    def add_column(self, column_key: CK, column_data: BaseArray, column_type: ColumnType, dimension: BaseDimension, display_unit: Union[United, BaseUnit]) -> None:
        """
        Add a new column to the dataframe.
        
        Args:
            column_key (CK): The column key
            column_data (BaseArray): The column data
            column_type (ColumnType): The column type
            dimension (BaseDimension): The dimension
            display_unit (Union[United, BaseUnit]): The display unit
            
        Raises:
            ValueError: If the dataframe is read-only or the column already exists
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key in self._column_keys:
                raise ValueError(f"Column key {column_key} already exists in the dataframe.")
            
            # Check that the column data has the right length
            if len(column_data) != len(self._internal_canonical_dataframe):
                raise ValueError(f"Column data length ({len(column_data)}) does not match dataframe length ({len(self._internal_canonical_dataframe)}).")
            
            # Convert BaseUnit to United if needed
            if isinstance(display_unit, BaseUnit):
                display_unit = United(display_unit)
            
            # Add to internal structures
            self._column_keys.append(column_key)
            self._column_types[column_key] = column_type
            self._dimensions[column_key] = dimension
            self._display_units[column_key] = display_unit
            
            # Add to internal dataframe
            internal_column_name = self.create_internal_dataframe_column_name(column_key)
            self._internal_dataframe_column_strings[column_key] = internal_column_name
            self._internal_canonical_dataframe[internal_column_name] = column_data.to_pandas()

    def remove_column(self, column_key: CK) -> None:
        """
        Remove a column from the dataframe.
        
        Args:
            column_key (CK): The column key to remove
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Remove from internal structures
            self._column_keys.remove(column_key)
            del self._column_types[column_key]
            del self._dimensions[column_key]
            del self._display_units[column_key]
            
            # Remove from internal dataframe
            internal_column_name = self._internal_dataframe_column_strings.pop(column_key)
            self._internal_canonical_dataframe.drop(columns=[internal_column_name], inplace=True)

    def replace_column(self, column_key: CK, column_data: BaseArray) -> None:
        """
        Replace the data in a column.
        
        Args:
            column_key (CK): The column key
            column_data (BaseArray): The new column data
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Check that the column data has the right length
            if len(column_data) != len(self._internal_canonical_dataframe):
                raise ValueError(f"Column data length ({len(column_data)}) does not match dataframe length ({len(self._internal_canonical_dataframe)}).")
            
            # Replace in internal dataframe
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            self._internal_canonical_dataframe[internal_column_name] = column_data.to_pandas()

    def get_column(self, column_key: CK) -> BaseArray:
        """
        Get the column data for a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BaseArray: The column data
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            pandas_series = self._internal_canonical_dataframe[internal_column_name]
            
            # Convert pandas series back to appropriate array type
            column_type = self._column_types[column_key]
            dimension = self._dimensions[column_key]
            display_unit = self._display_units[column_key]
            
            # Create array based on column type
            from ...arrays.utils import create_array_from_pandas
            return create_array_from_pandas(pandas_series, column_type, dimension, display_unit) 