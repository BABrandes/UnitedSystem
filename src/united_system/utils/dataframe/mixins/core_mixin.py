"""
Core functionality mixin for UnitedDataframe.

Contains basic properties, initialization helpers, and core utility methods.
"""

from typing import Any, Optional, Sequence, TYPE_CHECKING
from collections.abc import Sequence
import pandas as pd
import numpy as np

from .dataframe_protocol import UnitedDataframeProtocol, CK
from ....column_type import SCALAR_TYPE, ARRAY_TYPE, ColumnType
from ...units.united import United
from ...dataframe.internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter
from ....unit import Unit

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class CoreMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Core functionality mixin for UnitedDataframe.
    
    Provides basic properties, initialization helpers, and core utility methods
    that are used throughout the dataframe implementation.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
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
            return self._number_of_rows()
        
    def _number_of_rows(self) -> int:
        """
        Internal: Get the number of rows in the dataframe (no lock).
        """
        return len(self._internal_dataframe)
    
    def _number_of_columns(self) -> int:
        """
        Internal: Get the number of columns in the dataframe (no lock).
        """
        return len(self._column_keys)
    
    @property
    def internal_dataframe_column_name_formatter(self) -> InternalDataFrameColumnNameFormatter:
        """
        Get the internal dataframe column name formatter.
        """
        return self._internal_dataframe_column_name_formatter

    def has_unit(self, column_key: CK) -> bool:
        """
        Public: Check if a column has a unit (with lock).
        """
        with self._rlock:
            return self._unit_has(column_key)

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
            return len(self._internal_dataframe)

    @property
    def size(self) -> int:
        """
        Get the total number of elements in the dataframe.
        
        Returns:
            int: Total number of elements (rows × columns)
        """
        with self._rlock:
            return self._internal_dataframe.size

    @property
    def empty(self) -> bool:
        """
        Check if the dataframe is empty.
        
        Returns:
            bool: True if the dataframe has no rows, False otherwise
        """
        with self._rlock:
            return self._internal_dataframe.empty

    # Internal utilities
    def _create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Internal: Create the internal dataframe column name for a column (no lock).
        """
        return self._internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, self._column_units[column_key])

    def create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Public: Create the internal dataframe column name for a column (with lock).

        Args:
            column_key (CK): The column key to create the internal dataframe column name for

        Returns:
            str: The internal dataframe column name for the column
        """
        with self._rlock:
            return self._create_internal_dataframe_column_name(column_key)

    def _get_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Internal: Get the internal dataframe column string for a column key (no lock).
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        return self._internal_dataframe_column_names[column_key]

    def get_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Public: Get the internal dataframe column string for a column key (with lock).
        
        Args:
            column_key (CK): The column key
            
        Returns:
            str: The internal column string
        """
        with self._rlock:
            return self._get_internal_dataframe_column_name(column_key)
        
    def _get_internal_dataframe_column_names(self, column_key: CK|Sequence[CK]|None = None) -> list[str]:
        """
        Internal: Get the internal dataframe column strings for a list of column keys (no lock).
        """
        if column_key is None:
            column_key = self._column_keys
        if isinstance(column_key, Sequence):
            return [self._get_internal_dataframe_column_name(column_key) for column_key in column_key] # type: ignore
        else:
            return [self._get_internal_dataframe_column_name(column_key)]

    def get_internal_dataframe_column_names(self, column_key: CK|Sequence[CK]|None = None) -> list[str]:
        """
        Public: Get the internal dataframe column strings for a list of column keys (with lock).
        """
        with self._rlock:
            return self._get_internal_dataframe_column_names(column_key)

    @staticmethod
    def column_key_to_string(column_key: CK, internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter, column_unit: Optional[Unit]) -> str:
        """
        Public: Convert a column key to string. (with lock)
        """
        return internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_unit)
    
    def _is_compatible_with_column(self, column_key: CK, value: SCALAR_TYPE|ARRAY_TYPE) -> bool:
        """
        Internal: Check if a value is compatible with a column (no lock).

        This method checks if the given value(s) can be set in the column.
        It also checks if the unit is compatible with the column unit (if the column has a unit).

        Args:
            column_key (CK): The column key to check compatibility with
            value: The value to check
            
        Returns:
            bool: True if the value is compatible with the column type
        """
        column_type: ColumnType = self._column_types[column_key]
        if column_type.has_unit:
            unit: Optional[Unit] = self._column_units[column_key]
            if unit is None:
                return False
            if not isinstance(value, United):
                return False
            value_dimension: BaseDimension[Any, Any] = value.dimension # type: ignore
            if not unit.dimension == value_dimension:
                return False
        return column_type.check_compatibility(value)

    def is_compatible_with_column(self, column_key: CK, value: SCALAR_TYPE|ARRAY_TYPE) -> bool:
        """
        Public: Check if a value is compatible with a column (with lock).
        
        Args:
            column_key (CK): The column key to check compatibility with
            value: The value to check
            
        Returns:
            bool: True if the value is compatible with the column type
        """
        with self._rlock:
            return self._is_compatible_with_column(column_key, value)

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
            return [column_key for column_key in self._column_keys if self._colkey_is_numeric(column_key)]
        
    def __repr__(self) -> str:
        """
        Return a string representation of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.__repr__()
        
    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the dataframe for Jupyter notebook display.
        
        The internal dataframe already contains unit information in column names,
        so we can directly expose the pandas HTML representation.
        """
        with self._rlock:
            return self._internal_dataframe._repr_html_() # type: ignore
        
    def to_html(self, **kwargs: Any) -> str:
        """
        Return an HTML representation of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.to_html(**kwargs) # type: ignore
        
    def __contains__(self, item: Any) -> bool:
        """
        Check if the dataframe contains a column key.
        """
        with self._rlock:
            return item in self._column_keys
        
    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.shape
        
    @property
    def dtypes(self) -> pd.Series: # type: ignore
        """
        Return the dtypes of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.dtypes # type: ignore
        
    @property
    def index(self) -> pd.Index: # type: ignore[reportUnknownReturnType]
        """
        Return the index of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.index # type: ignore
        
    @property
    def columns(self) -> pd.Index: # type: ignore[reportUnknownReturnType]
        """
        Return the columns of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.columns # type: ignore

    @property
    def values(self) -> np.ndarray:
        """
        Return the values of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.values
        
    def head(self, n: int = 5) -> "UnitedDataframe[CK]":
        """
        Return the first n rows of the dataframe.
        """
        with self._rlock:
            return self._create_with_replaced_dataframe(self._internal_dataframe.head(n))
        
    def tail(self, n: int = 5) -> "UnitedDataframe[CK]":
        """
        Return the last n rows of the dataframe.
        """
        with self._rlock:
            return self._create_with_replaced_dataframe(self._internal_dataframe.tail(n))