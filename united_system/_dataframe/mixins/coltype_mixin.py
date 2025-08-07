"""
Column type operations mixin for UnitedDataframe.

Contains all operations related to column types, including retrieval,
filtering by type, and column type management.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, TypeVar
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._dataframe.column_type import ColumnType

CK_CF = TypeVar("CK_CF", bound=str)

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class ColTypeMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Column type operations mixin for UnitedDataframe.
    
    Provides all functionality related to column types, including retrieval,
    filtering by type, and column type management.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Column types ------------

    @property
    def coltypes(self) -> dict[CK, ColumnType]:
        """
        Get a copy of all column types.
        
        Returns:
            dict[CK, ColumnType]: A copy of the dictionary of column types
        """
        with self._rlock:  # Full IDE support!
            return self._column_types.copy()  # Protocol knows _column_types exists!
        
    def coltype_get(self, column_key: CK) -> ColumnType:
        """
        Get the column type for a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            ColumnType: The column type
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            return self._column_types[column_key]

    def column_keys_of_column_type(self, *column_types: ColumnType) -> list[CK]:
        """
        Get the column keys of a given column type.
        """
        with self._rlock:
            return [column_key for column_key in self._column_keys if self._column_types[column_key] in column_types]

    # ----------- Setters: Column types ------------

    def set_column_type(self, column_key: CK, column_type: ColumnType):
        """
        Set the column type for a column.
        
        Args:
            column_key (CK): The column key
            column_type (ColumnType): The new column type
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            self._column_types[column_key] = column_type 