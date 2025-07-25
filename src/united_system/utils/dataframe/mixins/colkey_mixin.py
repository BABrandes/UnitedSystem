"""
Column key operations mixin for UnitedDataframe.

Contains all operations related to column keys, including retrieval,
filtering by type, and column key management.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING
from .dataframe_protocol import UnitedDataframeProtocol, CK, CK_CF

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class ColKeyMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Column key operations mixin for UnitedDataframe.
    
    Provides all functionality related to column keys, including retrieval,
    filtering by type, and column key management.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Column keys ------------

    @property
    def colkeys(self) -> list[CK]:
        """
        Get a copy of all column keys.
        
        Returns:
            list[CK]: A copy of the list of column keys
        """
        with self._rlock:  # Full IDE support!
            return self._column_keys.copy()  # Protocol knows _column_keys exists!
        
    def _colkey_exists(self, column_key: CK) -> bool:
        """
        Internal: Check if a column exists by column key (no lock).
        """
        return column_key in self._column_keys

    def colkey_exists(self, column_key: CK) -> bool:
        """
        Public: Check if a column exists by index or column key (with lock).
        
        Args:
            column_key (CK): The column key to check
            
        Returns:
            bool: True if the column exists, False otherwise
        """
        with self._rlock:
            return self._colkey_exists(column_key)

    def _colkey_get_of_type(self, *column_key_types: type[CK_CF]) -> list[CK_CF]:
        """
        Internal: Get the column keys of a given type. (no lock)
        """

        column_keys_to_keep_filtered_type: list[CK_CF] = []
        for column_key in self._column_keys:
            if isinstance(column_key, tuple(column_key_types)):
                column_keys_to_keep_filtered_type.append(column_key)
        return column_keys_to_keep_filtered_type

    def colkey_get_of_type(self, *column_key_types: type[CK_CF]) -> list[CK_CF]:
        """
        Get the column keys of a given type.
        """
        with self._rlock:
            return self._colkey_get_of_type(*column_key_types)
        
    def _colkey_get_numeric(self) -> list[CK]:
        """
        Internal: Get the numeric column keys (no lock).
        """
        return [column_key for column_key in self._column_keys if self._colkey_is_numeric(column_key)]
    
    def colkey_get_numeric(self) -> list[CK]:
        """
        Get the numeric column keys.
        """
        with self._rlock:
            return self._colkey_get_numeric()
        
    def _colkey_is_numeric(self, column_key: CK) -> bool:
        """
        Internal: Check if a column is numeric (no lock).
        """
        return self._column_types[column_key].is_numeric

    def colkey_is_numeric(self, column_key: CK) -> bool:
        """
        Public: Check if a column is numeric (with lock).
        """
        with self._rlock:
            return self._colkey_is_numeric(column_key)