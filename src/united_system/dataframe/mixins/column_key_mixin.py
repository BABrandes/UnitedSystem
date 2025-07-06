"""
Column key operations mixin for UnitedDataframe.

Contains all operations related to column keys, including retrieval,
filtering by type, and column key management.
"""

from typing import Generic, TypeVar

from ..column_information import ColumnKey, ColumnInformation

CK = TypeVar("CK", bound=ColumnKey|str, default=str)
CK_CF = TypeVar("CK_CF", bound=ColumnKey|str, default=str)

class ColumnKeyMixin(Generic[CK]):
    """
    Column key operations mixin for UnitedDataframe.
    
    Provides all functionality related to column keys, including retrieval,
    filtering by type, and column key management.
    """

    # ----------- Retrievals: Column keys ------------

    @property
    def column_keys(self) -> list[CK]:
        """
        Get a copy of all column keys.
        
        Returns:
            list[CK]: A copy of the list of column keys
        """
        with self._rlock:
            return self._column_keys.copy()
        
    def has_column(self, column_key: CK) -> bool:
        """
        Check if a column exists by index or column key.
        
        Args:
            column_key (CK): The column key to check
            
        Returns:
            bool: True if the column exists, False otherwise
        """
        with self._rlock:
            return column_key in self._column_keys

    def column_keys_of_type(self, *column_key_types: type[CK_CF]) -> list[CK_CF]:
        """
        Get the column keys of a given type.
        """
        with self._rlock:
            column_keys_to_keep_filtered_type: list[CK_CF] = []
            for column_key in self._column_keys:
                if isinstance(column_key, tuple(column_key_types)):
                    column_keys_to_keep_filtered_type.append(column_key)
            return column_keys_to_keep_filtered_type

    # ----------- Column Information ------------

    def get_column_information_dict(self) -> dict[CK, ColumnInformation[CK]]:
        """
        Get the column information list.
        """
        with self._rlock:
            return {column_key: ColumnInformation[CK](column_key, self._dimensions[column_key], self._column_types[column_key], self._display_units[column_key]) for column_key in self._column_keys}

    def column_information_of_type(self, *column_key_types: type[CK_CF]) -> list[tuple[CK_CF, ColumnInformation[CK_CF]]]:
        """
        Filter the dataframe by column key type.
        """
        with self._rlock:
            column_information_of_type: list[ColumnInformation[CK_CF]] = []
            for column_key in self._column_keys:
                column_key_filtered_type: CK = column_key
                if isinstance(column_key_filtered_type, tuple(column_key_types)):
                    column_information_of_type.append(ColumnInformation[CK_CF](
                        column_key_filtered_type,
                        self._dimensions[column_key],
                        self._column_types[column_key],
                        self._display_units[column_key]))
            return [(column_key, column_information) for column_key, column_information in column_information_of_type]

    # ----------- Setters: Column names ------------

    def rename_column(self, current_column_key: CK, new_column_key: CK):
        """
        Set the column key for a column at the specified index.
        
        Args:
            current_column_key (CK): The current column key
            new_column_key (CK): The new column key
            
        Raises:
            ValueError: If the dataframe is read-only, the name already exists,
                                       or the new column name conflicts with existing columns
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(current_column_key):
                raise ValueError(f"Column key {current_column_key} does not exist in the dataframe.")
            if self.has_column(new_column_key):
                raise ValueError(f"Column key {new_column_key} already exists in the dataframe.")
            current_internal_dataframe_column_name: str = self.internal_dataframe_column_string(current_column_key)
            new_internal_dataframe_column_name: str = self.create_internal_dataframe_column_name(new_column_key)
            if new_internal_dataframe_column_name in self._internal_canonical_dataframe.columns:
                raise ValueError(f"Column name {new_internal_dataframe_column_name} already exists in the dataframe.")
            self._column_information[new_column_key] = self._column_information.pop(current_column_key)
            self._internal_canonical_dataframe.rename(columns={current_internal_dataframe_column_name: new_internal_dataframe_column_name}, inplace=True)
            self._column_keys[self._column_keys.index(current_column_key)] = new_column_key
            self._internal_dataframe_column_strings.pop(current_column_key)
            self._internal_dataframe_column_strings[new_column_key] = new_internal_dataframe_column_name
            self._dimensions[new_column_key] = self._dimensions.pop(current_column_key)
            self._column_types[new_column_key] = self._column_types.pop(current_column_key)
            self._display_units[new_column_key] = self._display_units.pop(current_column_key) 