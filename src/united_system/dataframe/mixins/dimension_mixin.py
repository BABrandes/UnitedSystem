"""
Dimension operations mixin for UnitedDataframe.

Contains all operations related to dimensions, including retrieval
and dimension management.
"""

from typing import Generic, TypeVar, overload

from ...dimension import Dimension

CK = TypeVar("CK", bound=str, default=str)

class DimensionMixin(Generic[CK]):
    """
    Dimension operations mixin for UnitedDataframe.
    
    Provides all functionality related to dimensions, including retrieval
    and dimension management.
    """

    # ----------- Retrievals: UnitQuantity ------------

    def dimension(self, column_key: CK) -> Dimension:
        with self._rlock:
            return self._dimensions[column_key]

    @overload
    def dimensions(self, column_keys: CK) -> Dimension:
        ...

    @overload
    def dimensions(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[Dimension]:
        ...

    @overload
    def dimensions(self, column_keys: list[CK]) -> list[Dimension]:
        ...

    @overload
    def dimensions(self, column_keys: set[CK]) -> set[Dimension]:
        ...

    def dimensions(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> Dimension|list[Dimension]|set[Dimension]:
        """
        Get the dimension(s) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the dimension(s) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the value type(s) of.
            
        Returns:
            Dimension|list[Dimension]|set[Dimension]: The dimension(s) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case str():
                    if len(more_column_keys) == 0:
                        return self._dimensions[column_keys]
                    else:
                        return [self._dimensions[column_keys]] + [self._dimensions[more_column_key] for more_column_key in more_column_keys]
                case list():
                    dimensions_as_list: list[Dimension] = []
                    for column_key in column_keys:
                        dimensions_as_list.append(self._dimensions[column_key])
                    return dimensions_as_list
                case set():
                    dimensions_as_set: set[Dimension] = set()
                    for column_key in column_keys:
                        dimensions_as_set.add(self._dimensions[column_key])
                    return dimensions_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def dimensions_dict(self) -> dict[CK, Dimension]:
        """
        Get a dictionary mapping column keys to their dimensions.
        
        Returns:
            dict[CK, Dimension]: Dictionary mapping column keys to dimensions
        """
        with self._rlock:
            return self._dimensions.copy()

    # ----------- Internal Dataframe Column Strings ------------

    def internal_dataframe_column_string(self, column_key: CK) -> str:
        with self._rlock:
            return self._internal_dataframe_column_strings[column_key]

    @overload
    def internal_dataframe_column_strings(self, column_keys: CK) -> str:
        ...

    @overload
    def internal_dataframe_column_strings(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[str]:
        ...

    @overload
    def internal_dataframe_column_strings(self, column_keys: list[CK]) -> list[str]:
        ...

    @overload
    def internal_dataframe_column_strings(self, column_keys: set[CK]) -> set[str]:
        ...

    def internal_dataframe_column_strings(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> str|list[str]|set[str]:
        """
        Get the internal dataframe column strings by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the internal dataframe column strings of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the internal dataframe column strings of.
            
        Returns:
            str|list[str]|set[str]: The internal dataframe column strings of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case str():
                    if len(more_column_keys) == 0:
                        return self._internal_dataframe_column_strings[column_keys]
                    else:
                        return [self._internal_dataframe_column_strings[column_keys]] + [self._internal_dataframe_column_strings[more_column_key] for more_column_key in more_column_keys]
                case list():
                    internal_dataframe_column_strings_as_list: list[str] = []
                    for column_key in column_keys:
                        internal_dataframe_column_strings_as_list.append(self._internal_dataframe_column_strings[column_key])
                    return internal_dataframe_column_strings_as_list
                case set():
                    internal_dataframe_column_strings_as_set: set[str] = set()
                    for column_key in column_keys:
                        internal_dataframe_column_strings_as_set.add(self._internal_dataframe_column_strings[column_key])
                    return internal_dataframe_column_strings_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def internal_dataframe_column_strings_dict(self) -> dict[CK, str]:
        """
        Get a dictionary mapping column keys to their internal dataframe column strings.
        
        Returns:
            dict[CK, str]: Dictionary mapping column keys to internal dataframe column strings
        """
        with self._rlock:
            return self._internal_dataframe_column_strings.copy() 