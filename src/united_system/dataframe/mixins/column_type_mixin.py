"""
Column type operations mixin for UnitedDataframe.

Contains all operations related to column types, including retrieval,
and column type management.
"""

from typing import Generic, TypeVar, overload

from ..column_type import ColumnType

CK = TypeVar("CK", bound=str, default=str)

class ColumnTypeMixin(Generic[CK]):
    """
    Column type operations mixin for UnitedDataframe.
    
    Provides all functionality related to column types, including retrieval
    and column type management.
    """

    # ----------- Retrievals: Column types ------------

    def column_type(self, column_key: CK) -> ColumnType:
        with self._rlock:
            return self._column_types[column_key]

    @overload
    def column_types(self, column_keys: CK) -> ColumnType:
        ...

    @overload
    def column_types(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[ColumnType]:
        ...

    @overload
    def column_types(self, column_keys: list[CK]) -> list[ColumnType]:
        ...

    @overload
    def column_types(self, column_keys: set[CK]) -> set[ColumnType]:
        ...

    def column_types(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> ColumnType|list[ColumnType]|set[ColumnType]:
        """
        Get the value type(s) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the value type(s) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the value type(s) of.
            
        Returns:
            Value_Type|list[Value_Type]|set[Value_Type]: The value type(s) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case str():
                    if len(more_column_keys) == 0:
                        return self._column_types[column_keys]
                    else:
                        return [self._column_types[column_keys]] + [self._column_types[more_column_key] for more_column_key in more_column_keys]
                case list():
                    column_types_as_list: list[ColumnType] = []
                    for column_key in column_keys:
                        column_types_as_list.append(self._column_types[column_key])
                    return column_types_as_list
                case set():
                    column_types_as_set: set[ColumnType] = set()
                    for column_key in column_keys:
                        column_types_as_set.add(self._column_types[column_key])
                    return column_types_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def column_type_dict(self) -> dict[CK, ColumnType]:
        """
        Get a dictionary mapping column keys to their value types.
        
        Returns:
            dict[CK, Value_Type]: Dictionary mapping column keys to value types
        """
        with self._rlock:
            return self._column_types.copy() 