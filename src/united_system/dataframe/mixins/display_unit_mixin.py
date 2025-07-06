"""
Display unit operations mixin for UnitedDataframe.

Contains all operations related to display units, including retrieval
and display unit management.
"""

from typing import Generic, TypeVar, overload

from ...unit import Unit

CK = TypeVar("CK", bound=str, default=str)

class DisplayUnitMixin(Generic[CK]):
    """
    Display unit operations mixin for UnitedDataframe.
    
    Provides all functionality related to display units, including retrieval
    and display unit management.
    """

    # ----------- Retrievals: Display Units ------------

    def display_unit(self, column_key: CK) -> Unit:
        with self._rlock:
            return self._display_units[column_key]

    @overload
    def display_units(self, column_keys: CK) -> Unit:
        ... 

    @overload
    def display_units(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[Unit]:
        ...

    @overload
    def display_units(self, column_keys: list[CK]) -> list[Unit]:
        ...

    @overload
    def display_units(self, column_keys: set[CK]) -> set[Unit]:
        ...

    def display_units(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> Unit|list[Unit]|set[Unit]:
        """
        Get the display unit(s) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the display unit(s) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the display unit(s) of.
            
        Returns:
            Unit|list[Unit]|set[Unit]: The display unit(s) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case str():
                    if len(more_column_keys) == 0:
                        return self._display_units[column_keys]
                    else:
                        return [self._display_units[column_keys]] + [self._display_units[more_column_key] for more_column_key in more_column_keys]
                case list():
                    display_units_as_list: list[Unit] = []
                    for column_key in column_keys:
                        display_units_as_list.append(self._display_units[column_key])
                    return display_units_as_list
                case set():
                    display_units_as_set: set[Unit] = set()
                    for column_key in column_keys:
                        display_units_as_set.add(self._display_units[column_key])
                    return display_units_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def display_unit_dict(self) -> dict[CK, Unit]:
        """
        Get a dictionary mapping column keys to their display units.
        
        Returns:
            dict[CK, Unit]: Dictionary mapping column keys to display units
        """
        with self._rlock:
            return self._display_units.copy() 