"""
Display unit operations mixin for UnitedDataframe.

Contains all operations related to display units, including retrieval,
setting, and display unit management.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Union
from .dataframe_protocol import UnitedDataframeMixin, CK
from ...units.united import United
from ...units.base_classes.base_unit import BaseUnit

class DisplayUnitMixin(UnitedDataframeMixin[CK]):
    """
    Display unit operations mixin for UnitedDataframe.
    
    Provides all functionality related to display units, including retrieval,
    setting, and display unit management.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Display units ------------

    @property
    def display_units(self) -> dict[CK, United]:
        """
        Get a copy of all display units.
        
        Returns:
            dict[CK, United]: A copy of the dictionary of display units
        """
        with self._rlock:  # Full IDE support!
            return self._display_units.copy()  # Protocol knows _display_units exists!
        
    def get_display_unit(self, column_key: CK) -> United:
        """
        Get the display unit for a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            United: The display unit
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            return self._display_units[column_key]

    # ----------- Setters: Display units ------------

    def set_display_unit(self, column_key: CK, display_unit: Union[United, BaseUnit]):
        """
        Set the display unit for a column.
        
        Args:
            column_key (CK): The column key
            display_unit (United|BaseUnit): The new display unit
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Convert BaseUnit to United if needed
            if isinstance(display_unit, BaseUnit):
                display_unit = United(display_unit)
            
            self._display_units[column_key] = display_unit 