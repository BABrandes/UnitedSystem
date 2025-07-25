"""
Display unit operations mixin for UnitedDataframe.

Contains all operations related to display units, including retrieval,
setting, and display unit management.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""
from typing import TYPE_CHECKING
import numpy as np
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ....unit import Unit

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class UnitMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Unit operations mixin for UnitedDataframe.
    
    Provides all functionality related to units, including retrieval,
    setting, and unit management.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Units ------------

    @property
    def units(self) -> dict[CK, Unit|None]:
        """
        Get a copy of all units.
        
        Returns:
            dict[CK, Unit|None]: A copy of the dictionary of display units
        """
        with self._rlock:
            return self._column_units.copy()
        
    def _unit_has(self, column_key: CK) -> bool:
        """
        Internal: Check if a column has a unit (no lock).
        """
        return self._column_types[column_key].has_unit

    def _unit_get(self, column_key: CK) -> Unit:
        """
        Internal: Get the unit of a column (no lock).
        """
        unit: Unit|None = self._column_units[column_key]
        if unit is None:
            raise ValueError(f"Column {column_key} has no unit.")
        return unit
    
    def unit_has_unit(self, column_key: CK) -> bool:
        """
        Check if a column has a unit.
        """
        with self._rlock:
            return self._unit_has(column_key)

    def unit_get_unit(self, column_key: CK) -> Unit:
        """
        Get the unit for a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            Unit|None: The display unit
        """
        with self._rlock:
            return self._unit_get(column_key)

    # ----------- Setters: Units ------------

    def unit_change_unit(self, column_key: CK, unit: Unit):
        """
        Changes the unit for a column, but not the dimension.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._unit_change(column_key, unit)

    def _unit_change(self, column_key: CK, unit: Unit):
        """
        Changes the unit for a column, but not the dimension.
        
        Args:
            column_key (CK): The column key
            unit (Unit): The new unit
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """

        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        old_unit: Unit|None = self._column_units[column_key]
        if old_unit is None:
            raise ValueError(f"Column key {column_key} has no unit.")
        if not unit.compatible_to(old_unit):    
            raise ValueError(f"Unit {unit} is not compatible with the current unit {old_unit}.")
        
        dataframe_column_name: str = self._internal_dataframe_column_names[column_key]
        numpy_array_in_old_unit: np.ndarray = self._internal_dataframe[dataframe_column_name].to_numpy() # type: ignore
        numpy_array_in_new_unit: np.ndarray = Unit.convert(numpy_array_in_old_unit, old_unit, unit) # type: ignore
        self._internal_dataframe[dataframe_column_name] = numpy_array_in_new_unit
        self._column_units[column_key] = unit