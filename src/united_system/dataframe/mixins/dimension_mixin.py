"""
Dimension operations mixin for UnitedDataframe.

Contains all operations related to dimensions, including retrieval,
setting, and dimension management.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Union
from .dataframe_protocol import UnitedDataframeMixin, CK
from ...units.base_classes.base_dimension import BaseDimension

class DimensionMixin(UnitedDataframeMixin[CK]):
    """
    Dimension operations mixin for UnitedDataframe.
    
    Provides all functionality related to dimensions, including retrieval,
    setting, and dimension management.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Dimensions ------------

    @property
    def dimensions(self) -> dict[CK, BaseDimension]:
        """
        Get a copy of all dimensions.
        
        Returns:
            dict[CK, BaseDimension]: A copy of the dictionary of dimensions
        """
        with self._rlock:  # Full IDE support!
            return self._dimensions.copy()  # Protocol knows _dimensions exists!
        
    def get_dimension(self, column_key: CK) -> BaseDimension:
        """
        Get the dimension for a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BaseDimension: The dimension
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            return self._dimensions[column_key]

    # ----------- Setters: Dimensions ------------

    def set_dimension(self, column_key: CK, dimension: BaseDimension):
        """
        Set the dimension for a column.
        
        Args:
            column_key (CK): The column key
            dimension (BaseDimension): The new dimension
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            self._dimensions[column_key] = dimension

    # ----------- Internal dataframe column string operations ------------

    def internal_dataframe_column_string(self, column_key: CK) -> str:
        """
        Get the internal dataframe column string for a column key.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            str: The internal column string
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            return self._internal_dataframe_column_strings[column_key]

    def create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Create an internal dataframe column name for a column key.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            str: The internal column name
        """
        with self._rlock:
            return self._internal_dataframe_name_formatter(column_key)  # Protocol knows this exists! 