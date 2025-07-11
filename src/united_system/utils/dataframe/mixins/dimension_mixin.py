"""
Dimension operations mixin for UnitedDataframe.

Contains all operations related to dimensions, including retrieval,
setting, and dimension management.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import Optional
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ....dimension import Dimension
from ....unit import Unit

class DimensionMixin(UnitedDataframeProtocol[CK]):
    """
    Dimension operations mixin for UnitedDataframe.
    
    Provides all functionality related to dimensions, including retrieval,
    setting, and dimension management.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Dimensions ------------

    @property
    def dimensions(self) -> dict[CK, Optional[Dimension]]:
        """
        Get a copy of all dimensions.
        
        Returns:
            dict[CK, Optional[Dimension]]: A copy of the dictionary of dimensions
        """
        with self._rlock:
            dimensions: dict[CK, Optional[Dimension]] = {}
            for column_key in self._column_keys:
                unit: Unit|None = self._column_units[column_key]
                if unit is not None:
                    dimensions[column_key] = unit.dimension
                else:
                    dimensions[column_key] = None
            return dimensions
        
    def get_dimension(self, column_key: CK) -> Dimension:
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
            unit: Unit|None = self._column_units[column_key]
            if unit is None:
                raise ValueError(f"Column key {column_key} does not have a unit.")
            return unit.dimension

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
            return self._internal_dataframe_column_names[column_key] 