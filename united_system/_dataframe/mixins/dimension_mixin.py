"""
Dimension operations mixin for UnitedDataframe.

Contains all operations related to dimensions, including retrieval,
setting, and dimension management.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, Optional
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._units_and_dimension.dimension import Dimension
from ..._units_and_dimension.unit import Unit

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class DimensionMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Dimension operations mixin for UnitedDataframe.
    
    Provides all functionality related to dimensions, including retrieval,
    setting, and dimension management.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Retrievals: Dimensions ------------

    @property
    def dimensions(self) -> Mapping[CK, Optional[Dimension]]:
        """
        Get a copy of all dimensions.
        
        Returns:
            Mapping[CK, Optional[Dimension]]: A copy of the dictionary of dimensions
        """
        with self._rlock:
            dimensions: Mapping[CK, Optional[Dimension]] = {}
            for column_key in self._column_keys:
                unit: Unit|None = self._column_units[column_key]
                if unit is not None:
                    dimensions[column_key] = unit.dimension
                else:
                    dimensions[column_key] = None
            return dimensions
        
    def dim_get_dimension(self, column_key: CK) -> Dimension:
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
        
    def dim_has_dimension(self, column_key: CK) -> bool:
        """
        Check if a column has a dimension.
        """
        with self._rlock:
            return self._dim_has_dimension(column_key)
        
    def _dim_has_dimension(self, column_key: CK) -> bool:
        """
        Internal: Check if a column has a dimension. (no lock, no read-only check)
        """
        return self._column_units[column_key] is not None