"""
AccessorMixin for UnitedDataframe magic methods.

This mixin provides the core access patterns for the dataframe including
iteration, indexing, and item assignment through magic methods.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING
from .dataframe_protocol import UnitedDataframeProtocol, CK

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe


class DataframeMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Mixin providing dataframe methods.
    
    This mixin implements:
    - contains_nan: Check if the dataframe contains any missing values
    - contains_inf: Check if the dataframe contains any infinite values
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    def dataframe_contains_nan(self) -> bool:
        """
        Check if the dataframe contains any missing values
        """
        return self._internal_dataframe.isna().any().any() # type: ignore
    
    def dataframe_contains_inf(self) -> bool:
        """
        Check if the dataframe contains any infinite values
        """
        return self._internal_dataframe.isin([np.inf, -np.inf]).any().any() # type: ignore