"""
UnitedDataframe package.

This package provides a comprehensive dataframe implementation with full units support.
The main class is UnitedDataframe, which combines functionality from multiple mixins
to provide a complete dataframe solution.
"""

from ..united_dataframe import UnitedDataframe
from .column_information import ColumnInformation, InternalDataFrameNameFormatter
from .column_key import ColumnKey
from .column_type import ColumnType

# Import all mixins for advanced usage
from .mixins import (
    CoreMixin,
    ColumnKeyMixin,
    ColumnTypeMixin,
    DisplayUnitMixin,
    DimensionMixin,
    ColumnOperationsMixin,
    ColumnStatisticsMixin,
    RowOperationsMixin,
    CellOperationsMixin,
    MaskOperationsMixin,
    RowAccessMixin,
    FilterMixin,
    SerializationMixin,
    ConstructorMixin,
    GroupbyMixin,
)

__all__ = [
    # Main class
    "UnitedDataframe",
    
    # Supporting classes
    "ColumnInformation",
    "InternalDataFrameNameFormatter", 
    "ColumnKey",
    "ColumnType",
    
    # Mixins (for advanced usage)
    "CoreMixin",
    "ColumnKeyMixin",
    "ColumnTypeMixin", 
    "DisplayUnitMixin",
    "DimensionMixin",
    "ColumnOperationsMixin",
    "ColumnStatisticsMixin",
    "RowOperationsMixin",
    "CellOperationsMixin",
    "MaskOperationsMixin",
    "RowAccessMixin",
    "FilterMixin",
    "SerializationMixin",
    "ConstructorMixin",
    "GroupbyMixin",
]

# Version information
__version__ = "1.0.0"
__author__ = "UnitedSystem Development Team" 