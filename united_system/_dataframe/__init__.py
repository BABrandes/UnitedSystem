"""
UnitedDataframe package.

This package provides a comprehensive dataframe implementation with full units support.
The main class is UnitedDataframe, which combines functionality from multiple mixins
to provide a complete dataframe solution.
"""

from typing import TYPE_CHECKING

from .internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter
from .column_key import ColumnKey
from .column_type import ColumnType

# Import all mixins for advanced usage
from .mixins import (
    CoreMixin,
    ColKeyMixin,
    ColTypeMixin,
    UnitMixin,
    DimensionMixin,
    ColumnAccessMixin,
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

if TYPE_CHECKING:
    from .united_dataframe import UnitedDataframe

__all__ = [
    # Main class
    "UnitedDataframe",
    
    # Supporting classes
    "InternalDataFrameColumnNameFormatter", 
    "ColumnKey",
    "ColumnType",
    
    # Mixins (for advanced usage)
    "CoreMixin",
    "ColKeyMixin",
    "ColTypeMixin", 
    "UnitMixin",
    "DimensionMixin",
    "ColumnAccessMixin",
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
__version__ = "0.1.0"
__author__ = "Benedikt Axel Brandes" 