"""
Mixins package for UnitedDataframe.

This package contains all the mixins that compose the UnitedDataframe class.
Each mixin provides a specific set of functionality, and together they create
the complete UnitedDataframe interface.

All mixins now inherit from UnitedDataframeProtocol for full IDE support and type checking.
The UnitedDataframeProtocol defines the exact interface that mixins can expect.

Mixins included:
- CoreMixin: Basic properties, initialization, locks, utility methods
- ColumnKeyMixin: Column key operations and metadata
- ColTypeMixin: Column type management
- DimensionMixin: Dimension operations and internal column strings
- ColumnOperationsMixin: Column CRUD operations
- ColumnStatisticsMixin: All column statistics methods
- RowOperationsMixin: Row CRUD operations
- CellOperationsMixin: Individual cell operations
- MaskOperationsMixin: Boolean masking operations
- RowAccessMixin: Row access patterns (head, tail, slice)
- FilterMixin: Advanced filtering operations
- SerializationMixin: JSON/HDF5/CSV/Pickle serialization
- ConstructorMixin: Class factory methods
- UnitMixin: Unit operations
- GroupbyMixin: Complete groupby functionality
- AccessorGetitemMixin: Magic methods (__getitem__, __iter__)
- AccessorSetitemMixin: Magic methods (__setitem__)
- SegmentMixin: Complete segment functionality
"""

from .dataframe_protocol import UnitedDataframeProtocol, CK
from .core_mixin import CoreMixin
from .colkey_mixin import ColKeyMixin
from .coltype_mixin import ColTypeMixin
from .dimension_mixin import DimensionMixin
from .column_access_mixin import ColumnAccessMixin
from .column_operations_mixin import ColumnOperationsMixin
from .column_statistics_mixin import ColumnStatisticsMixin
from .row_access_mixin import RowAccessMixin
from .row_operations_mixin import RowOperationsMixin
from .cell_operations_mixin import CellOperationsMixin
from .mask_operations_mixin import MaskOperationsMixin
from .filter_mixin import FilterMixin
from .serialization_mixin import SerializationMixin
from .constructor_mixin import ConstructorMixin
from .group_mixin import GroupbyMixin
from .accessor_getitem_mixin import AccessorGetitemMixin
from .accessor_setitem_mixin import AccessorSetitemMixin
from .iter_mixin import IterMixin
from .row_statistics_mixin import RowStatisticsMixin
from .unit_mixin import UnitMixin
from .segment_mixin import SegmentMixin

__all__ = [
    # Protocol
    "UnitedDataframeProtocol",
    "CK",
    
    # All mixins with full IDE support
    "CoreMixin",
    "ColKeyMixin", 
    "ColTypeMixin",
    "UnitMixin",
    "DimensionMixin",
    "ColumnAccessMixin",
    "ColumnOperationsMixin",
    "ColumnStatisticsMixin", 
    "RowAccessMixin",
    "RowOperationsMixin",
    "CellOperationsMixin",
    "MaskOperationsMixin",
    "SegmentMixin",
    "RowAccessMixin",
    "FilterMixin", 
    "SerializationMixin",
    "ConstructorMixin",
    "GroupbyMixin",
    "AccessorGetitemMixin",
    "AccessorSetitemMixin",
    "IterMixin",
    "RowStatisticsMixin",
]