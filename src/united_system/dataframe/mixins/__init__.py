"""
Mixins package for UnitedDataframe.

This package contains all the mixins that compose the UnitedDataframe class.
Each mixin provides a specific set of functionality, and together they create
the complete UnitedDataframe interface.

All mixins now inherit from UnitedDataframeMixin for full IDE support and type checking.
The UnitedDataframeProtocol defines the exact interface that mixins can expect.

Mixins included:
- CoreMixin: Basic properties, initialization, locks, utility methods
- ColumnKeyMixin: Column key operations and metadata
- ColumnTypeMixin: Column type management
- DisplayUnitMixin: Display unit operations
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
- GroupbyMixin: Complete groupby functionality
- AccessorMixin: Magic methods (__getitem__, __setitem__, __iter__)
"""

from .dataframe_protocol import UnitedDataframeProtocol, UnitedDataframeMixin, CK
from .core_mixin import CoreMixin
from .column_key_mixin import ColumnKeyMixin
from .column_type_mixin import ColumnTypeMixin
from .display_unit_mixin import DisplayUnitMixin
from .dimension_mixin import DimensionMixin
from .column_operations_mixin import ColumnOperationsMixin
from .column_statistics_mixin import ColumnStatisticsMixin
from .row_operations_mixin import RowOperationsMixin
from .cell_operations_mixin import CellOperationsMixin
from .mask_operations_mixin import MaskOperationsMixin
from .row_access_mixin import RowAccessMixin
from .filter_mixin import FilterMixin
from .serialization_mixin import SerializationMixin
from .constructor_mixin import ConstructorMixin
from .groupby_mixin import GroupbyMixin
from .accessor_mixin import AccessorMixin

__all__ = [
    # Protocol and base mixin
    "UnitedDataframeProtocol",
    "UnitedDataframeMixin", 
    "CK",
    
    # All mixins with full IDE support
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
    "AccessorMixin",
]