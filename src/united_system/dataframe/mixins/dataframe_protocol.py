"""
Core protocol/interface for UnitedDataframe mixins.

This protocol defines the exact interface that all mixins can rely on,
solving the "blind mixins" problem where mixins can't see what methods
and attributes are available.
"""

from typing import Protocol, TypeVar, Generic, Dict, List, runtime_checkable
from typing import TYPE_CHECKING
import pandas as pd
from readerwriterlock import rwlock

if TYPE_CHECKING:
    from ..column_information import ColumnInformation, InternalDataFrameNameFormatter
    from ..column_key import ColumnKey
    from ..column_type import ColumnType
    from ...unit import Unit
    from ...dimension import Dimension
    from ...united_dataframe import UnitedDataframe

CK = TypeVar("CK", bound="ColumnKey | str")

@runtime_checkable
class UnitedDataframeProtocol(Protocol, Generic[CK]):
    """
    Protocol defining the core interface that all UnitedDataframe mixins can rely on.
    
    This makes it explicit what attributes and methods are available to mixins,
    enabling proper IDE support, type checking, and making dependencies clear.
    """
    
    # ===== CORE DATA STRUCTURES =====
    _internal_canonical_dataframe: pd.DataFrame
    _column_information: Dict[CK, "ColumnInformation"]
    _internal_dataframe_column_name_formatter: "InternalDataFrameNameFormatter"
    
    # ===== DERIVED DATA STRUCTURES =====
    _column_keys: List[CK]
    _column_types: Dict[CK, "ColumnType"]
    _display_units: Dict[CK, "Unit"]
    _dimensions: Dict[CK, "Dimension"]
    _internal_dataframe_column_strings: Dict[CK, str]
    
    # ===== STATE =====
    _read_only: bool
    
    # ===== THREAD SAFETY =====
    _lock: rwlock.RWLockFairD
    _rlock: rwlock.RWLockFairD._aReader
    _wlock: rwlock.RWLockFairD._aWriter
    
    # ===== CORE METHODS THAT MIXINS CAN CALL =====
    
    def __len__(self) -> int:
        """Return the number of rows in the dataframe."""
        ...
    
    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the dataframe as (rows, columns)."""
        ...
    
    @property
    def empty(self) -> bool:
        """Check if the dataframe is empty."""
        ...
    
    def has_column(self, column_key: CK) -> bool:
        """Check if a column exists by column key."""
        ...
    
    def column_type(self, column_key: CK) -> "ColumnType":
        """Get the column type for a column."""
        ...
    
    def display_unit(self, column_key: CK) -> "Unit":
        """Get the display unit for a column."""
        ...
    
    def dimension(self, column_key: CK) -> "Dimension":
        """Get the dimension for a column."""
        ...
    
    def create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """Create the internal dataframe column name for a column."""
        ...
    
    def compatible_with_column(self, column_key: CK, value) -> bool:
        """Check if a value is compatible with a column."""
        ...
    
    def copy(self, deep: bool = True) -> "UnitedDataframe[CK]":
        """Create a copy of the dataframe."""
        ...
    
    # ===== CROSS-MIXIN DEPENDENCIES =====
    # These methods are provided by other mixins but commonly used
    
    def cell_get_value(self, row_index: int, column_key: CK):
        """Get a single cell value (from CellOperationsMixin)."""
        ...
    
    def cell_set_value(self, row_index: int, column_key: CK, value):
        """Set a single cell value (from CellOperationsMixin)."""
        ...
    
    def remove_column(self, column_key: CK) -> None:
        """Remove a column (from ColumnOperationsMixin)."""
        ...
    
    def rows_remove_by_indices(self, row_indices: List[int]) -> None:
        """Remove rows by indices (from RowOperationsMixin)."""
        ...


# ===== BASE MIXIN CLASS =====

class UnitedDataframeMixin(Generic[CK]):
    """
    Base class for all UnitedDataframe mixins.
    
    Provides type hints so that mixins can access the UnitedDataframe interface
    with full IDE support and type checking.
    """
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure all mixins properly declare their protocol compliance
        if not hasattr(cls, '__annotations__'):
            cls.__annotations__ = {}
    
    # Explicit type annotations to help IDE understand the protocol
    if TYPE_CHECKING:
        # Core data structures - these tell IDE what attributes are available
        _internal_canonical_dataframe: pd.DataFrame
        _column_information: Dict[CK, "ColumnInformation"]
        _internal_dataframe_column_name_formatter: "InternalDataFrameNameFormatter"
        
        # Derived data structures
        _column_keys: List[CK]
        _column_types: Dict[CK, "ColumnType"]
        _display_units: Dict[CK, "Unit"]
        _dimensions: Dict[CK, "Dimension"]
        _internal_dataframe_column_strings: Dict[CK, str]
        
        # State
        _read_only: bool
        
        # Thread safety
        _lock: rwlock.RWLockFairD
        _rlock: rwlock.RWLockFairD._aReader
        _wlock: rwlock.RWLockFairD._aWriter
        
        def __init__(self: UnitedDataframeProtocol[CK]) -> None:
            ...
    
    @property
    def _self(self) -> UnitedDataframeProtocol[CK]:
        """
        Helper property that provides typed access to self.
        
        Usage in mixins:
            def some_method(self):
                # Now you get full autocomplete and type checking!
                with self._self._rlock:
                    return len(self._self._internal_canonical_dataframe)
        """
        return self  # type: ignore 