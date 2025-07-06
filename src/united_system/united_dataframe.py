"""
Main UnitedDataframe class that combines all mixins.

This is the primary class that users will interact with. It inherits from all
the mixins to provide a complete dataframe implementation with units support.
"""

from dataclasses import dataclass
from typing import TypeVar, Generic, Dict, Any, Callable, Union
import pandas as pd
from readerwriterlock import rwlock

from .dataframe.mixins import *
from .dataframe.column_information import ColumnInformation, InternalDataFrameNameFormatter
from .dataframe.column_key import ColumnKey
from .dataframe.column_type import ColumnType
from .unit import Unit
from .dimension import Dimension

CK = TypeVar("CK", bound=Union[ColumnKey, str])

@dataclass
class UnitedDataframe(
    CoreMixin[CK],
    ColumnKeyMixin[CK],
    ColumnTypeMixin[CK],
    DisplayUnitMixin[CK],
    DimensionMixin[CK],
    ColumnOperationsMixin[CK],
    ColumnStatisticsMixin[CK],
    RowOperationsMixin[CK],
    CellOperationsMixin[CK],
    MaskOperationsMixin[CK],
    RowAccessMixin[CK],
    FilterMixin[CK],
    SerializationMixin[CK],
    ConstructorMixin[CK],
    GroupbyMixin[CK],
    AccessorMixin[CK],
    Generic[CK]
):
    """
    A dataframe implementation with full units support.
    
    UnitedDataframe combines all the functionality from various mixins to provide
    a comprehensive dataframe implementation that supports:
    - Units and dimensions for all columns
    - Type safety with proper scalar/array types
    - Thread-safe operations with read/write locks
    - Comprehensive statistical operations
    - Advanced filtering and masking
    - Serialization support (JSON, HDF5, CSV, Pickle)
    - GroupBy operations
    - Magic methods for intuitive access patterns (__iter__, __getitem__, __setitem__)
    - And much more!
    
    This class is the main entry point for users and combines all the mixins
    to provide a complete dataframe solution.
    """
    
    # Core data structures
    _internal_canonical_dataframe: pd.DataFrame
    _column_information: Dict[CK, ColumnInformation[CK]]
    _internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter
    
    # Derived data structures (populated in __post_init__)
    _column_keys: list[CK] = None
    _column_types: Dict[CK, ColumnType] = None
    _display_units: Dict[CK, Unit] = None
    _dimensions: Dict[CK, Dimension] = None
    _internal_dataframe_column_strings: Dict[CK, str] = None
    
    # Read-only state
    _read_only: bool = False
    
    # Thread safety
    _lock: rwlock.RWLockFairD = None
    _rlock: rwlock.RWLockFairD._aReader = None
    _wlock: rwlock.RWLockFairD._aWriter = None

    def __post_init__(self):
        """
        Initialize derived data structures and set up thread safety.
        """
        # Initialize locks
        self._lock = rwlock.RWLockFairD()
        object.__setattr__(self, '_rlock', self._lock.gen_rlock())
        object.__setattr__(self, '_wlock', self._lock.gen_wlock())
        
        # Initialize derived data structures
        object.__setattr__(self, '_column_keys', list(self._column_information.keys()))
        object.__setattr__(self, '_column_types', {})
        object.__setattr__(self, '_display_units', {})
        object.__setattr__(self, '_dimensions', {})
        object.__setattr__(self, '_internal_dataframe_column_strings', {})
        
        # Populate derived data structures
        for column_key, column_info in self._column_information.items():
            self._column_types[column_key] = column_info.column_type
            self._display_units[column_key] = column_info.display_unit
            self._dimensions[column_key] = column_info.dimension
            self._internal_dataframe_column_strings[column_key] = self._internal_dataframe_column_name_formatter(column_info)

    def __str__(self) -> str:
        """
        Return a string representation of the dataframe.
        """
        with self._rlock:
            return f"UnitedDataframe[{type(self).__name__}]({self.shape[0]} rows, {self.shape[1]} columns)"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the dataframe.
        """
        with self._rlock:
            return f"UnitedDataframe[{type(self).__name__}](\n  Shape: {self.shape},\n  Columns: {self.column_keys},\n  Read-only: {self._read_only}\n)"



    # Alternative constructor methods
    @classmethod
    def from_dataframe_and_column_information_list(
        cls,
        dataframe: pd.DataFrame,
        column_information: Dict[CK, ColumnInformation[CK]],
        internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter,
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a pandas DataFrame and column information.
        
        Args:
            dataframe: The pandas DataFrame containing the data
            column_information: Dictionary mapping column keys to their information
            internal_dataframe_column_name_formatter: Function to format internal column names
            read_only: Whether the dataframe should be read-only
            
        Returns:
            UnitedDataframe: New instance with the provided data
        """
        instance = cls(
            dataframe.copy(),
            column_information,
            internal_dataframe_column_name_formatter
        )
        instance._read_only = read_only
        return instance

    @classmethod
    def create_from_dataframe_and_column_information_list(
        cls,
        dataframe: pd.DataFrame,
        column_information: Dict[CK, ColumnInformation[CK]],
        internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter,
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        """
        Alternative name for from_dataframe_and_column_information_list for backward compatibility.
        """
        return cls.from_dataframe_and_column_information_list(
            dataframe, column_information, internal_dataframe_column_name_formatter, read_only
        )

    # Context manager support
    def __enter__(self) -> "UnitedDataframe[CK]":
        """
        Enter context manager (acquire write lock).
        """
        self._wlock.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager (release write lock).
        """
        self._wlock.__exit__(exc_type, exc_val, exc_tb)