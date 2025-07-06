"""
Core base functionality for UnitedDataframe.

This module contains the essential functionality that all other mixins depend on.
"""

import pandas as pd
import numpy as np
from typing import Generic, TypeVar, Dict, List, Any, Optional, Union, Tuple
from contextlib import ExitStack
from readerwriterlock import rwlock
from datetime import datetime
from h5py._hl.group import Group

from ...units.base_classes.base_unit import BaseUnit, UnitQuantity
from ...scalars.united_scalar import UnitedScalar
from ...scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from ...scalars.complex_united_scalar.complex_united_scalar import ComplexUnitedScalar
from ...utils import JSONable, HDF5able
from ...units.utils import United
from ...arrays.utils import ArrayLike
from ..utils import ColumnKey, ColumnInformation, InternalDataFrameNameFormatter, SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
from ..column_type import ColumnType, SCALAR_TYPE, ARRAY_TYPE

CK = TypeVar("CK", bound=ColumnKey|str, default=str)


class UnitedDataframeCore(JSONable, HDF5able, Generic[CK]):
    """
    Core functionality and state management for UnitedDataframe.
    
    This class provides the essential infrastructure that all other mixins depend on:
    - Core data storage and metadata
    - Thread safety with read/write locks
    - Basic properties and validation
    - Copy and state management
    """
    
    def __init__(self,
                 internal_canonical_dataframe: pd.DataFrame,
                 column_information: Dict[CK, ColumnInformation],
                 internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter[CK] = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER):
        """
        Initialize the core UnitedDataframe infrastructure.
        
        Args:
            internal_canonical_dataframe: The underlying pandas DataFrame
            column_information: Metadata about each column
            internal_dataframe_column_name_formatter: Function to format internal column names
        """
        # Core data storage
        self._internal_canonical_dataframe: pd.DataFrame = internal_canonical_dataframe
        self._column_information: Dict[CK, ColumnInformation] = column_information.copy()
        self._internal_dataframe_column_name_formatter = internal_dataframe_column_name_formatter
        
        # Validate core structure
        self._validate_core_structure()
        
        # Generate derived data structures
        self._generate_derived_structures()
        
        # Initialize state
        self._read_only: bool = False
        
        # Initialize thread safety
        self._lock: rwlock.RWLockFairD = rwlock.RWLockFairD()
        self._rlock: rwlock.RWLockFairD._aReader = self._lock.gen_rlock()
        self._wlock: rwlock.RWLockFairD._aWriter = self._lock.gen_wlock()
    
    def _validate_core_structure(self) -> None:
        """Validate that the core data structures are consistent."""
        if len(self._column_information) != len(self._internal_canonical_dataframe.columns):
            raise ValueError(
                f"Column count mismatch: {len(self._column_information)} columns in metadata "
                f"vs {len(self._internal_canonical_dataframe.columns)} in dataframe"
            )
    
    def _generate_derived_structures(self) -> None:
        """Generate derived data structures from column information."""
        self._column_keys: List[CK] = list(self._column_information.keys())
        self._unit_quantities: Dict[CK, Optional[UnitQuantity]] = {
            col_key: col_info.unit_quantity 
            for col_key, col_info in self._column_information.items()
        }
        self._display_units: Dict[CK, Optional[BaseUnit]] = {
            col_key: col_info.display_unit 
            for col_key, col_info in self._column_information.items()
        }
        self._column_types: Dict[CK, ColumnType] = {
            col_key: col_info.column_type 
            for col_key, col_info in self._column_information.items()
        }
        self._internal_dataframe_column_strings: Dict[CK, str] = {
            col_key: col_info.internal_dataframe_column_name(
                col_key, self._internal_dataframe_column_name_formatter
            ) 
            for col_key, col_info in self._column_information.items()
        }
    
    def __len__(self) -> int:
        """Return the number of rows in the dataframe."""
        with self._rlock:
            return len(self._internal_canonical_dataframe)
    
    # ============================================================================
    # Properties
    # ============================================================================
    
    @property
    def cols(self) -> int:
        """Return the number of columns in the dataframe."""
        with self._rlock:
            return len(self._column_keys)
    
    @property
    def rows(self) -> int:
        """Return the number of rows in the dataframe."""
        with self._rlock:
            return len(self._internal_canonical_dataframe)
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get the shape of the dataframe as (rows, columns)."""
        with self._rlock:
            return self._internal_canonical_dataframe.shape
    
    @property
    def size(self) -> int:
        """Get the total number of elements in the dataframe."""
        with self._rlock:
            return self._internal_canonical_dataframe.size
    
    @property
    def empty(self) -> bool:
        """Check if the dataframe is empty."""
        with self._rlock:
            return self._internal_canonical_dataframe.empty
    
    @property
    def column_keys(self) -> List[CK]:
        """Get a copy of all column keys."""
        with self._rlock:
            return self._column_keys.copy()
    
    @property
    def internal_dataframe_deepcopy(self) -> pd.DataFrame:
        """Get a deep copy of the internal pandas DataFrame."""
        with self._rlock:
            return self._internal_canonical_dataframe.copy(deep=True)
    
    # ============================================================================
    # State Management
    # ============================================================================
    
    def copy(self, deep: bool = True) -> "UnitedDataframeCore[CK]":
        """Create a copy of the dataframe."""
        with self._rlock:
            return UnitedDataframeCore(
                self._internal_canonical_dataframe.copy(deep=deep),
                self._column_information,
                self._internal_dataframe_column_name_formatter
            )
    
    def is_read_only(self) -> bool:
        """Check if the dataframe is in read-only mode."""
        with self._rlock:
            return self._read_only
    
    def set_read_only(self, read_only: bool) -> None:
        """Set the read-only status of the dataframe."""
        with self._wlock:
            self._read_only = read_only
    
    # ============================================================================
    # Lock Management
    # ============================================================================
    
    def acquire_read_lock(self) -> rwlock.RWLockFairD._aReader:
        """Acquire a read lock."""
        return self._rlock
    
    def acquire_write_lock(self) -> rwlock.RWLockFairD._aWriter:
        """Acquire a write lock."""
        return self._wlock
    
    def release_read_lock(self, lock: rwlock.RWLockFairD._aReader) -> None:
        """Release a read lock."""
        lock.release()
    
    def release_write_lock(self, lock: rwlock.RWLockFairD._aWriter) -> None:
        """Release a write lock."""
        lock.release()
    
    # ============================================================================
    # Column Information Access
    # ============================================================================
    
    def has_column(self, column_key: CK) -> bool:
        """Check if a column exists by column key."""
        with self._rlock:
            return column_key in self._column_keys
    
    def column_type(self, column_key: CK) -> ColumnType:
        """Get the column type for a column."""
        with self._rlock:
            return self._column_types[column_key]
    
    def display_unit(self, column_key: CK) -> Optional[BaseUnit]:
        """Get the display unit for a column."""
        with self._rlock:
            return self._display_units[column_key]
    
    def unit_quantity(self, column_key: CK) -> Optional[UnitQuantity]:
        """Get the unit quantity for a column."""
        with self._rlock:
            return self._unit_quantities[column_key]
    
    def internal_dataframe_column_string(self, column_key: CK) -> str:
        """Get the internal dataframe column string for a column."""
        with self._rlock:
            return self._internal_dataframe_column_strings[column_key]
    
    def get_column_information_dict(self) -> Dict[CK, ColumnInformation[CK]]:
        """Get the column information dictionary."""
        with self._rlock:
            return {
                column_key: ColumnInformation[CK](
                    column_key, 
                    self._unit_quantities[column_key], 
                    self._column_types[column_key], 
                    self._display_units[column_key]
                ) 
                for column_key in self._column_keys
            }
    
    # ============================================================================
    # Internal Utilities
    # ============================================================================
    
    @staticmethod
    def column_key_as_str(column_key: CK) -> str:
        """Get the string representation of a column key."""
        if isinstance(column_key, ColumnKey):
            return column_key.to_string()
        elif isinstance(column_key, str):
            return column_key
        else:
            raise ValueError(f"Invalid column key: {column_key}")
    
    def _get_dataframe_with_new_canonical_dataframe(self, new_canonical_dataframe: pd.DataFrame) -> "UnitedDataframeCore[CK]":
        """Get a new dataframe with a new canonical dataframe, using the same column information."""
        return UnitedDataframeCore[CK](
            new_canonical_dataframe,
            self._column_information,
            self._internal_dataframe_column_name_formatter
        )
