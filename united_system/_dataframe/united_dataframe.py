"""
Main UnitedDataframe class that combines all mixins.

This is the primary class that users will interact with. It inherits from all
the mixins to provide a complete dataframe implementation with units support.
"""

from typing import Generic, Dict, Optional, Type
from collections.abc import Sequence
from types import TracebackType
import pandas as pd
from readerwriterlock import rwlock

from .mixins import *
from .mixins.dataframe_protocol import CK
from .column_type import ColumnType
from .internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter, SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
from .._units_and_dimension.unit import Unit
from .._units_and_dimension.dimension import Dimension

class UnitedDataframe(
    CoreMixin[CK],
    ColKeyMixin[CK],
    ColTypeMixin[CK],
    UnitMixin[CK],
    DimensionMixin[CK],
    ColumnAccessMixin[CK],
    ColumnOperationsMixin[CK],
    ColumnStatisticsMixin[CK],
    RowOperationsMixin[CK],
    RowAccessMixin[CK],
    RowStatisticsMixin[CK],
    CellOperationsMixin[CK],
    MaskOperationsMixin[CK],
    FilterMixin[CK],
    SerializationMixin[CK],
    ConstructorMixin[CK],
    GroupbyMixin[CK],
    IterMixin[CK],
    AccessorGetitemMixin[CK],
    AccessorSetitemMixin[CK],
    SegmentMixin[CK],
    Generic[CK],
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

    def __init__(
            self,
            column_keys: Sequence[CK] = [],
            column_types: Dict[CK, ColumnType] = {},
            column_units: Dict[CK, Optional[Unit]] = {},
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> None:
        """
        Initialize a UnitedDataframe instance.
        """
        pass

    def __new__(
            cls,
            column_keys: Sequence[CK] = [],
            column_types: Dict[CK, ColumnType] = {},
            column_units: Dict[CK, Optional[Unit]] = {},
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> "UnitedDataframe[CK]":
        """
        Create a new UnitedDataframe instance.
        """
        
        # If column_keys are provided, create an empty dataframe with those columns
        if column_keys:
            column_units_or_dimensions: dict[CK, Optional[Unit | Dimension]] = {key: column_units.get(key) for key in column_keys}
            return cls.create_empty_dataframe(
                column_keys=list(column_keys),
                column_types=column_types,
                column_units_or_dimensions=column_units_or_dimensions,
                internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter
            )
        else:
            # Empty dataframe case
            return cls._construct(
                dataframe=pd.DataFrame(),
                column_keys=column_keys,
                column_types=column_types,
                column_units=column_units,
                internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
                read_only=read_only,
            )
    
    @classmethod
    def _construct(
            cls,
            dataframe: pd.DataFrame,
            column_keys: Sequence[CK],
            column_types: Dict[CK, ColumnType],
            column_units: Dict[CK, Optional[Unit]],
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter,
            read_only: bool = False,
            copy_dataframe: bool = False,
            rename_dataframe_columns: bool = False
    ) -> "UnitedDataframe[CK]":
        """
        Initialize derived data structures and set up thread safety.
        """

        if len(dataframe.columns) != len(column_keys):
            raise ValueError(f"Number of columns in dataframe ({len(dataframe.columns)}) does not match number of column keys ({len(column_keys)}).")
        if len(column_keys) != len(column_types):
            raise ValueError(f"Number of column keys ({len(column_keys)}) does not match number of column types ({len(column_types)}).")
        if len(column_keys) != len(column_units):
            raise ValueError(f"Number of column keys ({len(column_keys)}) does not match number of column units ({len(column_units)}).")
        
        dataframe_column_names: Dict[CK, str] = {}
        for column_index, column_key in enumerate(column_keys):
            dataframe_column_names[column_key] = internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_units[column_key])
            if dataframe_column_names[column_key] != dataframe.columns[column_index]:
                if rename_dataframe_columns:
                    dataframe.rename(columns={dataframe.columns[column_index]: dataframe_column_names[column_key]}, inplace=True)
                else:
                    raise ValueError(f"Column {column_key} predicts a different name in the dataframe ({dataframe_column_names[column_key]}) than the actual name in the dataframe ({dataframe.columns[column_index]}).")

        if copy_dataframe:
            dataframe = dataframe.copy(deep=True)

        instance: "UnitedDataframe[CK]" = object.__new__(cls)

        # Initialize locks
        instance._lock = rwlock.RWLockFairD()
        object.__setattr__(instance, '_rlock', instance._lock.gen_rlock())
        object.__setattr__(instance, '_wlock', instance._lock.gen_wlock())
        
        # Initialize derived data structures
        object.__setattr__(instance, '_column_keys', list(column_keys))
        object.__setattr__(instance, '_column_types', column_types)
        object.__setattr__(instance, '_column_units', column_units)
        object.__setattr__(instance, '_internal_dataframe', dataframe)
        object.__setattr__(instance, '_internal_dataframe_column_names', dataframe_column_names)
        object.__setattr__(instance, '_internal_dataframe_column_name_formatter', internal_dataframe_column_name_formatter)
        object.__setattr__(instance, '_read_only', read_only)

        return instance

    def __str__(self) -> str:
        """
        Return a string representation of the dataframe.
        """
        with self._rlock:
            rows = len(self._internal_dataframe)
            cols = len(self._column_keys)
            return f"UnitedDataframe[{type(self).__name__}]({rows} rows, {cols} columns)"

    def __repr__(self) -> str:
        """
        Return a detailed string representation of the dataframe.
        """
        with self._rlock:
            rows = len(self._internal_dataframe)
            cols = len(self._column_keys)
            return f"UnitedDataframe[{type(self).__name__}](\n  Shape: ({rows}, {cols}),\n  Columns: {self._column_keys},\n  Read-only: {self._read_only}\n)"

    # Context manager support
    def __enter__(self) -> "UnitedDataframe[CK]":
        """
        Enter context manager (acquire write lock).
        """
        self._wlock.__enter__()
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[Exception], exc_tb: Optional[TracebackType]) -> Optional[bool]:
        """
        Exit context manager (release write lock).
        """
        return self._wlock.__exit__(exc_type, exc_val, exc_tb)