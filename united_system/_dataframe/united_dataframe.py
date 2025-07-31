"""
Main UnitedDataframe class that combines all mixins.

This is the primary class that users will interact with. It inherits from all
the mixins to provide a complete dataframe implementation with units support.
"""

from typing import Generic, Optional, Type, Tuple, overload
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
            column_keys: Sequence[CK]|dict[CK, Tuple[ColumnType, Optional[Unit|Dimension]]|ColumnType]|dict[CK, Tuple[ColumnType, Optional[Unit]]|ColumnType]|dict[CK, Tuple[ColumnType, Optional[Dimension]]|ColumnType] = [],
            column_types: Optional[dict[CK, ColumnType]] = None,
            column_units: Optional[dict[CK, Optional[Unit|Dimension]]] = None,
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> None:
        """
        Initialize a UnitedDataframe instance.
        """
        pass

    @overload
    def __new__(
            cls,
            column_keys: dict[CK, Tuple[ColumnType, Optional[Unit]]|ColumnType],
            column_types: None = None,
            column_units: None = None,
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> "UnitedDataframe[CK]":
        """
        Initialize a UnitedDataframe instance.
        """
        ...

    @overload
    def __new__(
            cls,
            column_keys: dict[CK, Tuple[ColumnType, Optional[Dimension]]|ColumnType],
            column_types: None = None,
            column_units: None = None,
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> "UnitedDataframe[CK]":
        """
        Initialize a UnitedDataframe instance.
        """
        ...

    @overload
    def __new__(
            cls,
            column_keys: dict[CK, Tuple[ColumnType, Optional[Unit|Dimension]]|ColumnType],
            column_types: None = None,
            column_units: None = None,
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> "UnitedDataframe[CK]":
        """
        Initialize a UnitedDataframe instance.
        """
        ...

    @overload
    def __new__(
            cls,
            column_keys: Sequence[CK],
            column_types: dict[CK, ColumnType],
            column_units: dict[CK, Optional[Unit|Dimension]],
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter,
            read_only: bool,
    ) -> "UnitedDataframe[CK]":
        """
        Initialize a UnitedDataframe instance.
        """
        ...
        
    def __new__(
            cls,
            column_keys: Sequence[CK]|dict[CK, Tuple[ColumnType, Optional[Unit|Dimension]]|ColumnType]|dict[CK, Tuple[ColumnType, Optional[Unit]]|ColumnType]|dict[CK, Tuple[ColumnType, Optional[Dimension]]|ColumnType] = [],
            column_types: Optional[dict[CK, ColumnType]] = None,
            column_units: Optional[dict[CK, Optional[Unit|Dimension]]] = None,
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
            read_only: bool = False,
    ) -> "UnitedDataframe[CK]":
        """
        Create a new UnitedDataframe instance.
        """

        _column_types: dict[CK, ColumnType] = {}
        _column_units: dict[CK, Optional[Unit|Dimension]] = {}
        if isinstance(column_keys, dict):
            _column_keys: Sequence[CK] = list(column_keys.keys())
            if not column_types == None or not column_units == None:
                raise ValueError("If column_keys is a dict, column_types and column_units must be None.")
            for key, value in column_keys.items():
                if isinstance(value, tuple):
                    _column_type: ColumnType = value[0]
                    _column_unit: Optional[Unit|Dimension] = value[1]
                else:
                    _column_type: ColumnType = value
                    _column_unit: Optional[Unit|Dimension] = None
                _column_types[key] = _column_type
                _column_units[key] = _column_unit
        else:
            _column_keys: Sequence[CK] = column_keys
            if column_types == None:
                raise ValueError("If column_keys is a sequence, column_types must be provided.")
            if column_units == None:
                for key in _column_keys:
                    _column_units[key] = None
            else:
                _column_units: dict[CK, Optional[Unit|Dimension]] = column_units

        # If column_keys are provided, create an empty dataframe with those columns
        if _column_keys:
            column_units_or_dimensions: dict[CK, Optional[Unit | Dimension]] = {key: _column_units.get(key) for key in column_keys}
            return cls.create_empty(
                column_keys=list(_column_keys),
                column_types=_column_types,
                column_units_or_dimensions=column_units_or_dimensions,
                internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter
            )
        else:
            # Empty dataframe case
            units_for_dataframe: dict[CK, Optional[Unit]] = {}
            for key, value in _column_units.items():
                if isinstance(value, Unit):
                    units_for_dataframe[key] = value
                elif isinstance(value, Dimension):
                    units_for_dataframe[key] = value.canonical_unit
                else:
                    units_for_dataframe[key] = None

            return cls._construct(
                dataframe=pd.DataFrame(),
                column_keys=_column_keys,
                column_types=_column_types,
                column_units=units_for_dataframe,
                internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
                read_only=read_only,
            )
    
    @classmethod
    def _construct(
            cls,
            dataframe: pd.DataFrame,
            column_keys: Sequence[CK],
            column_types: dict[CK, ColumnType],
            column_units: dict[CK, Optional[Unit]],
            internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter,
            read_only: bool = False,
            copy_dataframe: bool = False,
            rename_dataframe_columns: bool = False
    ) -> "UnitedDataframe[CK]":
        """
        INTERNAL: Initialize derived data structures and set up thread safety.

        The dataframe must match the expected column keys, types, and units.
        """

        if len(dataframe.columns) != len(column_keys):
            raise ValueError(f"Number of columns in dataframe ({len(dataframe.columns)}) does not match number of column keys ({len(column_keys)}).")
        if len(column_keys) != len(column_types):
            raise ValueError(f"Number of column keys ({len(column_keys)}) does not match number of column types ({len(column_types)}).")
        if len(column_keys) != len(column_units):
            raise ValueError(f"Number of column keys ({len(column_keys)}) does not match number of column units ({len(column_units)}).")
        
        dataframe_column_names: dict[CK, str] = {}
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