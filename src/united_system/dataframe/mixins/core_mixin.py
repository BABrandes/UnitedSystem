"""
Core functionality mixin for UnitedDataframe.

Contains basic properties, initialization helpers, and core utility methods.
"""

from typing import Generic, TypeVar, Literal
import pandas as pd
from readerwriterlock import rwlock
from pandas._typing import Dtype

from ...utils import JSONable, HDF5able
from ..column_information import ColumnKey, ColumnInformation, InternalDataFrameNameFormatter
from ..column_type import ColumnType, SCALAR_TYPE, ARRAY_TYPE
from ...unit import Unit
from ...dimension import Dimension

import numpy as np

CK = TypeVar("CK", bound=ColumnKey|str, default=str)

class CoreMixin(JSONable, HDF5able, Generic[CK]):
    """
    Core functionality mixin for UnitedDataframe.
    
    Provides basic properties, initialization helpers, and core utility methods
    that are used throughout the dataframe implementation.
    """
    
    # Basic properties and information
    def __len__(self) -> int:
        """
        Return the number of rows in the dataframe.
        
        Returns:
            int: The number of rows in the dataframe
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe)
    
    def is_empty(self) -> bool:
        """
        Check if the dataframe is empty.
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe) == 0
        
    def is_numeric(self, column_key: CK) -> bool:
        """
        Check if a column is numeric.
        """
        with self._rlock:
            return self.column_type(column_key).is_numeric()
        
    def has_unit(self, column_key: CK) -> bool:
        """
        Check if a column has a unit.
        """
        with self._rlock:
            return self.column_type(column_key).has_unit()

    @property
    def cols(self) -> int:
        """
        Return the number of columns in the dataframe.
        """
        with self._rlock:
            return len(self._column_keys)
        
    @property
    def rows(self) -> int:
        """
        Return the number of rows in the dataframe.
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe)

    @property
    def shape(self) -> tuple[int, int]:
        """
        Get the shape of the dataframe as (rows, columns).
        
        Returns:
            tuple[int, int]: A tuple containing (number_of_rows, number_of_columns)
        """
        with self._rlock:
            return self._internal_canonical_dataframe.shape

    @property
    def size(self) -> int:
        """
        Get the total number of elements in the dataframe.
        
        Returns:
            int: Total number of elements (rows × columns)
        """
        with self._rlock:
            return self._internal_canonical_dataframe.size

    @property
    def empty(self) -> bool:
        """
        Check if the dataframe is empty.
        
        Returns:
            bool: True if the dataframe has no rows, False otherwise
        """
        with self._rlock:
            return self._internal_canonical_dataframe.empty

    # Internal utilities
    def create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Create the internal dataframe column name for a column.

        Args:
            column_key (CK): The column key to create the internal dataframe column name for

        Returns:
            str: The internal dataframe column name for the column
        """
        if isinstance(column_key, ColumnKey):
            column_key_str: str = column_key.to_string()
        else:
            column_key_str: str = column_key
        return self._internal_dataframe_column_name_formatter(self._column_information[column_key])

    @staticmethod
    def column_key_to_string(column_key: CK) -> str:
        if isinstance(column_key, ColumnKey):
            return column_key.to_string()
        else:
            return column_key

    @staticmethod
    def column_key_as_str(column_key: CK) -> str:
        """
        Get the string representation of a column key.
        
        Args:
            column_key (CK): The column key to convert
            
        Returns:
            str: The string representation of the column key
        """
        match column_key:
            case ColumnKey():
                return column_key.to_string()
            case str():
                return column_key
            case _:
                raise ValueError(f"Invalid column key: {column_key}.")

    def compatible_with_column(self, column_key: CK, value: SCALAR_TYPE|ARRAY_TYPE|np.ndarray|pd.Series) -> bool:
        """Check if a value is compatible with a value type and /or unit."""
        with self._rlock:
            column_type: ColumnType = self.column_type(column_key)
            # Check for the united_quantity
            match column_type.value.has_unit, isinstance(value, United):
                case True, True:
                    # Good so far: The column has a unit, and the value has a unit.
                    if value.unit_quantity != self.dimensions(column_key):
                        # Failed: The units are not the same.
                        return False
                case True, False:
                    # Failed: The column has a unit, but the value does not.
                    return False
                case False, False:
                    # All good! The column has no unit, and the value does not have a unit.
                    pass
                case False, True:
                    # Failed: The column has no unit, but the value does.
                    return False
                case _, _:
                    raise ValueError(f"Invalid value type: {type(value)}")
            # Check for the value type
            return column_type.check_compatibility(value)

    @property
    def internal_dataframe_deepcopy(self) -> pd.DataFrame:
        """
        Get a deep copy of the internal pandas DataFrame.
        
        Returns:
            pd.DataFrame: A deep copy of the underlying pandas DataFrame
        """
        with self._rlock:
            return self._internal_canonical_dataframe.copy(deep=True)

    def copy(self, deep: bool = True) -> "UnitedDataframe[CK]":
        """
        Create a deep copy of the United_Dataframe.
        
        Returns:
            United_Dataframe: A new instance with copied data and metadata
        """
        with self._rlock:
            new_df: "UnitedDataframe[CK]" = UnitedDataframe(
                self._internal_canonical_dataframe.copy(deep=deep),
                self._column_information,
                self._internal_dataframe_column_name_formatter)
            # The locks will be created in __post_init__, but we need to ensure they're properly initialized
            return new_df
    
    # Read-only state management
    def is_read_only(self) -> bool:
        """
        Check if the dataframe is in read-only mode.
        
        Returns:
            bool: True if the dataframe is read-only, False otherwise
        """
        with self._rlock:
            return self._read_only

    def set_read_only(self, read_only: bool) -> None:
        """
        Set the read-only status of the dataframe.
        
        Args:
            read_only (bool): True to make the dataframe read-only, False to allow modifications
        """
        with self._wlock:
            self._read_only = read_only

    # Lock management
    def acquire_read_lock(self) -> rwlock.RWLockFairD._aReader:
        return self._rlock

    def acquire_write_lock(self) -> rwlock.RWLockFairD._aWriter:
        return self._wlock
    
    def release_read_lock(self, lock: rwlock.RWLockFairD._aReader) -> None:
        lock.release()

    def release_write_lock(self, lock: rwlock.RWLockFairD._aWriter) -> None:
        lock.release()

    # Internal helper methods
    def _get_dataframe_with_new_canonical_dataframe(self, new_canonical_dataframe: pd.DataFrame) -> "UnitedDataframe[CK]":
        """
        Get a new United_Dataframe with a new canonical dataframe, but using the same column information.
        """
        return UnitedDataframe[CK].create_from_dataframe_and_column_information_list(
            new_canonical_dataframe,
            self._column_information,
            self._internal_dataframe_column_name_formatter,
            True)
    
    def _get_numpy_dtype_from_precision(self, column_key_or_type: CK|ColumnType, precision: Literal[8, 16, 32, 64, 128, 256]|None) -> Dtype:
        """
        Get the numpy dtype from the precision.
        """
        column_type: ColumnType = self.column_type(column_key_or_type) if isinstance(column_key_or_type, CK) else column_key_or_type
        if precision is None:
            return column_type.value.numpy_storage_options[0]
        else:
            for numpy_dtype in column_type.value.numpy_storage_options:
                if numpy_dtype.itemsize == precision:
                    return numpy_dtype
            raise ValueError(f"Precision {precision} not found in the numpy storage options for column type {column_type}.")

    # Utility methods
    def get_numeric_column_keys(self) -> list[CK]:
        """
        Get a list of column keys for numeric columns only.
        
        Returns:
            list[CK]: List of column keys for numeric columns
        """
        with self._rlock:
            return [column_key for column_key in self._column_keys if self.is_numeric(column_key)]

    def describe(self) -> pd.DataFrame:
        """
        Generate descriptive statistics for numeric columns.
        
        Returns:
            pd.DataFrame: A pandas DataFrame containing descriptive statistics
                         with unit information in column names
        """
        with self._rlock:
            numeric_columns: list[CK] = self.get_numeric_column_keys()
            
            if not numeric_columns:
                return pd.DataFrame()
            
            description = self._internal_canonical_dataframe[numeric_columns].describe()
            
            # Add unit information to column names
            unit_info = {}
            for column_key in numeric_columns:
                unit = self.display_unit(column_key)
                unit_info[column_key] = f"{column_key} [{unit}]" if unit is not None else f"{column_key} [-]"
            
            description.columns = [unit_info.get(col, col) for col in description.columns]
            return description

    def info(self, verbose: bool = True, max_cols: int = 20, memory_usage: bool = True) -> None:
        """
        Print a concise summary of the dataframe.
        
        Args:
            verbose (bool): If True, print detailed information about each column
            max_cols (int): Maximum number of columns to display (unused in this implementation)
            memory_usage (bool): If True, print memory usage information
        """
        with self._rlock:
            print(f"<United_Dataframe>")
            print(f"Index: {len(self._internal_canonical_dataframe)} entries, 0 to {len(self._internal_canonical_dataframe) - 1}")
            print(f"Data columns (total {len(self._column_keys)} columns):")
                
            if verbose:
                for i, column_key in enumerate(self._column_keys):
                    non_null_count = self._internal_canonical_dataframe.iloc[:, i].count()
                    dtype = self._column_types[column_key].value.corresponding_pandas_type
                    unit_str = f" [{self._display_units[column_key]}]" if self._display_units[column_key] is not None else " [-]"
                    print(f" {i}  {self.column_key_as_str(column_key)}{unit_str}  {dtype}  {non_null_count} non-null")
            else:
                print(f" {len(self._column_keys)} columns")
            
            if memory_usage:
                memory_usage_bytes = self._internal_canonical_dataframe.memory_usage(deep=True).sum()
                print(f"memory usage: {memory_usage_bytes} bytes")

    def sample(self, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> "UnitedDataframe[CK]":
        """
        Get a random sample of rows from the dataframe.
        
        Args:
            n (int | None): Number of rows to sample
            frac (float | None): Fraction of rows to sample (0.0 to 1.0)
            random_state (int | None): Random seed for reproducibility
            
        Returns:
            United_Dataframe: A new dataframe containing the sampled rows
            
        Raises:
            ValueError: If both n and frac are specified
        """
        with self._rlock:
            if n is None and frac is None:
                n = 1
            elif n is not None and frac is not None:
                raise ValueError("Cannot specify both n and frac")
            
            sampled_df = self._internal_canonical_dataframe.sample(n=n, frac=frac, random_state=random_state)
            return UnitedDataframe(
                sampled_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter) 