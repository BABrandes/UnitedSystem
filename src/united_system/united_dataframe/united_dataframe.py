from h5py._hl.group import Group
from ..units.unit import Unit, UnitQuantity
from ..scalars.united_scalar import UnitedScalar
import pandas as pd
import numpy as np
from typing import Callable, Generic, TypeVar, overload, cast, Iterator, Literal
from datetime import datetime
from pandas._typing import Dtype
from ..arrays.united_array import UnitedArray
from ..units.unit_quantity import UnitQuantity
from ..scalars.real_united_scalar import RealUnitedScalar
from ..scalars.complex_united_scalar import ComplexUnitedScalar
from ..united_dataframe._column_accessor import _ColumnAccessor
from ..united_dataframe._row_accessor import _RowAccessor
from ..united_dataframe._group import GroupBy
from contextlib import ExitStack
from readerwriterlock import rwlock
import math
from typing import Any
import operator
from ..utils import JSONable, HDF5able
from ..units.utils import United
from ..arrays.utils import ArrayLike
from ..scalars.real_united_scalar import RealUnitedScalar
from ..scalars.complex_united_scalar import ComplexUnitedScalar
from ..arrays.real_united_array import RealUnitedArray
from ..arrays.complex_united_array import ComplexUnitedArray
from ..arrays.string_array import StringArray
from ..arrays.int_array import IntArray
from ..arrays.float_array import FloatArray
from ..arrays.bool_array import BoolArray
from ..arrays.timestamp_array import TimestampArray
from ..arrays.complex_array import ComplexArray
from pandas import Timestamp
from .utils import ColumnKey, ColumnInformation, InternalDataFrameNameFormatter, SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
from .column_type import ColumnType, SCALAR_TYPE, ARRAY_TYPE

CK = TypeVar("CK", bound=ColumnKey|str, default=str)
CK_I2 = TypeVar("CK_I2", bound=ColumnKey|str, default=str)

CK_CF = TypeVar("CK_CF", bound=ColumnKey|str, default=str)

class UnitedDataframe(JSONable, HDF5able, Generic[CK]):
    """
    A unit-aware DataFrame that maintains type safety and thread safety.
    
    This class extends pandas DataFrame functionality with unit system integration,
    providing a type-safe, thread-safe interface for handling data with physical units.
    
    Attributes:
        _canonical_dataframe: The underlying pandas DataFrame storing the data
        _column_keys: List of column keys that identify each column
        _si_quantities: List of SI quantities corresponding to each column
        _value_types: List of value types (FLOAT64, INT32, etc.) for each column
        _read_only: Whether the dataframe is read-only
        _display_units: List of display units for each column
        _internal_column_name_formatter: Function to format internal column names
        _lock: Read-write lock for thread safety
    """
    
    def __init__(self,
                 internal_canonical_dataframe: pd.DataFrame,
                 column_information: dict[CK, ColumnInformation],
                 internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter[CK] = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER):
        """
        **This method is not meant to be called by the user.**
        **Use the class methods to create a United_Dataframe.**
        
        Initialize the United_Dataframe.
        
        **The canonical dataframe is set as a reference**
        **No deepcopy is performed on the canonical dataframe.**

        Args:
            internal_canonical_dataframe (pd.DataFrame): The underlying pandas DataFrame storing the data
            column_information (list[Column_Information[CK]]): List of column information that identify each column
            internal_dataframe_column_name_formatter (InternalDataFrameNameFormatter[CK]): Function to format internal column names

        Performs validation checks:
        - Ensures that the number of columns in the dataframe matches the number of column information
        - Ensures that each dataframe column has a corresponding column information
        - Ensures that the internal dataframe column names match with the dataframe columns
        - Ensures that the column information is unique
        - Initializes thread safety locks
        """

        # Step 1: Set the fields from the constructor
        self._internal_canonical_dataframe: pd.DataFrame = internal_canonical_dataframe
        self._column_information: dict[CK, ColumnInformation] = column_information.copy()
        self._internal_dataframe_column_name_formatter: InternalDataFrameNameFormatter[CK] = internal_dataframe_column_name_formatter

        # Step 2: Check that each dataframe column has a corresponding column information
        if len(column_information) != len(self._internal_canonical_dataframe.columns):
            raise ValueError(f"The number of columns in the dataframe ({len(self._internal_canonical_dataframe.columns)}) does not match the number of column information ({len(column_information)}).")
        
        # Step 3: Generate the dictionaries from the column information
        self._column_keys: list[CK] = list(self._column_information.keys())
        self._unit_quantities: dict[CK, UnitQuantity|None] = {column_key: column_information.unit_quantity for column_key, column_information in self._column_information.items()}
        self._display_units: dict[CK, Unit|None] = {column_key: column_information.display_unit for column_key, column_information in self._column_information.items()}
        self._column_types: dict[CK, ColumnType] = {column_key: column_information.column_type for column_key, column_information in self._column_information.items()}
        self._internal_dataframe_column_strings: dict[CK, str] = {column_key: column_information.internal_dataframe_column_name(column_key, self._internal_dataframe_column_name_formatter) for column_key, column_information in self._column_information.items()}

        # Step 4: Check that the internal dataframe column names match with the dataframe columns
        for icd_column_name in self._internal_canonical_dataframe.columns:
            if icd_column_name not in self._internal_dataframe_column_strings:
                raise ValueError(f"The dataframe column {icd_column_name} does not have a corresponding column information.")

        # Step 5: Set the other fields
        self._read_only: bool = False
        self._lock: rwlock.RWLockFairD = rwlock.RWLockFairD()
        self._rlock: rwlock.RWLockFairD._aReader = self._lock.gen_rlock()
        self._wlock: rwlock.RWLockFairD._aWriter = self._lock.gen_wlock()

    def __len__(self) -> int:
        """
        Return the number of rows in the dataframe.
        
        Returns:
            int: The number of rows in the dataframe
        """
        with self._rlock:
            return len(self._internal_canonical_dataframe)
    
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

    def column_information(self, column_key: CK) -> ColumnInformation[CK]:
        with self._rlock:
            return ColumnInformation(column_key, self._unit_quantities[column_key], self._value_types[column_key])
        
    def compatible_with_column(self, column_key: CK, value: SCALAR_TYPE|ARRAY_TYPE|np.ndarray|pd.Series) -> bool:
        """Check if a value is compatible with a value type and /or unit."""
        with self._rlock:
            column_type: ColumnType = self.column_type(column_key)
            # Check for the united_quantity
            match column_type.value.has_unit, isinstance(value, United):
                case True, True:
                    # Good so far: The column has a unit, and the value has a unit.
                    if value.unit_quantity != self.unit_quantities(column_key):
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
            new_df: UnitedDataframe[CK] = UnitedDataframe(
                self._internal_canonical_dataframe.copy(deep=deep),
                self._column_information,
                self._column_types,
                self._internal_dataframe_column_name_formatter)
            # The locks will be created in __post_init__, but we need to ensure they're properly initialized
            return new_df
    
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

    def acquire_read_lock(self) -> rwlock.RWLockFairD._aReader:
        return self._rlock

    def acquire_write_lock(self) -> rwlock.RWLockFairD._aWriter:
        return self._wlock
    
    def release_read_lock(self, lock: rwlock.RWLockFairD._aReader) -> None:
        lock.release()

    def release_write_lock(self, lock: rwlock.RWLockFairD._aWriter) -> None:
        lock.release()

    # ----------- Retrievals: Column keys ------------

    @property
    def column_keys(self) -> list[CK]:
        """
        Get a copy of all column keys.
        
        Returns:
            list[CK]: A copy of the list of column keys
        """
        with self._rlock:
            return self._column_keys.copy()
        
    def has_column(self, column_key: CK) -> bool:
        """
        Check if a column exists by index or column key.
        
        Args:
            index_or_column_key (int|CK): The index or column key to check
            
        Returns:
            bool: True if the column exists, False otherwise
        """
        with self._rlock:
            return column_key in self._column_keys

    def column_keys_of_type(self, *column_key_types: type[CK_CF]) -> list[CK_CF]:
        """
        Get the column keys of a given type.
        """
        with self._rlock:

            column_keys_to_keep_filtered_type: list[CK_CF] = []
            for column_key in self._column_keys:
                if isinstance(column_key, tuple(column_key_types)):
                    column_keys_to_keep_filtered_type.append(column_key)
            return column_keys_to_keep_filtered_type

    # ----------- Retrievals: Column types ------------

    def column_type(self, column_key: CK) -> ColumnType:
        with self._rlock:
            return self._column_types[column_key]

    @overload
    def column_types(self, column_keys: CK) -> ColumnType:
        ...

    @overload
    def column_types(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[ColumnType]:
        ...

    @overload
    def column_types(self, column_keys: list[CK]) -> list[ColumnType]:
        ...

    @overload
    def column_types(self, column_keys: set[CK]) -> set[ColumnType]:
        ...

    def column_types(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> ColumnType|list[ColumnType]|set[ColumnType]:
        """
        Get the value type(s) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the value type(s) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the value type(s) of.
            
        Returns:
            Value_Type|list[Value_Type]|set[Value_Type]: The value type(s) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case ColumnKey()|str():
                    if len(more_column_keys) == 0:
                        return self._value_types[column_keys]
                    else:
                        return [self._value_types[column_keys]] + [self._value_types[more_column_key] for more_column_key in more_column_keys]
                case list():
                    column_types_as_list: list[ColumnType] = []
                    for column_key in column_keys:
                        column_types_as_list.append(self._column_types[column_key])
                    return column_types_as_list
                case set():
                    column_types_as_set: set[ColumnType] = set()
                    for column_key in column_keys:
                        column_types_as_set.add(self._column_types[column_key])
                    return column_types_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def column_type_dict(self) -> dict[CK, ColumnType]:
        """
        Get a dictionary mapping column keys to their value types.
        
        Returns:
            dict[CK, Value_Type]: Dictionary mapping column keys to value types
        """
        with self._rlock:
            return self._value_types.copy()
    
    # ----------- Retrievals: Display Units ------------

    def display_unit(self, column_key: CK) -> Unit:
        with self._rlock:
            return self._display_units[column_key]

    @overload
    def display_units(self, column_keys: CK) -> Unit:
        ... 

    @overload
    def display_units(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[Unit]:
        ...

    @overload
    def display_units(self, column_keys: list[CK]) -> list[Unit]:
        ...

    @overload
    def display_units(self, column_keys: set[CK]) -> set[Unit]:
        ...

    def display_units(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> Unit|list[Unit]|set[Unit]:
        """
        Get the display unit(s) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the display unit(s) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the display unit(s) of.
            
        Returns:
            Unit|list[Unit]|set[Unit]: The display unit(s) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case ColumnKey()|str():
                    if len(more_column_keys) == 0:
                        return self._display_units[column_keys]
                    else:
                        return [self._display_units[column_keys]] + [self._display_units[more_column_key] for more_column_key in more_column_keys]
                case list():
                    display_units_as_list: list[Unit] = []
                    for column_key in column_keys:
                        display_units_as_list.append(self._display_units[column_key])
                    return display_units_as_list
                case set():
                    display_units_as_set: set[Unit] = set()
                    for column_key in column_keys:
                        display_units_as_set.add(self._display_units[column_key])
                    return display_units_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def display_unit_dict(self) -> dict[CK, Unit]:
        """
        Get a dictionary mapping column keys to their display units.
        
        Returns:
            dict[CK, Unit]: Dictionary mapping column keys to display units
        """
        with self._rlock:
            return self._display_units.copy()

    # ----------- Retrievals: UnitQuantity ------------

    def UnitQuantity(self, column_key: CK) -> UnitQuantity:
        with self._rlock:
            return self._unit_quantities[column_key]

    @overload
    def unit_quantities(self, column_keys: CK) -> UnitQuantity:
        ...

    @overload
    def unit_quantities(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[UnitQuantity]:
        ...

    @overload
    def unit_quantities(self, column_keys: list[CK]) -> list[UnitQuantity]:
        ...

    @overload
    def unit_quantities(self, column_keys: set[CK]) -> set[UnitQuantity]:
        ...

    def unit_quantities(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> UnitQuantity|list[UnitQuantity]|set[UnitQuantity]:
        """
        Get the SI quantity(ies) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the SI quantity(ies) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the value type(s) of.
            
        Returns:
            UnitQuantity|list[UnitQuantity]|set[UnitQuantity]: The SI quantity(ies) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case ColumnKey()|str():
                    if len(more_column_keys) == 0:
                        return self._unit_quantities[column_keys]
                    else:
                        return [self._unit_quantities[column_keys]] + [self._unit_quantities[more_column_key] for more_column_key in more_column_keys]
                case list():
                    si_quantities_as_list: list[UnitQuantity] = []
                    for column_key in column_keys:
                        si_quantities_as_list.append(self._unit_quantities[column_key])
                    return si_quantities_as_list
                case set():
                    si_quantities_as_set: set[UnitQuantity] = set()
                    for column_key in column_keys:
                        si_quantities_as_set.add(self._unit_quantities[column_key])
                    return si_quantities_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def UnitQuantity_dict(self) -> dict[CK, UnitQuantity]:
        """
        Get a dictionary mapping column keys to their SI quantities.
        
        Returns:
            dict[CK, UnitQuantity]: Dictionary mapping column keys to SI quantities
        """
        with self._rlock:
            return self._unit_quantities.copy()

    # ----------- Internal Dataframe Column Strings ------------

    def internal_dataframe_column_string(self, column_key: CK) -> str:
        with self._rlock:
            return self._internal_dataframe_column_strings[column_key]

    @overload
    def internal_dataframe_column_strings(self, column_keys: CK) -> str:
        ...

    @overload
    def internal_dataframe_column_strings(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[str]:
        ...

    @overload
    def internal_dataframe_column_strings(self, column_keys: list[CK]) -> list[str]:
        ...

    @overload
    def internal_dataframe_column_strings(self, column_keys: set[CK]) -> set[str]:
        ...

    def internal_dataframe_column_strings(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> str|list[str]|set[str]:
        """
        Get the internal dataframe column strings by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the internal dataframe column strings of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the internal dataframe column strings of.
            
        Returns:
            str|list[str]|set[str]: The internal dataframe column strings of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case ColumnKey()|str():
                    if len(more_column_keys) == 0:
                        return self._internal_dataframe_column_strings[column_keys]
                    else:
                        return [self._internal_dataframe_column_strings[column_keys]] + [self._internal_dataframe_column_strings[more_column_key] for more_column_key in more_column_keys]
                case list():
                    internal_dataframe_column_strings_as_list: list[str] = []
                    for column_key in column_keys:
                        internal_dataframe_column_strings_as_list.append(self._internal_dataframe_column_strings[column_key])
                    return internal_dataframe_column_strings_as_list
                case set():
                    internal_dataframe_column_strings_as_set: set[str] = set()
                    for column_key in column_keys:
                        internal_dataframe_column_strings_as_set.add(self._internal_dataframe_column_strings[column_key])
                    return internal_dataframe_column_strings_as_set
                case _:
                    raise ValueError(f"Invalid column keys: {column_keys}.")
    
    @property
    def internal_dataframe_column_strings_dict(self) -> dict[CK, str]:
        """
        Get a dictionary mapping column keys to their internal dataframe column strings.
        
        Returns:
            dict[CK, str]: Dictionary mapping column keys to internal dataframe column strings
        """
        with self._rlock:
            return self._internal_dataframe_column_strings.copy()
    
    # ----------- Column Information ------------

    def get_column_information_dict(self) -> dict[CK, ColumnInformation[CK]]:
        """
        Get the column information list.
        """
        with self._rlock:
            return {column_key: ColumnInformation[CK](column_key, self._unit_quantities[column_key], self._column_types[column_key], self._display_units[column_key]) for column_key in self._column_keys}

    def column_information_of_type(self, *column_key_types: type[CK_CF]) -> list[tuple[CK_CF, ColumnInformation[CK_CF]]]:
        """
        Filter the dataframe by column key type.
        """
        with self._rlock:

            column_information_of_type: list[ColumnInformation[CK_CF]] = []
            for column_key in self._column_keys:
                column_key_filtered_type: CK = column_key
                if isinstance(column_key_filtered_type, tuple(column_key_types)):
                    column_information_of_type.append(ColumnInformation[CK_CF](
                        column_key_filtered_type,
                        self._unit_quantities[column_key],
                        self._column_types[column_key],
                        self._display_units[column_key]))
            return [(column_key, column_information) for column_key, column_information in column_information_of_type]

    # ----------- Internal stuff ------------
        
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

    def _get_dataframe_with_new_canonical_dataframe(self, new_canonical_dataframe: pd.DataFrame) -> "UnitedDataframe[CK]":
        """
        Get a new United_Dataframe with a new canonical dataframe, but using the same column information.
        """
        return UnitedDataframe[CK].create_from_dataframe_and_column_information_list(
            new_canonical_dataframe,
            self._column_information,
            self._column_types,
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

    # ----------- Setters: Column names ------------

    def rename_column(self, current_column_key: CK, new_column_key: CK):
        """
        Set the column key for a column at the specified index.
        
        Args:
            current_column_key (CK): The current column key
            new_column_key (CK): The new column key
            
        Raises:
            ValueError: If the dataframe is read-only, the name already exists,
                                       or the new column name conflicts with existing columns
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(current_column_key):
                raise ValueError(f"Column key {current_column_key} does not exist in the dataframe.")
            if self.has_column(new_column_key):
                raise ValueError(f"Column key {new_column_key} already exists in the dataframe.")
            current_internal_dataframe_column_name: str = self.internal_dataframe_column_string(current_column_key)
            new_internal_dataframe_column_name: str = self.create_internal_dataframe_column_name(new_column_key)
            if new_internal_dataframe_column_name in self._internal_canonical_dataframe.columns:
                raise ValueError(f"Column name {new_internal_dataframe_column_name} already exists in the dataframe.")
            self._column_information[new_column_key] = self._column_information.pop(current_column_key)
            self._internal_canonical_dataframe.rename(columns={current_internal_dataframe_column_name: new_internal_dataframe_column_name}, inplace=True)
            self._column_keys[self._column_keys.index(current_column_key)] = new_column_key
            self._internal_dataframe_column_strings.pop(current_column_key)
            self._internal_dataframe_column_strings[new_column_key] = new_internal_dataframe_column_name
            self._unit_quantities[new_column_key] = self._unit_quantities.pop(current_column_key)
            self._column_types[new_column_key] = self._column_types.pop(current_column_key)
            self._display_units[new_column_key] = self._display_units.pop(current_column_key)

    # ----------- Column operations ------------

    def column_values_as_numpy_array(self, column_key: CK, in_units: Unit|None=None, precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> np.ndarray:
        """
        Get a column as a numpy array in the specified units.
        
        Args:
            column_key (CK): The column key of the column
            in_units (Unit): The units to return the data in
            
        Returns:
            np.ndarray: The column data as a numpy array in the specified units
            
        Raises:
            ValueError: If the column doesn't exist, unit compatibility issues,
                                       or type mismatches between numeric/non-numeric data
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            column_type: ColumnType = self._column_types[column_key]
            numpy_type: Dtype = self._get_numpy_dtype_from_precision(column_key, precision)
            match column_type.value.has_unit, in_units:
                case False, None:
                    return self._internal_canonical_dataframe[dataframe_column_name].to_numpy().astype(numpy_type)
                case True, None:
                    raise ValueError(f"Unit for column {dataframe_column_name} expected, but got None.")
                case False, _:
                    raise ValueError(f"Column {dataframe_column_name} must not have a unit, but got {in_units}.")
                case True, _:
                    unit_quantity: UnitQuantity = self._unit_quantities[column_key]
                    if in_units.unit_quantity != unit_quantity:
                        raise ValueError(f"Unit {in_units} is not compatible with the unit quantity {unit_quantity} of the column {dataframe_column_name}.")
                    return in_units.from_canonical_value(self._internal_canonical_dataframe[dataframe_column_name].to_numpy()).astype(numpy_type)
        
    def column_values_as_numpy_array_in_canonical_units(self, column_key: CK) -> np.ndarray:
        """
        Get a column as a numpy array in the specified units.
        """
        with self._rlock:
            column_type: ColumnType = self._column_types[column_key]
            if not column_type.value.has_unit:
                raise ValueError(f"There is no canonical unit for column {column_key}.")
            return self.column_values_as_numpy_array(column_key, self._unit_quantities[column_key].canonical_unit())

    def column_values_as_array(self, column_key: CK, display_unit: Unit|None=None) -> ArrayLike[RealUnitedScalar|ComplexUnitedArray|float|int|str|bool|Timestamp]:
        """
        Get a column as a United_Array with its display unit.
        
        Args:
            column_key (CK): The column key of the column
            display_unit (Unit|None): The display unit to use. If None, the column's display unit is used.
            
        Returns:
            United_Array: The column data as a United_Array with the column's display unit
            
        Raises:
            ValueError: If the column doesn't exist
        """
        with self._rlock:
            column_type: ColumnType = self._column_types[column_key]
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            match column_type.value.array_type:
                case RealUnitedArray():
                    return RealUnitedArray.create(self.column_values_as_numpy_array(column_key, display_unit), display_unit)
                case ComplexUnitedArray():
                    return ComplexUnitedArray.create(self.column_values_as_numpy_array(column_key, display_unit), display_unit)
                case StringArray():
                    if self._internal_canonical_dataframe[dataframe_column_name].hasnans:
                        raise ValueError(f"Column {column_key} contains NaN values.")
                    return StringArray.create(self._internal_canonical_dataframe[dataframe_column_name])
                case BoolArray():
                    if self._internal_canonical_dataframe[dataframe_column_name].hasnans:
                        raise ValueError(f"Column {column_key} contains NaN values.")
                    return BoolArray.create(self._internal_canonical_dataframe[dataframe_column_name])
                case TimestampArray():
                    return TimestampArray.create(self._internal_canonical_dataframe[dataframe_column_name])
                case IntArray():
                    if self._internal_canonical_dataframe[dataframe_column_name].hasnans:
                        raise ValueError(f"Column {column_key} contains NaN values.")
                    return IntArray.create(self._internal_canonical_dataframe[dataframe_column_name])
                case FloatArray():
                    return FloatArray.create(self._internal_canonical_dataframe[dataframe_column_name])
                case ComplexArray():
                    return ComplexArray.create(self._internal_canonical_dataframe[dataframe_column_name])
                case _:
                    raise ValueError(f"Column {column_key} is not a United_Array of type {column_type.value.array_type}.")

    def column_values_as_real_united_array(self, column_key: CK, display_unit: Unit|None=None) -> RealUnitedArray:
        """
        Get a column as a RealUnitedArray with its display unit.
        """
        with self._rlock:
            if self._column_types[column_key].value.array_type != RealUnitedArray():
                raise ValueError(f"Column {column_key} is not a RealUnitedArray.")
            return cast(RealUnitedArray, self.column_values_as_array(column_key, display_unit))
        
    def column_values_as_string_array(self, column_key: CK, display_unit: Unit|None=None) -> ComplexUnitedArray:
        """
        Get a column as a ComplexUnitedArray with its display unit.
        """
        with self._rlock:
            if self._column_types[column_key].value.array_type != StringArray():
                raise ValueError(f"Column {column_key} is not a StringArray.")
            return cast(StringArray, self.column_values_as_array(column_key, display_unit))

    def set_column_values_from_numpy_array(self, column_key: CK, values: np.ndarray, unit: Unit, precision: Literal["32", "64"]|None=None) -> None:
        """
        Set column values from a numpy array with the specified unit.
        
        Args:
            column_key (CK): The column key of the column
            values (np.ndarray): The values to set
            unit (Unit): The unit of the provided values
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if unit.unit_quantity != self._unit_quantities[column_key]:
                raise ValueError(f"Unit {unit} is not compatible with the SI quantity {self._unit_quantities[column_key]} of column {column_key}.")
            array: np.ndarray = unit.to_canonical_value(values)
            array = array.astype(self.column_type(column_key).value.dataframe_storage_type)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def set_column_values_from_list(self, column_key: CK, values: list, unit: Unit) -> None:
        """
        Set column values from a list with the specified unit.
        
        Args:
            column_key (CK): The column key of the column
            values (list): The values to set
            unit (Unit): The unit of the provided values
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if unit.UnitQuantity != self._unit_quantities[column_key]:
                raise ValueError(f"Unit {unit} is not compatible with the SI quantity {self._unit_quantities[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = np.array(values, dtype=self.column_type(column_key).value.dataframe_storage_type)
            array = unit.to_canonical_value(array)
            array = array.astype(self.column_type(column_key).value.dataframe_storage_type)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def set_column_values_from_pandas_series(self, column_key: CK, values: pd.Series, unit: Unit) -> None:
        """
        Set column values from a pandas Series with the specified unit.
        
        Args:
            column_key (CK): The column key of the column
            values (pd.Series): The values to set
            unit (Unit): The unit of the provided values
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if unit.UnitQuantity != self._unit_quantities[column_key]:
                raise ValueError(f"Unit {unit} is not compatible with the SI quantity {self._unit_quantities[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = unit.to_canonical_value(values.to_numpy())
            array = array.astype(self.column_type(column_key).value.dataframe_storage_type)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def set_column_values_from_united_array(self, column_key: CK, values: UnitedArray) -> None:
        """
        Set column values from a United_Array.
        
        Args:
            column_key (CK): The column key of the column
            values (United_Array): The values to set
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if values.unit_quantity != self._unit_quantities[column_key]:
                raise ValueError(f"Unit {values.unit_quantity} is not compatible with the unit quantity {self._unit_quantities[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = values.get_as_numpy_array().astype(self.column_type(column_key).value.dataframe_storage_type)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def add_empty_column(self, column_key: CK, unit_quantity: UnitQuantity|Unit, value_type: ColumnType) -> None:
        """
        Add an empty column to the dataframe.
        
        Args:
            column_key (CK): The key for the new column
            UnitQuantity_or_display_unit (UnitQuantity|Unit): The SI quantity or display unit for the column
            value_type (Value_Type): The value type for the column
            
        Raises:
            ValueError: If the dataframe is read-only, the column key already exists,
                                       or the SI quantity and value type are incompatible
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if self.has_column(column_key):
                raise ValueError(f"Column key {column_key} already exists in the dataframe.")
        
            if isinstance(unit_quantity, UnitQuantity):
                unit_quantity: UnitQuantity = unit_quantity
                display_unit: Unit = unit_quantity.canonical_unit()
            elif isinstance(unit_quantity, Unit):
                display_unit: Unit = unit_quantity
                unit_quantity: UnitQuantity = display_unit.unit_quantity
            else:
                raise ValueError(f"Unit quantity {unit_quantity} is not a UnitQuantity or Unit.")

            dataframe_column_name: str = self.create_internal_dataframe_column_name(column_key)
            if dataframe_column_name in self._internal_canonical_dataframe.columns:
                raise ValueError(f"Column key {column_key} already exists in the dataframe.")
            self._internal_canonical_dataframe[dataframe_column_name] = pd.Series([pd.NA] * len(self._internal_canonical_dataframe), dtype=value_type.value.corresponding_pandas_type)
            self._column_information[column_key] = ColumnInformation(unit_quantity, value_type, display_unit)
            self._column_keys.append(column_key)
            self._internal_dataframe_column_strings.append(dataframe_column_name)
            self._unit_quantities[column_key] = unit_quantity
            self._column_types[column_key] = value_type

    def remove_column(self, column_key: CK) -> None:
        """
        Remove a column from the dataframe.
        
        Args:
            column_key (CK): The column key of the column to remove
            
        Raises:
            ValueError: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            self._internal_canonical_dataframe.drop(columns=[dataframe_column_name], inplace=True)
            self._column_information.pop(column_key)
            self._internal_dataframe_column_strings.pop(dataframe_column_name)
            self._column_keys.remove(column_key)
            self._column_information.pop(column_key)
            self._unit_quantities.pop(column_key)
            self._column_types.pop(column_key)

    def remove_columns(self, column_keys: list[CK]|set[CK]) -> None:
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            for column_key in column_keys:
                self.remove_column(column_key)

    def add_column_from_list(self, column_key: CK, values: list, unit: Unit, column_type: ColumnType) -> None:
        """
        Add a new column with data from a list.
        
        Args:
            column_key (CK): The key for the new column
            values (list): The values for the column
            unit (Unit): The unit of the values
            value_type (Value_Type): The value type for the column
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            self.add_empty_column(column_key, unit, column_type)
            self.set_column_values_from_list(column_key, values, unit)
    
    def add_column_from_numpy_array(self, column_key: CK, values: np.ndarray, unit: Unit, column_type: ColumnType) -> None:
        """
        Add a new column with data from a numpy array.
        
        Args:
            column_key (CK): The key for the new column
            values (np.ndarray): The values for the column
            unit (Unit): The unit of the values
            value_type (Value_Type): The value type for the column
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            self.add_empty_column(column_key, unit, column_type)
            self.set_column_values_from_numpy_array(column_key, values, unit)
    
    def add_column_from_pandas_series(self, column_key: CK, values: pd.Series, unit: Unit, column_type: ColumnType) -> None:
        """
        Add a new column with data from a pandas Series.
        
        Args:
            column_key (CK): The key for the new column
            values (pd.Series): The values for the column
            unit (Unit): The unit of the values
            value_type (Value_Type): The value type for the column
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            self.add_empty_column(column_key, unit, column_type)
            self.set_column_values_from_pandas_series(column_key, values, unit)
    
    def add_column_from_array(self, column_key: CK, values: RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray, precision: Literal[32, 64, 128]|None=None) -> None:
        """
        Add a new column with data from a United_Array.
        
        Args:
            column_key (CK): The key for the new column
            values (RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray): The values for the column
            precision (Literal[32, 64, 128]|None): The precision of the values
            
        Raises:
            ValueError: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            column_type: ColumnType = ColumnType.infer_approbiate_column_type(type(values), precision)
            self.add_empty_column(column_key, values.unit_quantity, column_type)
            self.set_column_values_from_array(column_key, values)
            
    def get_iterator_for_column(self, column_key: CK) -> Iterator[UnitedScalar]:
        """
        Get an iterator over the values of a column.

        Args:
            column_key (CK): The column key of the column to get the iterator for

        Returns:
            Iterator[UnitedScalar]: An iterator over the values of the column
        """
        with self._rlock:
            return (self.cell_value_get(row_index, column_key) for row_index in range(len(self._internal_canonical_dataframe)))

    # ----------- Row operations ------------

    def row(self, row_index: int) -> dict[CK, UnitedScalar]:
        """
        Get a row from the dataframe.

        Args:
            index (int): The index of the row to get

        Returns:
            dict[CK, UnitedScalar]: A dictionary of column keys and values for the row
        """
        with self._rlock:
            return {column_key: self.cell_value_get(row_index, column_key) for column_key in self._column_keys}
    
    def iterrows(self) -> Iterator[tuple[int, dict[CK, UnitedScalar]]]:
        """
        Iterate over dataframe rows as (row_index, row_dict) pairs.
        """
        with self._rlock:
            for row_index in range(len(self._internal_canonical_dataframe)):
                yield (row_index, self.row(row_index))

    def get_iterator_for_row(self, row_index: int) -> Iterator[UnitedScalar]:
        """
        Get an iterator over the values of a row.

        Args:
            row_index (int): The index of the row to get the iterator to iterate over the values of that row

        Returns:
            Iterator[UnitedScalar]: An iterator over the values of the row
        """
        with self._rlock:
            return (self.cell_value_get(row_index, column_key) for column_key in self._column_keys)

    def remove_row(self, row: int) -> None:
        """
        Remove a row from the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row < 0 or row >= len(self._internal_canonical_dataframe):
                raise ValueError(f"Row index {row} is out of bounds.")
            self._internal_canonical_dataframe.drop(index=row, inplace=True)

    def remove_all_rows(self) -> None:
        """
        Remove all rows from the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._internal_canonical_dataframe = self._internal_canonical_dataframe.iloc[0:0]

    def remove_rows(self, row_indices: list[int]|set[int]|slice) -> None:
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if isinstance(row_indices, slice):
                self._internal_canonical_dataframe = self._internal_canonical_dataframe.iloc[row_indices]
            else:
                for row_index in sorted(row_indices, reverse=True):
                    self.remove_row(row_index)

    def insert_row(self, row: int, values: list[UnitedScalar]|dict[CK, UnitedScalar]) -> None:
        """
        Insert a row at the specified index.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if row < 0 or row > len(self._internal_canonical_dataframe):
                raise ValueError(f"Row index {row} is out of bounds.")
            if len(values) != len(self._column_keys):
                raise ValueError(f"The number of values ({len(values)}) does not match the number of columns ({len(self._column_keys)}).")
            
            if isinstance(values, list):
                if len(values) != len(self._column_keys):
                    raise ValueError(f"The number of values ({len(values)}) does not match the number of columns ({len(self._column_keys)}).")
                for index, value in enumerate(values):
                    dataframe_column_name: str = self.internal_dataframe_column_string(self._column_keys[index])
                    if self._check_scalar_compatibility(self._column_keys[index], value):
                        self._internal_canonical_dataframe.at[row, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                    else:
                        raise ValueError(f"Value {value} is not compatible with the value type {self.value_type(self._column_keys[index])}.")
            if isinstance(values, dict):
                if any(not self.has_column(key) for key in values.keys()):
                    for key in values.keys():
                        if not self.has_column(key):
                            raise ValueError(f"Column key {key} does not exist in the dataframe.")
                values_to_insert = [] * len(self._column_keys)
                for column_key in self._column_keys:
                    dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
                    if column_key in values:
                        value: UnitedScalar = values[column_key]
                        if self._check_scalar_compatibility(column_key, value):
                            self._internal_canonical_dataframe.at[row, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                        else:
                            raise ValueError(f"Value {value} is not compatible with the value type {self.value_type(column_key)}.")
                    else:
                        self._internal_canonical_dataframe.at[row, dataframe_column_name] = pd.NA

    def add_row(self, values: list[UnitedScalar]|dict[CK, UnitedScalar]) -> None:
        """
        Add a row to the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self.insert_row(len(self._internal_canonical_dataframe), values)

    def add_empty_rows(self, number_of_rows: int=1) -> None:
        """
        Add empty rows to the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            self._internal_canonical_dataframe = pd.concat([self._internal_canonical_dataframe, pd.DataFrame(index=range(len(self._internal_canonical_dataframe), len(self._internal_canonical_dataframe) + number_of_rows))], ignore_index=True)

    # ----------- Cell operations ------------

    def cell_value_is_empty(self, row_index: int, column_key: CK) -> bool:
        """
        Check if a cell is empty.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            return pd.isna(self._internal_canonical_dataframe.at[row_index, dataframe_column_name])

    def cell_value_get(self, row_index: int, column_key: CK) -> SCALAR_TYPE:
        """
        Get the value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Returns:
            RealUnitedScalar|ComplexUnitedScalar|str|bool|Timestamp|float|int: The cell value in the appropriate type
            
        Raises:
            ValueError: If the column doesn't exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if self.cell_value_is_empty(row_index, column_key) and self.column_type(column_key).value.none_value is None:
                raise ValueError(f"The requested cell is empty, but no NA value is defined for the value type {self.column_type(column_key).__name__} of the column {column_key}.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            value: Any = self._internal_canonical_dataframe.at[row_index, dataframe_column_name]
            match self.column_type(column_key):
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    return RealUnitedScalar.create(value, self._display_units[column_key])
                case ColumnType.COMPLEX_NUMBER_128:
                    return ComplexUnitedScalar.create(value, self._display_units[column_key])
                case ColumnType.STRING:
                    return value
                case ColumnType.BOOL:
                    return value
                case ColumnType.TIMESTAMP:
                    return value
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    return value
                case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    return value
                case _:
                    raise ValueError(f"Invalid column value type: {self.column_type(column_key).__name__}")
                    
    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: CK) -> "_ColumnAccessor[CK]":
        """
        Get a column accessor for pandas-like column access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK): The column index or column key
            
        Returns:
            ColumnAccessor[CK]: An accessor object for the specified column
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int) -> "_RowAccessor[CK]":
        """
        Get a column accessor for pandas-like column access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK): The column index or column key
            
        Returns:
            ColumnAccessor[CK]: An accessor object for the specified column
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[CK]|set[CK]) -> "UnitedDataframe[CK]":
        """
        Get a new dataframe with the selected columns.
        
        Args:
            index_or_column_key_or_list_of_keys (list[int|CK]): The column indices or column keys
            
        Returns:
            United_Dataframe[CK]: A new dataframe with the selected columns as a shallow copy
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[int]|set[int]|slice) -> "UnitedDataframe[CK]":
        """
        Get a new dataframe with the selected rows.
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: tuple[int, CK]|tuple[CK, int]) -> SCALAR_TYPE:
        """
        Get a cell value for pandas-like cell access.
        """
        ...

    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int|CK|list[int]|set[int]|slice|list[CK]|set[CK]|tuple[int, CK]|tuple[CK, int]) -> "_ColumnAccessor[CK] | _RowAccessor[CK] | UnitedDataframe[CK] | UnitedScalar":
        """
        Get a column accessor for pandas-like column access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK|list[int|CK]): The column index or column key or a list of column keys
            
        Returns:
            ColumnAccessor[CK] | United_Dataframe[CK]: An accessor object for the specified column or a new dataframe with the selected columns as a shallow copy
            
        Raises:
            ValueError: If the column doesn't exist
        """
        with self._rlock:

            match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position:
                case int():
                    return _RowAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case ColumnKey()|str():
                    return _ColumnAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case slice():
                    new_united_dataframe = self.copy(deep=True)
                    new_united_dataframe.remove_rows(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                    return new_united_dataframe
                case list() | set():
                    if len(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) == 0:
                        return UnitedDataframe[CK].create_empty([], [], [], 0, self._internal_column_name_formatter)
                    if isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), int):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.remove_rows(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) # type: ignore[arg-type]
                        return new_united_dataframe
                    elif isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), ColumnKey|str):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.remove_columns(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) # type: ignore[arg-type]
                        return new_united_dataframe
                    else:
                        raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                case tuple():
                    match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0], column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1]:
                        case int(), ColumnKey()|str():
                            row_index: int = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0] # type: ignore[assignment]
                            column_key: CK = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1] # type: ignore[assignment]
                            return self.cell_value_get(row_index, column_key)
                        case ColumnKey()|str(), int():
                            column_key: CK = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0] # type: ignore[assignment]
                            row_index: int = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1] # type: ignore[assignment]
                            return self.cell_value_get(row_index, column_key)
                        case _:
                            raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                case _:
                    raise ValueError(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")

    def cell_value_set(self, row_index: int, column_key: CK, value: SCALAR_TYPE) -> None:
        """
        Set the value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (UnitedScalar): The value to set
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist,
                                       the row is out of bounds, or the value type is incompatible
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if not 0 <= row_index < len(self._internal_canonical_dataframe):
                raise ValueError(f"Row {row_index} does not exist in the dataframe. The dataframe has {len(self._internal_canonical_dataframe)} rows.")
            if not self._check_scalar_compatibility(column_key, value):
                raise ValueError(f"Value {value} is not compatible with the column type {self.column_type(column_key).__name__} and /or unit of the column {column_key}.")

            column_type: ColumnType = self.column_type(column_key)
            match column_type:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    if isinstance(value, RealUnitedScalar):
                        self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value) # type: ignore[attr-defined]
                    else:
                        raise ValueError(f"Value {value} is not a RealUnitedScalar.")
                case ColumnType.COMPLEX_NUMBER_128:
                    if isinstance(value, ComplexUnitedScalar):
                        self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                    else:
                        raise ValueError(f"Value {value} is not a ComplexUnitedScalar.")
                case ColumnType.STRING:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = column_type.cast_value(value)
                case ColumnType.BOOL:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = column_type.cast_value(value)
                case ColumnType.TIMESTAMP:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = column_type.cast_value(value)
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = column_type.cast_value(value)
    
    def __setitem__(self, cell_position: tuple[int, CK]|tuple[CK, int], value: UnitedScalar):
        """
        Set a cell value using pandas-like syntax.
        
        Args:
            cell_position (tuple[int, CK]|tuple[CK, int]): The cell position
            value (UnitedScalar): The value to set
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist, or the row is out of bounds
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            match cell_position:
                case int(), ColumnKey()|str():
                    row_index: int = cell_position[0]
                    column_key: CK = cell_position[1]
                case ColumnKey()|str(), int():
                    column_key: CK = cell_position[0]
                    row_index: int = cell_position[1]
                case _:
                    raise ValueError(f"Invalid key: {cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not (0 <= row_index < len(self._internal_canonical_dataframe)):
                raise ValueError(f"The row index {row_index} does not exist. The dataframe has {len(self)} rows.")
            self.cell_value_set(row_index, column_key, value)

    # ----------- Column functions operations ------------

    def colfun_min(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> RealUnitedScalar:
        """
        Get the minimum value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The minimum value with appropriate unit information
            
        Raises:
            ValueError: If the column doesn't exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if len(self._internal_canonical_dataframe) == 0:
                return RealUnitedScalar.create(np.nan, self._display_units[column_key])

            match case:
                case "only_positive":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] > 0]
                case "only_negative":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] < 0]
                case "only_non_negative":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] >= 0]
                case "only_non_positive":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] <= 0]
                case "all":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(values) == 0:
                return RealUnitedScalar.create(np.nan, self._display_units[column_key])
            else:
                return RealUnitedScalar.create(np.min(values), self._display_units[column_key])

    def colfun_max(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> RealUnitedScalar:
        """
        Get the maximum value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The maximum value with appropriate unit information
            
        Raises:
            ValueError: If the column doesn't exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if len(self._internal_canonical_dataframe) == 0:
                return RealUnitedScalar.create(np.nan, self._display_units[column_key])
            
            match case:
                case "only_positive":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] > 0]
                case "only_negative":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] < 0]
                case "only_non_negative":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] >= 0]
                case "only_non_positive":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name][self._internal_canonical_dataframe[dataframe_column_name] <= 0]    
                case "all":
                    values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                case _:
                    raise ValueError(f"Invalid case: {case}")
        
            if len(values) == 0:
                return RealUnitedScalar.create(np.nan, self._display_units[column_key])
            else:
                return RealUnitedScalar.create(np.max(values), self._display_units[column_key])

    def colfun_sum(self, column_key: CK) -> RealUnitedScalar:
        """
        Calculate the sum of a numeric column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            UnitedScalar: The sum with appropriate unit information
            
        Raises:
            ValueError: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                return RealUnitedScalar.create(np.sum(values), self._display_units[column_key])
            else:
                raise ValueError(f"Column '{column_key}' is not numeric.")

    def colfun_mean(self, column_key: CK) -> RealUnitedScalar:
        """
        Calculate the mean of a numeric column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            UnitedScalar: The mean with appropriate unit information
            
        Raises:
            ValueError: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                return RealUnitedScalar.create(np.mean(values), self._display_units[column_key])
            else:
                raise ValueError(f"Column '{column_key}' is not numeric.")

    def colfun_std(self, column_key: CK) -> RealUnitedScalar:
        """
        Calculate the standard deviation of a numeric column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            UnitedScalar: The standard deviation with appropriate unit information
            
        Raises:
            ValueError: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                return RealUnitedScalar.create(np.std(values), self._display_units[column_key])
            else:
                raise ValueError(f"Column '{column_key}' is not numeric.")
    
    def colfun_unique(self, column_key: CK) -> list[RealUnitedScalar]|list[ComplexUnitedScalar]|list[str]|list[bool]|list[Timestamp]|list[float]|list[int]:
        """
        Get the unique values of a column.

        Args:
            index_or_column_key (int|CK): The index or column key of the column

        Returns:
            list[UnitedScalar]: The unique values with appropriate unit information
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            unique_values: list[RealUnitedScalar]|list[ComplexUnitedScalar]|list[str]|list[bool]|list[Timestamp]|list[float]|list[int] = []
            
            match self.column_type(column_key):
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    display_unit: Unit = self._display_units[column_key]
                    for value in values.unique():
                        unique_values.append(RealUnitedScalar.create(value, display_unit))
                case ColumnType.COMPLEX_NUMBER_128:
                    display_unit: Unit = self._display_units[column_key]
                    for value in values.unique():
                        unique_values.append(ComplexUnitedScalar.create(value, display_unit))
                case ColumnType.STRING:
                    for value in values.unique():
                        unique_values.append(value)
                case ColumnType.BOOL:
                    for value in values.unique():
                        unique_values.append(value)
                case ColumnType.TIMESTAMP:
                    for value in values.unique():
                        unique_values.append(value)
                case _:
                    raise ValueError(f"Invalid value type: {self.column_type(column_key)}")
            
            return unique_values
        
    def colfun_unique_as_array(self, column_key: CK) -> RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray:
        """
        Get the unique values of a column.

        Args:
            index_or_column_key (int|CK): The index or column key of the column

        Returns:
            list[UnitedScalar]: The unique values with appropriate unit information
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            unique_values: np.ndarray = values.unique()
            
            match self.column_type(column_key):
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    display_unit: Unit = self._display_units[column_key]
                    return RealUnitedArray.create(unique_values, display_unit)
                case ColumnType.COMPLEX_NUMBER_128:
                    display_unit: Unit = self._display_units[column_key]
                    return ComplexUnitedArray.create(unique_values, display_unit)
                case ColumnType.STRING:
                    return StringArray.create(unique_values)
                case ColumnType.BOOL:
                    return BoolArray.create(unique_values)
                case ColumnType.TIMESTAMP:
                    return TimestampArray.create(unique_values)
                case _:
                    raise ValueError(f"Invalid value type: {self.column_type(column_key)}")
            
            return unique_values
        
    def colfun_smallest_positive_nonzero_value(self, column_key: CK) -> UnitedScalar:
        """
        Get the smallest positive non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            positive_nonzero_values: pd.Series = values[(values > 0) & values.notna()]
            if len(positive_nonzero_values) == 0:
                return UnitedScalar.united_number(np.nan, self._display_units[column_key])
            return UnitedScalar.united_number(np.min(positive_nonzero_values), self._display_units[column_key])
        
    def colfun_largest_positive_nonzero_value(self, column_key: CK) -> UnitedScalar:
        """
        Get the largest positive non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            positive_nonzero_values: pd.Series = values[(values > 0) & values.notna()]
            if len(positive_nonzero_values) == 0:
                return UnitedScalar.united_number(np.nan, self._display_units[column_key])
            return UnitedScalar.united_number(np.max(positive_nonzero_values), self._display_units[column_key])

    def colfun_largest_negative_nonzero_value(self, column_key: CK) -> UnitedScalar:
        """
        Get the largest negative non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            negative_nonzero_values: pd.Series = values[(values < 0) & values.notna()]
            if len(negative_nonzero_values) == 0:
                return UnitedScalar.united_number(np.nan, self._display_units[column_key])
            return UnitedScalar.united_number(np.max(negative_nonzero_values), self._display_units[column_key])
        
    def colfun_smallest_negative_nonzero_value(self, column_key: CK) -> UnitedScalar:
        """
        Get the smallest negative non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            negative_nonzero_values: pd.Series = values[(values < 0) & values.notna()]
            if len(negative_nonzero_values) == 0:
                return UnitedScalar.united_number(np.nan, self._display_units[column_key])
            return UnitedScalar.united_number(np.min(negative_nonzero_values), self._display_units[column_key])
    
    @overload
    def colfun_count_value_occurances(self, column_key: CK) -> dict[UnitedScalar, int]:
        """Count the number of occurrences of each unique value in the column."""
        ...
    @overload
    def colfun_count_value_occurances(self, column_key: CK, value_to_count: UnitedScalar) -> int:
        """Count the number of occurrences of the specified value in the column."""
        ...
    def colfun_count_value_occurances(self, column_key: CK, value_to_count: SCALAR_TYPE|None = None) -> dict[UnitedScalar, int]|int:
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            column_type: ColumnType = self.column_type(column_key)
            display_unit: Unit|None = self._display_units[column_key]
            if value_to_count is None:
                unique_values: np.ndarray = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)].unique()
                column_as_pd_series: pd.Series = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)]
                occurance_counts_dict: dict[UnitedScalar, int] = {}
                for value in unique_values:
                    occurance_count: int = column_as_pd_series.eq(value).sum()
                    value_key: SCALAR_TYPE = column_type.create_scalar_from_value(value, display_unit)
                    occurance_counts_dict[value_key] = occurance_count
                return occurance_counts_dict
            else:
                if not self.compatible_with_column(column_key, value_to_count):
                    raise ValueError(f"The value {value_to_count} is not compatible with the column {column_key}.")
                value_casted_for_dataframe: Any = column_type.cast_for_dataframe(value_to_count)
                column_values: pd.Series = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)]                
                occurance_count: int = column_values.eq(value_casted_for_dataframe).sum()
                return occurance_count
            
    def colfun_row_index(self, column_key: CK, value: UnitedScalar, case: Literal["first", "last"] = "first") -> int:
        """
        Get the row index of the first occurrence of a value in a column.

        Args:
            column_key (CK): The column key of the column
            value (UnitedScalar): The value to get the row index of
            case (Literal["first", "last"]): The case to get the row index of

        Returns:
            int: The row index of the first occurrence of the value in the column. Returns -1 if the value is not found.
        """
        with self._rlock:
            column_type: ColumnType = self.column_type(column_key)
            value_casted_for_dataframe: Any = column_type.cast_for_dataframe(value)
            column_values: pd.Series = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)]
            row_index: int|str
            try:
                match case:
                    case "first":
                        row_index = column_values.eq(value_casted_for_dataframe).idxmax()
                    case "last":
                        row_index = column_values.eq(value_casted_for_dataframe).idxmin()
                    case _:
                        raise ValueError(f"Invalid case: {case}")
            except ValueError:
                return -1
            if isinstance(row_index, str):
                raise ValueError(f"The row index of the value {value} in the column {column_key} is not a number. It appears the index of the internal dataframe has been set.")
            else:
                return int(row_index)

    # ----------- Mask operations ------------

    def maskfun_isna(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are NA/NaN.
        
        This method works similarly to pandas DataFrame.isna(), returning a numpy array
        with the same shape as the original, where True indicates NA/NaN values.
        
        Args:
            subset (list[CK] | None): List of column keys to check for NA values.
                                    If None, checks all columns.
        
        Returns:
            np.ndarray: Boolean array with same shape as original, where True indicates NA values
            
        Examples:
            # Check all columns for NA values
            na_mask = df.isna()
            
            # Check specific columns for NA values
            na_mask = df.isna(['column1', 'column2'])
            
            # Use the mask for filtering
            non_na_rows = df[~df.isna().any(axis=1)]
        """
        with self._rlock:
            if subset is None:
                columns_to_check = self._column_keys
            else:
                # Validate that all subset columns exist
                for col in subset:
                    if not self.has_column(col):
                        raise ValueError(f"Column key {col} does not exist in the dataframe.")
                columns_to_check = subset
            
            # Create result array with same shape
            result_array = np.zeros((len(self._internal_canonical_dataframe), len(columns_to_check)), dtype=bool)
            
            for i, column_key in enumerate(columns_to_check):
                dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
                value_type = self.value_type(column_key)
            
                # Get the pandas Series for this column
                column_series = self._internal_canonical_dataframe[dataframe_column_name]
                
                # Check for NA values using pandas isna
                result_array[:, i] = pd.isna(column_series).values
            
            return result_array

    def maskfun_notna(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are not NA/NaN.
        
        This is the inverse of isna() - returns True for non-NA values.
        
        Args:
            subset (list[CK] | None): List of column keys to check for non-NA values.
                                    If None, checks all columns.
        
        Returns:
            np.ndarray: Boolean array with same shape as original, where True indicates non-NA values
            
        Examples:
            # Check all columns for non-NA values
            non_na_mask = df.notna()
            
            # Check specific columns for non-NA values
            non_na_mask = df.notna(['column1', 'column2'])
            
            # Use the mask for filtering
            non_na_rows = df[df.notna().all(axis=1)]
        """
        return ~self.maskfun_isna(subset)
    
    class _UnitedScalar_Proxy:
        def __init__(self, symbol: str = "x"):
            self.symbol = symbol

        def _wrap(self, op, other):
            if isinstance(other, UnitedScalar):
                val = other.canonical_float
                # Return a numpy-compatible ufunc style lambda
                return lambda x: op(x, val)
            raise TypeError("Only comparisons against UnitedScalar are supported.")

        def __gt__(self, other): return self._wrap(operator.gt, other)
        def __ge__(self, other): return self._wrap(operator.ge, other)
        def __lt__(self, other): return self._wrap(operator.lt, other)
        def __le__(self, other): return self._wrap(operator.le, other)
        def __eq__(self, other): return self._wrap(operator.eq, other)
        def __ne__(self, other): return self._wrap(operator.ne, other)

    def _convert_filter(self, filter_function: Callable[[UnitedScalar], bool]) -> Callable[[np.ndarray], np.ndarray]:
        proxy: UnitedDataframe._UnitedScalar_Proxy = UnitedDataframe._UnitedScalar_Proxy()
        scalar_filter: bool = filter_function(proxy) # type: ignore[arg-type]

        if not callable(scalar_filter):
            raise TypeError("The filter_function must return a callable when evaluated with proxy input.")

        def ufunc_filter(arr: np.ndarray) -> np.ndarray:
            return scalar_filter(arr)

        return ufunc_filter

    def maskfun_get_from_filter(
        self,
        column_key_and_callable: dict[CK, Callable[[UnitedScalar], bool] | Callable[[str], bool]]
    ) -> np.ndarray:
        """
        Return a boolean mask of rows that satisfy all filter functions on selected columns.

        Args:
            column_key_and_callable (dict[CK, Callable[[UnitedScalar], bool] | Callable[[str], bool]]): A dictionary of column keys and filter functions.

        Returns:
            np.ndarray: A boolean mask of rows that satisfy all filter functions on selected columns.
        """

        with self._rlock:
            mask_of_dataframe = np.ones(self.rows, dtype=bool)

            for column_key, filter_function in column_key_and_callable.items():
                column_type: ColumnType = self.column_type(column_key)

                if column_type.is_numeric:
                    # Numeric column
                    column_array = self.column_values_as_numpy_array(
                        column_key,
                        self.display_unit(column_key)
                    )
                    float_mask_func = self._convert_filter(filter_function)  # type: ignore[arg-type]
                    mask_of_column = float_mask_func(column_array)

                elif column_type == ColumnType.STRING:
                    # String column
                    string_array = self._internal_canonical_dataframe[
                        self.internal_dataframe_column_string(column_key)
                    ].to_numpy().ravel()

                    vectorized = np.frompyfunc(filter_function, 1, 1)
                    mask_of_column = vectorized(string_array).astype(bool)  # type: ignore[assignment]

                else:
                    raise ValueError(f"Unsupported value type for column {column_key}.")

                mask_of_dataframe &= mask_of_column

            return mask_of_dataframe
        
    def maskfun_apply_mask(self, mask: np.ndarray) -> "UnitedDataframe[CK]":
        """
        Get a new dataframe with the rows that satisfy the numpy mask.

        Args:
            mask (np.ndarray): The numpy mask. The mask must have the same number of columns as the dataframe and the same number of rows as the dataframe. The mask must be a boolean array.

        Returns:
            United_Dataframe[CK]: A new dataframe with the rows that satisfy the numpy mask

        Raises:
        """
        with self._rlock:

            # Check the dimensions of the mask
            if mask.ndim != self.cols:
                raise ValueError(f"The mask must have the same number of columns as the dataframe. The mask has {mask.ndim} columns, but the dataframe has {self.cols} columns.")

            # Check the length of the mask
            if len(mask) != self.rows:
                raise ValueError(f"The mask must have the same number of rows as the dataframe. The mask has {len(mask)} rows, but the dataframe has {self.rows} rows.")

            # Check the type of the mask
            if mask.dtype != bool:
                raise ValueError(f"The mask must be a boolean array. The mask has dtype {mask.dtype}.")

            # Create a new dataframe with the rows that satisfy the mask
            return UnitedDataframe[CK].create_from_dataframe_and_column_information_list(
                self._internal_canonical_dataframe[mask],
                None,
                self.get_column_information_list(),
                self._internal_column_name_formatter)

    # ----------- Other operations ------------

    def get_numeric_column_keys(self) -> list[CK]:
        """
        Get a list of column keys for numeric columns only.
        
        Returns:
            list[CK]: List of column keys for numeric columns
        """
        with self._rlock:
            return [column_key for column_key in self._column_keys if self.is_numeric(column_key)]

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

    def rowfun_head(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the first n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 1)
            
        Returns:
            United_Dataframe: A new dataframe containing the first n rows
            
        Raises:
            ValueError: If n is negative
            
        Examples:
            # Get first 1 rows (default)
            df.head()
            
            # Get first 10 rows
            df.head(10)
            
            # Get all rows if n is larger than dataframe size
            df.head(100)  # Returns all rows if dataframe has fewer than 100 rows
        """
        with self._rlock:
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            
            # If n is larger than the dataframe size, return all rows
            actual_n = min(n, len(self._internal_canonical_dataframe))
            head_df = self._internal_canonical_dataframe.head(actual_n)
            
            return UnitedDataframe[CK](
                head_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def rowfun_first(self) -> "UnitedDataframe[CK]":
        """
        Get the first row of the dataframe.
        
        Returns:
            United_Dataframe: A new dataframe containing only the first row
            
        Raises:
            ValueError: If the dataframe is empty
            
        Examples:
            # Get the first row
            first_row = df.first()
            
            # Access the first row's values
            first_row.loc[0, 'column_name']
        """
        with self._rlock:
            if self.empty:
                raise ValueError("Cannot get first row from an empty dataframe")
            
            first_df = self._internal_canonical_dataframe.head(1)
            
            return UnitedDataframe(
                first_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def rowfun_tail(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the last n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe: A new dataframe containing the last n rows
            
        Raises:
            ValueError: If n is negative
            
        Examples:
            # Get last 1 rows (default)
            df.tail()
            
            # Get last 10 rows
            df.tail(10)
            
            # Get all rows if n is larger than dataframe size
            df.tail(100)  # Returns all rows if dataframe has fewer than 100 rows
        """
        with self._rlock:
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            
            # If n is larger than the dataframe size, return all rows
            actual_n = min(n, len(self._internal_canonical_dataframe))
            tail_df = self._internal_canonical_dataframe.tail(actual_n)
            
            return UnitedDataframe(
                tail_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def rowfun_last(self) -> "UnitedDataframe[CK]":
        """
        Get the last row of the dataframe.
        
        Returns:
            United_Dataframe: A new dataframe containing only the last row
            
        Raises:
            ValueError: If the dataframe is empty
            
        Examples:
            # Get the last row
            last_row = df.last()
            
            # Access the last row's values
            last_row.loc[0, 'column_name']
        """
        with self._rlock:
            if self.empty:
                raise ValueError("Cannot get last row from an empty dataframe")
            
            last_df = self._internal_canonical_dataframe.tail(1)
            
            return UnitedDataframe(
                last_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

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
                    dtype = self._value_types[column_key].value.corresponding_pandas_type
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

    def dropna(self, subset: list[CK] | None = None) -> "UnitedDataframe[CK]":
        """
        Remove rows with missing values.
        
        Args:
            subset (list[CK] | None): Optional list of columns to check for missing values. If None, all columns are checked.
            
        Returns:
            United_Dataframe: A new dataframe with rows containing missing values removed
        """
        with self._rlock:
            dataframe_column_names: list[str] = self.internal_dataframe_column_strings(subset) if subset is not None else self.internal_dataframe_column_strings()
            cleaned_df = self._internal_canonical_dataframe.dropna(subset=dataframe_column_names)
            
            return UnitedDataframe(
                cleaned_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter)

    def fillna(self, value: UnitedScalar, subset: list[CK] | None = None) -> "UnitedDataframe[CK]":
        raise NotImplementedError("Not implemented")

    @classmethod
    def dataframes_can_concatenate(cls, *dataframes: "UnitedDataframe[CK]") -> bool:
        """
        Check if multiple dataframes can be concatenated.
        
        Args:
            *dataframes: Variable number of United_Dataframe instances to check
            
        Returns:
            bool: True if all dataframes have compatible column structures, False otherwise
        """
        sorted_dataframes = sorted(dataframes, key=lambda df: id(df._rlock))
        with ExitStack() as stack:
            for df in sorted_dataframes:
                stack.enter_context(df._lock._rlock) # type: ignore
            first_keys = sorted_dataframes[0].column_keys
            for dataframe in sorted_dataframes:
                if dataframe.column_keys != first_keys:
                    return False
        return True
    
    @classmethod
    def concatenate_dataframes(cls, dataframe: "UnitedDataframe[CK]", *dataframes: "UnitedDataframe[CK]") -> "UnitedDataframe[CK]":
        """
        Concatenate multiple dataframes vertically.
        
        Args:
            dataframe: The first dataframe
            *dataframes: Additional dataframes to concatenate
            
        Returns:
            United_Dataframe: A new dataframe containing all rows from the input dataframes
            
        Raises:
            ValueError: If the dataframes have incompatible column structures
        """
        sorted_dataframes = sorted(dataframes, key=lambda df: id(df._rlock))
        with ExitStack() as stack:
            for df in sorted_dataframes:
                stack.enter_context(df._lock._rlock) # type: ignore
            first_keys = dataframes[0].column_keys
            for dataframe in dataframes:
                if dataframe.column_keys != first_keys:
                    raise ValueError(f"The dataframes cannot be concatenated. The column names or the units are not compatible.")
        
        concatenated_dataframe: pd.DataFrame = dataframe.internal_dataframe_deepcopy
        for df in dataframes:
            concatenated_dataframe = pd.concat([concatenated_dataframe, df.internal_dataframe_deepcopy], ignore_index=True)
        return cls(
            concatenated_dataframe,
            dataframe._column_information,
            dataframe._internal_dataframe_column_name_formatter)

    def mask_in_range(self, column_key: CK, min_val: UnitedScalar, max_val: UnitedScalar) -> np.ndarray:
        """
        Check which rows have values in the specified range.
        
        Args:
            column_key (CK): The column key to check
            min_val (UnitedScalar): The minimum value (inclusive)
            max_val (UnitedScalar): The maximum value (inclusive)
            
        Returns:
            np.ndarray: Boolean array where True indicates values in range [min_val, max_val]
            
        Raises:
            ValueError: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                return values.between(min_val.canonical_value, max_val.canonical_value).to_numpy()
            else:
                raise ValueError(f"Column '{column_key}' is not numeric.")

    def colfun_clamp_column(self, column_key: CK, min_val: UnitedScalar, max_val: UnitedScalar, inclusive: Literal["both", "neither", "left", "right"] = "both") -> None:
        """
        Clamp all values in a column to the specified range.
        
        Args:
            column_key (CK): The column key to clamp
            min_val (UnitedScalar): The minimum value
            max_val (UnitedScalar): The maximum value
            inclusive (str): Which bounds to include ("both", "neither", "left", "right")
            
        Raises:
            ValueError: If the dataframe is read-only, the column doesn't exist,
                                       or the column is not numeric
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            if not self.has_column(column_key):
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                clamped_values = values.clip(lower=min_val.canonical_float, upper=max_val.canonical_float, inclusive=inclusive)
                self._internal_canonical_dataframe[dataframe_column_name] = clamped_values
            else:
                raise ValueError(f"Column '{column_key}' is not numeric.")

    # ----------- Column and Row Filtering ------------

    def filterfun_get_by_column_key_types(self, *column_key_types: type[CK_CF]) -> "UnitedDataframe[CK_CF]":
        """
        Filter the dataframe by column key type.
        """
        with self._rlock:

            column_information_of_type: list[ColumnInformation[CK_CF]] = self.column_information_of_type(*column_key_types)
            return UnitedDataframe[CK_CF].create_from_dataframe_and_column_information_list(
                self._internal_canonical_dataframe,
                None,
                column_information_of_type,
                self._internal_column_name_formatter,
                True)
        
    def filterfun_inplace_by_column_key_types(self, *column_key_types: type[CK]) -> None:
        """
        Filter the dataframe by column key type in place.
        """
        with self._wlock:
            if self._read_only:
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            column_keys_to_keep: list[CK] = self.column_keys_of_type(*column_key_types)
            column_keys_to_remove: list[CK] = [column_key for column_key in self._column_keys if column_key not in column_keys_to_keep]
            for column_key in column_keys_to_remove:
                self.remove_column(column_key)

    def filterfun_by_filterdict(self, filter_dict: dict[CK, SCALAR_TYPE]) -> "UnitedDataframe[CK]":
        """
        Filter the dataframe by a dictionary of column keys and values.

        Args:
            filter_dict (dict[CK, UnitedScalar|str|bool]): A dictionary of column keys and values to filter by

        Returns:
            United_Dataframe[CK]: A new dataframe with the filtered rows.

        Raises:
            ValueError: If the column key does not exist in the dataframe.
        """
        with self._rlock:
            filtered_df = self._internal_canonical_dataframe.copy()
            for column_key, value in filter_dict.items():
                if self.has_column(column_key):
                    column_type: ColumnType = self.column_type(column_key)
                    value_casted_for_dataframe: Any = column_type.cast_for_dataframe(value)
                    filtered_df = filtered_df[filtered_df[self.internal_dataframe_column_string(column_key)] == value_casted_for_dataframe]
                else:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")

            return self._get_dataframe_with_new_canonical_dataframe(filtered_df)

    # ----------- Serialization ------------

    def to_json(self) -> dict:
        """
        Serialize the United_Dataframe to a JSON-serializable dict.
        
        Returns:
            dict: JSON-serializable representation of the dataframe
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Serialization to JSON is not implemented yet.")

    @classmethod
    def from_json(cls, data: dict) -> "UnitedDataframe":
        """
        Deserialize a United_Dataframe from a JSON-serializable dict.
        
        Args:
            data (dict): JSON-serializable representation of the dataframe
            
        Returns:
            United_Dataframe: A new dataframe instance
            
        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("Deserialization from JSON is not implemented yet.")
    
    def to_hdf5(self, hdf5_group: Group) -> None:
        hdf5_group.create_dataset("internal_canonical_dataframe", data=self._internal_canonical_dataframe)
    
    @classmethod
    def from_hdf5(
        cls,
        hdf5_group: Group,
        retrieve_column_key_callable: Callable[[str], CK]|None = None,
        internal_column_name_formatter: InternalDataFrameNameFormatter[CK] = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER) -> "UnitedDataframe[CK]":

        dataframe: pd.DataFrame = pd.read_hdf(hdf5_group["internal_canonical_dataframe"])

        column_information: dict[CK, ColumnInformation[CK]] = {}
        for internal_column_name in dataframe.columns:
            d_type: Dtype = dataframe.dtypes[internal_column_name]
            if retrieve_column_key_callable is None:
                # Assuming that the column keys are strings
                column_key, column_information = internal_column_name_formatter.retrieve_from_internal_dataframe_column_name(internal_column_name, d_type, lambda x: x)
            else:
                column_key, column_information = internal_column_name_formatter.retrieve_from_internal_dataframe_column_name(internal_column_name, d_type, retrieve_column_key_callable)
            column_information[column_key] = column_information

        united_dataframe: UnitedDataframe[CK] = cls(
            dataframe,
            column_information,
            internal_column_name_formatter,
            deepcopy_dataframe=False
        )

        return united_dataframe
    
    # ----------- Constructors ------------

    @classmethod
    def create_from_pandas_dataframe_and_column_information(
        cls,
        dataframe: pd.DataFrame,
        dataframe_column_names: dict[CK, str],
        column_information: dict[CK, ColumnInformation[CK]],
        internal_column_name_formatter: InternalDataFrameNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
        deepcopy_dataframe: bool = True) -> "UnitedDataframe[CK]":

        # Rename the columns
        renaming_dict: dict[str, str] = {dataframe_column_name: internal_column_name_formatter(column_key, column_information[column_key].display_unit, column_information[column_key].value_type) for column_key, dataframe_column_name in dataframe_column_names.items()}
        dataframe.rename(columns=renaming_dict, inplace=True)

        # Create the United_Dataframe
        return UnitedDataframe[CK](
            dataframe,
            column_information,
            internal_column_name_formatter,
            dataframe.copy(deep=deepcopy_dataframe))

    @classmethod
    def create_from_pandas_dataframe(
        cls,
        dataframe: pd.DataFrame,
        dataframe_column_names: dict[CK, str],
        column_keys: list[CK] = [],
        units: list[Unit]|dict[CK, Unit] = [],
        value_types: list[ColumnType]|dict[CK, ColumnType] = [],
        internal_column_name_formatter: InternalDataFrameNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER,
        deepcopy_dataframe: bool = True,
        ) -> "UnitedDataframe[CK]":

        # Check that the number of columns is the same for all inputs
        number_of_columns: int = len(column_keys)
        if number_of_columns != len(dataframe_column_names) or number_of_columns != len(units) or number_of_columns != len(value_types):
            raise ValueError(f"The number of column keys, units, and value types must be the same. Got {number_of_columns} columns, {len(dataframe_column_names)} dataframe column names, {len(units)} units, and {len(value_types)} value types.")
        
        # Create the column information dict
        column_information: dict[CK, ColumnInformation[CK]] = {column_key: ColumnInformation(unit, column_type) for column_key, unit, column_type in zip(column_keys, units, value_types)}

        # Create the United_Dataframe
        return cls.create_from_pandas_dataframe_and_column_information(
            dataframe,
            dataframe_column_names,
            column_information,
            internal_column_name_formatter,
            deepcopy_dataframe)
    
    def create_empty(self) -> "UnitedDataframe[CK]":
        return UnitedDataframe[CK](
            pd.DataFrame(data={col: [pd.NA] * 0 for col in self._column_keys}),
            self._column_information,
            self._internal_dataframe_column_name_formatter
        )

    @classmethod
    def create_empty(
        cls,
        column_keys: list[CK] = [],
        units: list[Unit]|dict[CK, Unit] = [],
        column_types: list[ColumnType]|dict[CK, ColumnType] = [],
        initial_number_of_rows: int = 0,
        internal_column_name_formatter: InternalDataFrameNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
        ) -> "UnitedDataframe[CK]":

        column_information: dict[CK, ColumnInformation[CK]] = {column_key: ColumnInformation(unit, column_type) for column_key, unit, column_type in zip(column_keys, units, column_types)}
        column_names: list[str] = [internal_column_name_formatter.create_internal_dataframe_column_name(column_key, column_information[column_key]) for column_key in column_keys]
        empty_dataframe: pd.DataFrame = pd.DataFrame(data={col: [pd.NA] * initial_number_of_rows for col in column_names})

        return cls.create_from_pandas_dataframe(
            empty_dataframe,
            column_names,
            column_information,
            internal_column_name_formatter)
    
    def create_empty_from_column_information(
        self,
        column_information: dict[CK, ColumnInformation[CK]],
        initial_number_of_rows: int = 0,
        internal_column_name_formatter: InternalDataFrameNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
    ) -> "UnitedDataframe[CK]":
        
        column_names: list[str] = [internal_column_name_formatter.create_internal_dataframe_column_name(column_key, column_information[column_key]) for column_key in column_information.keys()]
        empty_dataframe: pd.DataFrame = pd.DataFrame(data={col: [pd.NA] * initial_number_of_rows for col in column_names})

        return self.create_from_pandas_dataframe_and_column_information(
            empty_dataframe,
            column_names,
            column_information,
            internal_column_name_formatter)
    
    def create_from_row_values_and_column_information(
        self,
        row_values: list[dict[CK, SCALAR_TYPE]],
        column_information: dict[CK, ColumnInformation[CK]],
        initial_number_of_rows: int = 0,
        internal_column_name_formatter: InternalDataFrameNameFormatter = SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
    ) -> "UnitedDataframe[CK]":
        
        column_names: list[str] = [internal_column_name_formatter.create_internal_dataframe_column_name(column_key, column_information[column_key]) for column_key in column_information.keys()]
        empty_dataframe: pd.DataFrame = pd.DataFrame(data={col: [pd.NA] * initial_number_of_rows for col in column_names})

        for row_index, row_value_dict in enumerate(row_values):
            for column_key, value in row_value_dict.items():
                column_type: ColumnType = column_information[column_key].column_type
                empty_dataframe.at[row_index, column_key] = column_type.cast_for_dataframe(value)

        return self.create_from_pandas_dataframe_and_column_information(
            empty_dataframe,
            column_names,
            column_information,
            internal_column_name_formatter)

    # ----------- GroupBy functionality ------------

    def groupby(self, by: CK|list[CK]|set[CK]) -> "GroupBy[CK]":
        """
        Group the dataframe by one or more columns.
        
        Args:
            by: Column index, column key, or list of columns to group by
            
        Returns:
            GroupBy: A GroupBy object for performing grouped operations
            
        Raises:
            ValueError: If any of the grouping columns don't exist
            
        Examples:
            # Group by a single column
            grouped = df.groupby('category')
            
            # Group by multiple columns
            grouped = df.groupby(['category', 'region'])
            
            # Group by column index
            grouped = df.groupby(0)
        """
        with self._rlock:
            if isinstance(by, (ColumnKey, str)):
                by = [by]
            
            # Validate all grouping columns exist
            for col in by:
                if not self.has_column(col):
                    raise ValueError(f"Grouping column {col} does not exist in the dataframe.")
            
            return GroupBy(self, by)