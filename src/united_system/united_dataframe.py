from .units.unit import Unit, NO_NUMBER, SI_QUANTITY, NUMBER
from .united_scalar import United_Value, UnitedValueValueType, create_united_value_unsafely
import pandas as pd
import numpy as np
from typing import NamedTuple, Literal, Set, Type, Protocol, runtime_checkable, Callable, Generic, TypeVar, Iterator, overload, Union, cast, Optional
from datetime import datetime
from pandas._typing import Dtype
from dataclasses import dataclass
from .united_array import UnitedArray, Value_Type as United_Array_Value_Type
from enum import Enum, auto
from contextlib import ExitStack
from readerwriterlock import rwlock
import math
from typing import Any
import operator

@runtime_checkable
class JSONable(Protocol):
    def to_json(self) -> dict:
        ...
    @classmethod
    def from_json(cls: Type['JSONable'], data: dict) -> 'JSONable':
        ...

@runtime_checkable
class Column_Key(Protocol):
    def to_string(self) -> str:
        ...
    @classmethod
    def from_string(cls: Type['Column_Key'], string: str) -> 'Column_Key':
        ...

class Series_With_Unit(NamedTuple):
    series: pd.Series
    unit: Unit

@dataclass(frozen=True, slots=True)
class Value_Type_Information(NamedTuple):
    name: str
    corresponding_pandas_type: Dtype
    corresponding_python_type: type
    corresponding_numpy_dtype: np.dtype
    corresponding_united_array_value_type: United_Array_Value_Type
    corresponding_united_value_type: type[UnitedValueValueType]
    corresponding_python_na_value: None|float
    corresponding_numpy_na_value: float|np.float32|np.datetime64
    corresponding_united_value_na_value: None|UnitedValueValueType
    corresponding_united_value_na_value_type: None|type[UnitedValueValueType]
    is_numeric: bool
    is_non_numeric: bool
    precision: int|None

class Value_Type(Enum):
    value: Value_Type_Information

    FLOAT64 = Value_Type_Information(   name="float64",  corresponding_python_type=float,    corresponding_pandas_type=pd.Float64Dtype(),          corresponding_numpy_dtype=np.dtype("float64"),        corresponding_united_array_value_type=United_Array_Value_Type.FLOAT64,    corresponding_united_value_type=float,    is_numeric=True,  is_non_numeric=False, precision=64,   corresponding_python_na_value=math.nan, corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=math.nan, corresponding_united_value_na_value_type=float)
    FLOAT32 = Value_Type_Information(   name="float32",  corresponding_python_type=float,    corresponding_pandas_type=pd.Float32Dtype(),          corresponding_numpy_dtype=np.dtype("float32"),        corresponding_united_array_value_type=United_Array_Value_Type.FLOAT32,    corresponding_united_value_type=float,    is_numeric=True,  is_non_numeric=False, precision=32,   corresponding_python_na_value=math.nan, corresponding_numpy_na_value=np.float32(np.nan),   corresponding_united_value_na_value=math.nan, corresponding_united_value_na_value_type=float)
    INT64 = Value_Type_Information(     name="int64",    corresponding_python_type=int,      corresponding_pandas_type=pd.Int64Dtype(),            corresponding_numpy_dtype=np.dtype("int64"),          corresponding_united_array_value_type=United_Array_Value_Type.INT64,      corresponding_united_value_type=float,    is_numeric=True,  is_non_numeric=False, precision=64,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)
    INT32 = Value_Type_Information(     name="int32",    corresponding_python_type=int,      corresponding_pandas_type=pd.Int32Dtype(),            corresponding_numpy_dtype=np.dtype("int32"),          corresponding_united_array_value_type=United_Array_Value_Type.INT32,      corresponding_united_value_type=float,    is_numeric=True,  is_non_numeric=False, precision=32,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)
    INT16 = Value_Type_Information(     name="int16",    corresponding_python_type=int,      corresponding_pandas_type=pd.Int16Dtype(),            corresponding_numpy_dtype=np.dtype("int16"),          corresponding_united_array_value_type=United_Array_Value_Type.INT16,      corresponding_united_value_type=float,    is_numeric=True,  is_non_numeric=False, precision=16,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)
    INT8 = Value_Type_Information(      name="int8",     corresponding_python_type=int,      corresponding_pandas_type=pd.Int8Dtype(),             corresponding_numpy_dtype=np.dtype("int8"),           corresponding_united_array_value_type=United_Array_Value_Type.INT8,       corresponding_united_value_type=float,    is_numeric=True,  is_non_numeric=False, precision=8,    corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)
    BOOLEAN = Value_Type_Information(   name="bool",     corresponding_python_type=bool,     corresponding_pandas_type=pd.BooleanDtype(),          corresponding_numpy_dtype=np.dtype("bool"),           corresponding_united_array_value_type=United_Array_Value_Type.BOOLEAN,    corresponding_united_value_type=bool,     is_numeric=False, is_non_numeric=True,  precision=None, corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)
    STRING = Value_Type_Information(    name="string",   corresponding_python_type=str,      corresponding_pandas_type=pd.StringDtype(),           corresponding_numpy_dtype=np.dtype("string"),         corresponding_united_array_value_type=United_Array_Value_Type.STRING,     corresponding_united_value_type=str,      is_numeric=False, is_non_numeric=True,  precision=None, corresponding_python_na_value=None,     corresponding_numpy_na_value=np.nan,               corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)
    DATETIME64 = Value_Type_Information(name="datetime", corresponding_python_type=datetime, corresponding_pandas_type=np.dtype("datetime64[ns]"), corresponding_numpy_dtype=np.dtype("datetime64[ns]"), corresponding_united_array_value_type=United_Array_Value_Type.DATETIME64, corresponding_united_value_type=datetime, is_numeric=True,  is_non_numeric=False,  precision=64,   corresponding_python_na_value=None,     corresponding_numpy_na_value=np.datetime64("NaT"), corresponding_united_value_na_value=None,     corresponding_united_value_na_value_type=None)

    @classmethod
    def find_value_type_by_united_value_type(cls, united_value_type: United_Array_Value_Type) -> "Value_Type":
        """
        Find the value type by united value type.

        Args:
            united_value_type (United_Array_Value_Type): The united value type to find the value type for

        Returns:
            Value_Type: The value type for the given united value type
        """
        for value_type in Value_Type:
            if value_type.value.corresponding_united_array_value_type == united_value_type:
                return value_type
        raise United_Dataframe_Exception(f"No value type found for united value type {united_value_type}.")
    
    @classmethod
    def find_value_type_by_pandas_type(cls, pandas_type: Dtype) -> "Value_Type":
        """
        Find the value type by pandas type.

        Args:
            pandas_type (Dtype): The pandas type to find the value type for

        Returns:
            Value_Type: The value type for the given pandas type
        """
        for value_type in cls:
            vtype = value_type.value.corresponding_pandas_type
            # Use string representation matching for extension types and fallback to equality for others
            if str(vtype) == str(pandas_type):
                return value_type
        raise United_Dataframe_Exception(f"No value type found for pandas type {pandas_type}.")
    
    @classmethod
    def find_value_type_by_numpy_dtype(cls, numpy_dtype: np.dtype) -> "Value_Type":
        """
        Find the value type by numpy dtype.

        Args:
            numpy_dtype (np.dtype): The numpy dtype to find the value type for

        Returns:
            Value_Type: The value type for the given numpy dtype
        """
        dtype: np.dtype = np.dtype(numpy_dtype)  # normalize dtype
        for value_type in cls:
            if value_type.value.corresponding_numpy_dtype == dtype:
                return value_type
        raise United_Dataframe_Exception(f"No value type found for numpy dtype {numpy_dtype}.")
    
    def coerce_to_type(self, value: float | int | str | bool | datetime) -> float | int | str | bool | datetime:
        try:
            return self.value.corresponding_python_type(value)
        except Exception as e:
            raise TypeError(f"Cannot coerce {value!r} to {self.value.name}") from e
        
    def has_python_na_value(self) -> bool:
        return self.value.corresponding_python_na_value is not None
    
    def has_numpy_na_value(self) -> bool:
        return self.value.corresponding_numpy_na_value is not None
    
    def has_united_value_na_value(self) -> bool:
        return self.value.corresponding_united_value_na_value is not None
    
    def has_united_value_na_value_type(self) -> bool:
        return self.value.corresponding_united_value_na_value_type is not None

CK = TypeVar("CK", bound=Column_Key|str, default=str)
CK_I2 = TypeVar("CK_I2", bound=Column_Key|str, default=str)

CK_CF = TypeVar("CK_CF", bound=Column_Key|str, default=str)

SIMPLE_UNITED_FORMATTER: Callable[[Column_Key|str, Unit, Value_Type], str] = lambda name, unit, _: f"{name} [{unit}]" if unit != NO_NUMBER.no_number else f"{name} [-]"

class Column_Information(Generic[CK]):

    def __init__(self, column_key: CK, si_quantity: SI_QUANTITY, value_type: Value_Type, display_unit: Unit|None = None):
        self._column_key: CK = column_key
        self._si_quantity: SI_QUANTITY = si_quantity
        self._value_type: Value_Type = value_type
        if display_unit is None:
            self._display_unit: Unit = si_quantity.si_base_unit
        else:
            self._display_unit: Unit = display_unit

    @property
    def column_key(self) -> CK:
        return self._column_key

    @property
    def si_quantity(self) -> SI_QUANTITY:
        return self._si_quantity

    @property
    def value_type(self) -> Value_Type:
        return self._value_type
    
    @property
    def display_unit(self) -> Unit:
        return self._display_unit
    
    @property
    def internal_dataframe_column_name(self, internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = SIMPLE_UNITED_FORMATTER) -> str:
        return internal_column_name_formatter(United_Dataframe[CK].column_key_to_string(self._column_key), self._si_quantity.si_base_unit, self._value_type)

class United_Dataframe_Exception(Exception):
    pass

class _InternalInitToken:
    pass
_INTERNAL_INIT_TOKEN = _InternalInitToken()

class United_Dataframe(JSONable, Generic[CK]):
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
                 internal_dataframe_column_strings: dict[CK, str]|None,
                 column_keys: list[CK],
                 si_quantities: dict[CK, SI_QUANTITY],
                 display_units: dict[CK, Unit],
                 value_types: dict[CK, Value_Type],
                 internal_column_name_formatter: Callable[[str, Unit, Value_Type], str],
                 internal_init_token: _InternalInitToken):
        """
        **This method is not meant to be called by the user.**
        **Use the class methods to create a United_Dataframe.**
        
        Initialize the United_Dataframe.
        
        **The canonical dataframe is set as a reference**
        **No deepcopy is performed on the canonical dataframe.**

        Args:
            canonical_dataframe (pd.DataFrame): The underlying pandas DataFrame storing the data
            column_keys (list[CK]): List of column keys that identify each column
            si_quantities (list[SI_QUANTITY]): List of SI quantities corresponding to each column
            display_units (list[Unit]): List of display units for each column
            value_types (list[Value_Type]): List of value types (FLOAT64, INT32, etc.) for each column
            _internal_column_name_formatter (Callable[[CK, Unit, Value_Type], str]): Function to format internal column names

        Performs validation checks:
        - Ensures internal init token is present
        - Validates that column counts match across all metadata lists
        - Validates that value types and units are compatible
        - Sets display units
        - Validates column naming consistency
        - Validates column key types
        - Initializes thread safety locks
        """

        # Step 1a: Set the fields from the constructor
        self._internal_canonical_dataframe: pd.DataFrame = internal_canonical_dataframe
        self._internal_dataframe_column_strings: dict[CK, str] = {}
        self._column_keys: list[CK] = column_keys.copy()
        self._si_quantities: dict[CK, SI_QUANTITY] = si_quantities.copy()
        self._display_units: dict[CK, Unit] = display_units.copy()
        self._value_types: dict[CK, Value_Type] = value_types.copy()
        self._internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = internal_column_name_formatter
        self._internal_init_token: _InternalInitToken = internal_init_token

        # Step 1b: Set the other fields
        self._read_only: bool = False
        self._lock: rwlock.RWLockFairD = rwlock.RWLockFairD()
        self._rlock: rwlock.RWLockFairD._aReader = self._lock.gen_rlock()
        self._wlock: rwlock.RWLockFairD._aWriter = self._lock.gen_wlock()

        # Step 2: Ensure the number of columns, units and value types match

        if self._internal_canonical_dataframe.columns.size != len(self._column_keys):
            raise United_Dataframe_Exception(f"The number of columns in the dataframe ({self._internal_canonical_dataframe.columns.size}) does not match the number of column names ({len(self._column_keys)}).")
        
        if self._internal_canonical_dataframe.columns.size != len(self._si_quantities):
            raise United_Dataframe_Exception(f"The number of columns in the dataframe ({self._internal_canonical_dataframe.columns.size}) does not match the number of SI quantities ({len(self._si_quantities)}).")
        
        if self._internal_canonical_dataframe.columns.size != len(self._value_types):
            raise United_Dataframe_Exception(f"The number of columns in the dataframe ({self._internal_canonical_dataframe.columns.size}) does not match the number of value types ({len(self._value_types)}).")
            
        # Step 3: Ensure that the value types and units match
        for column_key, si_quantity, value_type in zip(self._column_keys, self._si_quantities.values(), self._value_types.values()):
            if value_type.value.is_numeric and si_quantity == SI_QUANTITY.NO_NUMBER:
                raise United_Dataframe_Exception(f"The value type {value_type.value.name} is numeric, but the unit is NO_NUMBER.no_number.")
            if value_type.value.is_non_numeric and si_quantity != SI_QUANTITY.NO_NUMBER:
                raise United_Dataframe_Exception(f"The value type {value_type.value.name} is non-numeric, but the unit is not NO_NUMBER.no_number.")
            
        # Step 4: Set the display units
        for column_key, si_quantity in self._si_quantities.items():
            self._display_units[column_key] = si_quantity.si_base_unit

        # Step 5: If the given dataframe does not have the proper column names, rename the columns provided in the dataframe_column_names dictionary. If it is not provided, check if the column names are the same as the column keys.

        if internal_dataframe_column_strings is not None:
            renaming_dict: dict[str, str] = {}
            for column_key, si_quantity, value_type in zip(self._column_keys, self._si_quantities.values(), self._value_types.values()):
                current_dataframe_column_name: str = internal_dataframe_column_strings[column_key]
                proper_dataframe_column_name: str = self._internal_column_name_formatter(current_dataframe_column_name, si_quantity.si_base_unit, value_type)
                renaming_dict[current_dataframe_column_name] = proper_dataframe_column_name
                self._internal_dataframe_column_strings[column_key] = proper_dataframe_column_name
            self._internal_canonical_dataframe.rename(columns=renaming_dict, inplace=True)
        else:
            for column_key, si_quantity, value_type in zip(self._column_keys, self._si_quantities.values(), self._value_types.values()):
                current_dataframe_column_name: str = self._internal_canonical_dataframe.columns[self._column_keys.index(column_key)]
                proper_dataframe_column_name: str = self._internal_column_name_formatter(current_dataframe_column_name, si_quantity.si_base_unit, value_type)
                if current_dataframe_column_name != proper_dataframe_column_name:
                    raise United_Dataframe_Exception(f"The dataframe column name {current_dataframe_column_name} does not match the dataframe column name {proper_dataframe_column_name}.")
                self._internal_dataframe_column_strings[column_key] = current_dataframe_column_name

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
        if isinstance(column_key, Column_Key):
            column_key_str: str = column_key.to_string()
        else:
            column_key_str: str = column_key
        return self._internal_column_name_formatter(column_key_str, self._si_quantities[column_key].si_base_unit, self._value_types[column_key])

    @staticmethod
    def column_key_to_string(column_key: CK) -> str:
        if isinstance(column_key, Column_Key):
            return column_key.to_string()
        else:
            return column_key

    def column_information(self, column_key: CK) -> Column_Information[CK]:
        with self._rlock:
            return Column_Information(column_key, self._si_quantities[column_key], self._value_types[column_key])

    @property
    def internal_dataframe_deepcopy(self) -> pd.DataFrame:
        """
        Get a deep copy of the internal pandas DataFrame.
        
        Returns:
            pd.DataFrame: A deep copy of the underlying pandas DataFrame
        """
        with self._rlock:
            return self._internal_canonical_dataframe.copy(deep=True)

    def copy(self, deep: bool = True) -> "United_Dataframe[CK]":
        """
        Create a deep copy of the United_Dataframe.
        
        Returns:
            United_Dataframe: A new instance with copied data and metadata
        """
        with self._rlock:
            new_df: United_Dataframe[CK] = United_Dataframe(
                self._internal_canonical_dataframe.copy(deep=deep),
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)
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

    # ----------- Retrievals: Value types ------------

    def value_type(self, column_key: CK) -> Value_Type:
        with self._rlock:
            return self._value_types[column_key]

    @overload
    def value_types(self, column_keys: CK) -> Value_Type:
        ...

    @overload
    def value_types(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[Value_Type]:
        ...

    @overload
    def value_types(self, column_keys: list[CK]) -> list[Value_Type]:
        ...

    @overload
    def value_types(self, column_keys: set[CK]) -> set[Value_Type]:
        ...

    def value_types(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> Value_Type|list[Value_Type]|set[Value_Type]:
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
                case Column_Key()|str():
                    if len(more_column_keys) == 0:
                        return self._value_types[column_keys]
                    else:
                        return [self._value_types[column_keys]] + [self._value_types[more_column_key] for more_column_key in more_column_keys]
                case list():
                    value_types_as_list: list[Value_Type] = []
                    for column_key in column_keys:
                        value_types_as_list.append(self._value_types[column_key])
                    return value_types_as_list
                case set():
                    value_types_as_set: set[Value_Type] = set()
                    for column_key in column_keys:
                        value_types_as_set.add(self._value_types[column_key])
                    return value_types_as_set
                case _:
                    raise United_Dataframe_Exception(f"Invalid column keys: {column_keys}.")
    
    @property
    def value_type_dict(self) -> dict[CK, Value_Type]:
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
                case Column_Key()|str():
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
                    raise United_Dataframe_Exception(f"Invalid column keys: {column_keys}.")
    
    @property
    def display_unit_dict(self) -> dict[CK, Unit]:
        """
        Get a dictionary mapping column keys to their display units.
        
        Returns:
            dict[CK, Unit]: Dictionary mapping column keys to display units
        """
        with self._rlock:
            return self._display_units.copy()

    # ----------- Retrievals: SIQuantity ------------

    def si_quantity(self, column_key: CK) -> SI_QUANTITY:
        with self._rlock:
            return self._si_quantities[column_key]

    @overload
    def si_quantities(self, column_keys: CK) -> SI_QUANTITY:
        ...

    @overload
    def si_quantities(self, column_keys: CK|None=None, *more_column_keys: CK) -> list[SI_QUANTITY]:
        ...

    @overload
    def si_quantities(self, column_keys: list[CK]) -> list[SI_QUANTITY]:
        ...

    @overload
    def si_quantities(self, column_keys: set[CK]) -> set[SI_QUANTITY]:
        ...

    def si_quantities(self, column_keys: CK|list[CK]|set[CK]|None=None, *more_column_keys: CK) -> SI_QUANTITY|list[SI_QUANTITY]|set[SI_QUANTITY]:
        """
        Get the SI quantity(ies) by column key(s).
        
        Args:
            column_keys (CK|list[CK]|set[CK]|None): The column key(s) to get the SI quantity(ies) of. If None, all column keys are used.
            *more_column_keys (CK): Additional column keys to get the value type(s) of.
            
        Returns:
            SI_QUANTITY|list[SI_QUANTITY]|set[SI_QUANTITY]: The SI quantity(ies) of the specified column(s)
        """
        with self._rlock:
            match column_keys:
                case Column_Key()|str():
                    if len(more_column_keys) == 0:
                        return self._si_quantities[column_keys]
                    else:
                        return [self._si_quantities[column_keys]] + [self._si_quantities[more_column_key] for more_column_key in more_column_keys]
                case list():
                    si_quantities_as_list: list[SI_QUANTITY] = []
                    for column_key in column_keys:
                        si_quantities_as_list.append(self._si_quantities[column_key])
                    return si_quantities_as_list
                case set():
                    si_quantities_as_set: set[SI_QUANTITY] = set()
                    for column_key in column_keys:
                        si_quantities_as_set.add(self._si_quantities[column_key])
                    return si_quantities_as_set
                case _:
                    raise United_Dataframe_Exception(f"Invalid column keys: {column_keys}.")
    
    @property
    def si_quantity_dict(self) -> dict[CK, SI_QUANTITY]:
        """
        Get a dictionary mapping column keys to their SI quantities.
        
        Returns:
            dict[CK, SI_QUANTITY]: Dictionary mapping column keys to SI quantities
        """
        with self._rlock:
            return self._si_quantities.copy()

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
                case Column_Key()|str():
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
                    raise United_Dataframe_Exception(f"Invalid column keys: {column_keys}.")
    
    @property
    def internal_dataframe_column_strings_dict(self) -> dict[CK, str]:
        """
        Get a dictionary mapping column keys to their internal dataframe column strings.
        
        Returns:
            dict[CK, str]: Dictionary mapping column keys to internal dataframe column strings
        """
        with self._rlock:
            return self._internal_dataframe_column_strings.copy()

    # ----------- Column properties ------------

    def is_numeric(self, column_key: CK) -> bool:
        """
        Check if a column contains numeric data.
        
        Args:
            index_or_column_key (int|CK): The index or column key to check
            
        Returns:
            bool: True if the column is numeric, False otherwise
            
        Raises:
            United_Dataframe_Exception: If the column does not exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            return self._value_types[column_key].value.is_numeric
    
    def is_non_numeric(self, column_key: CK) -> bool:
        """
        Check if a column contains non-numeric data.
        
        Args:
            index_or_column_key (int|CK): The index or column key to check
            
        Returns:
            bool: True if the column is non-numeric, False otherwise
            
        Raises:
            United_Dataframe_Exception: If the column does not exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            return self._value_types[column_key].value.is_non_numeric
    
    # ----------- Column Information ------------

    def get_column_information_list(self) -> list[Column_Information[CK]]:
        """
        Get the column information list.
        """
        with self._rlock:
            return [Column_Information[CK](column_key, self._si_quantities[column_key], self._value_types[column_key], self._display_units[column_key]) for column_key in self._column_keys]

    def column_information_of_type(self, *column_key_types: type[CK_CF]) -> list[Column_Information[CK_CF]]:
        """
        Filter the dataframe by column key type.
        """
        with self._rlock:

            column_information_of_type: list[Column_Information[CK_CF]] = []
            for column_key in self._column_keys:
                column_key_filtered_type: CK = column_key
                if isinstance(column_key_filtered_type, tuple(column_key_types)):
                    column_information_of_type.append(Column_Information[CK_CF](
                        column_key_filtered_type,
                        self._si_quantities[column_key],
                        self._value_types[column_key],
                        self._display_units[column_key]))
            return column_information_of_type

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
            case Column_Key():
                return column_key.to_string()
            case str():
                return column_key
            case _:
                raise United_Dataframe_Exception(f"Invalid column key: {column_key}.")

    def _check_compatibility(self, column_key: CK, value: United_Value) -> bool:
        """
        Check if a value is compatible with a value type.
        """
        return self.value_type(column_key).value.corresponding_united_value_type == type(value.canonical_value)

    def _get_dataframe_with_new_canonical_dataframe(self, new_canonical_dataframe: pd.DataFrame) -> "United_Dataframe[CK]":
        """
        Get a new United_Dataframe with a new canonical dataframe, but using the same column information.
        """
        return United_Dataframe[CK].create_from_dataframe_and_column_information_list(
            new_canonical_dataframe,
            None,
            self.get_column_information_list(),
            self._internal_column_name_formatter,
            True)

    # ----------- Setters: Column names ------------

    def rename_column(self, current_column_key: CK, new_column_key: CK):
        """
        Set the column key for a column at the specified index.
        
        Args:
            current_column_key (CK): The current column key
            new_column_key (CK): The new column key
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, the name already exists,
                                       or the new column name conflicts with existing columns
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(current_column_key):
                raise United_Dataframe_Exception(f"Column key {current_column_key} does not exist in the dataframe.")
            if self.has_column(new_column_key):
                raise United_Dataframe_Exception(f"Column key {new_column_key} already exists in the dataframe.")
            current_internal_dataframe_column_name: str = self.internal_dataframe_column_string(current_column_key)
            new_internal_dataframe_column_name: str = self.create_internal_dataframe_column_name(new_column_key)
            if new_internal_dataframe_column_name in self._internal_canonical_dataframe.columns:
                raise United_Dataframe_Exception(f"Column name {new_internal_dataframe_column_name} already exists in the dataframe.")
            self._internal_canonical_dataframe.rename(columns={current_internal_dataframe_column_name: new_internal_dataframe_column_name}, inplace=True)
            self._column_keys[self._column_keys.index(current_column_key)] = new_column_key
            self._internal_dataframe_column_strings.pop(current_column_key)
            self._internal_dataframe_column_strings[new_column_key] = new_internal_dataframe_column_name
            self._si_quantities[new_column_key] = self._si_quantities.pop(current_column_key)
            self._value_types[new_column_key] = self._value_types.pop(current_column_key)
            self._display_units[new_column_key] = self._display_units.pop(current_column_key)

    # ----------- Column operations ------------

    def column_values_as_numpy_array(self, column_key: CK, in_units: Unit) -> np.ndarray:
        """
        Get a column as a numpy array in the specified units.
        
        Args:
            column_key (CK): The column key of the column
            in_units (Unit): The units to return the data in
            
        Returns:
            np.ndarray: The column data as a numpy array in the specified units
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist, unit compatibility issues,
                                       or type mismatches between numeric/non-numeric data
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            match self.is_numeric(column_key), in_units:
                case False, None | NO_NUMBER.no_number:
                        return self._internal_canonical_dataframe[dataframe_column_name].to_numpy()
                case False, _:
                    raise United_Dataframe_Exception(f"Column {dataframe_column_name} is not numeric, but the unit {in_units} is something else than NO_NUMBER.no_number or None.")
                case True, None | NO_NUMBER.no_number:
                    raise United_Dataframe_Exception(f"Column {dataframe_column_name} is numeric, but the unit {in_units} is something NO_NUMBER.no_number or None.")
                case True, _:
                    si_quantity: SI_QUANTITY = self._si_quantities[column_key]
                    if in_units.si_quantity != si_quantity:
                        raise United_Dataframe_Exception(f"Unit {in_units} is not compatible with the SI quantity {si_quantity} of the column {dataframe_column_name}.")
                    return in_units.from_canonical_unit_array(self._internal_canonical_dataframe[dataframe_column_name].to_numpy())
    
    def column_values_as_canonical_numpy_array(self, column_key: CK) -> np.ndarray:
        """
        Get a column as a numpy array in the specified units.
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            return self._internal_canonical_dataframe[dataframe_column_name].to_numpy()

    def column_values_as_pandas_series(self, column_key: CK, in_units: Unit) -> pd.Series:
        """
        Get a column as a pandas Series in the specified units.
        
        Args:
            column_key (CK): The column key of the column
            in_units (Unit): The units to return the data in
            
        Returns:
            pd.Series: The column data as a pandas Series in the specified units
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist, unit compatibility issues,
                                       or type mismatches between numeric/non-numeric data
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            match self.is_numeric(column_key), in_units:
                case False, None | NO_NUMBER.no_number:
                    return self._internal_canonical_dataframe[dataframe_column_name]
                case False, _:
                    raise United_Dataframe_Exception(f"Column {dataframe_column_name} is not numeric, but the unit {in_units} is something else than NO_NUMBER.no_number or None.")
                case True, None | NO_NUMBER.no_number:
                    raise United_Dataframe_Exception(f"Column {dataframe_column_name} is numeric, but the unit {in_units} is something NO_NUMBER.no_number or None.")
                case True, _:
                    si_quantity: SI_QUANTITY = self._si_quantities[column_key]
                    if in_units.si_quantity != si_quantity:
                        raise United_Dataframe_Exception(f"Unit {in_units} is not compatible with the SI quantity {si_quantity} of the column {dataframe_column_name}.")
                    return in_units.from_canonical_unit_series(self._internal_canonical_dataframe[dataframe_column_name])
    
    def column_values_as_canonical_pandas_series(self, column_key: CK) -> pd.Series:
        """
        Get a column as a pandas Series in the canonical units.
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            return self._internal_canonical_dataframe[dataframe_column_name]

    def column_values_as_united_array(self, column_key: CK, display_unit: Unit|None=None) -> UnitedArray:
        """
        Get a column as a United_Array with its display unit.
        
        Args:
            column_key (CK): The column key of the column
            display_unit (Unit|None): The display unit to use. If None, the column's display unit is used.
            
        Returns:
            United_Array: The column data as a United_Array with the column's display unit
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if display_unit is None:
                display_unit = self._display_units[column_key]
            return UnitedArray(self._internal_canonical_dataframe[dataframe_column_name].to_numpy(), self._value_types[column_key].value.corresponding_united_array_value_type, self._si_quantities[column_key], display_unit)
    
    def set_column_values_from_numpy_array(self, column_key: CK, values: np.ndarray, unit: Unit) -> None:
        """
        Set column values from a numpy array with the specified unit.
        
        Args:
            column_key (CK): The column key of the column
            values (np.ndarray): The values to set
            unit (Unit): The unit of the provided values
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if unit.si_quantity != self._si_quantities[column_key]:
                raise United_Dataframe_Exception(f"Unit {unit} is not compatible with the SI quantity {self._si_quantities[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = unit.to_canonical_unit_array(values)
            array = array.astype(self._value_types[column_key].value.corresponding_numpy_dtype)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def set_column_values_from_list(self, column_key: CK, values: list, unit: Unit) -> None:
        """
        Set column values from a list with the specified unit.
        
        Args:
            column_key (CK): The column key of the column
            values (list): The values to set
            unit (Unit): The unit of the provided values
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if unit.si_quantity != self._si_quantities[column_key]:
                raise United_Dataframe_Exception(f"Unit {unit} is not compatible with the SI quantity {self._si_quantities[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = np.array(values, dtype=self._value_types[column_key].value.corresponding_numpy_dtype)
            array = unit.to_canonical_unit_array(array)
            array = array.astype(self._value_types[column_key].value.corresponding_numpy_dtype)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def set_column_values_from_pandas_series(self, column_key: CK, values: pd.Series, unit: Unit) -> None:
        """
        Set column values from a pandas Series with the specified unit.
        
        Args:
            column_key (CK): The column key of the column
            values (pd.Series): The values to set
            unit (Unit): The unit of the provided values
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if unit.si_quantity != self._si_quantities[column_key]:
                raise United_Dataframe_Exception(f"Unit {unit} is not compatible with the SI quantity {self._si_quantities[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = unit.to_canonical_unit_array(values.to_numpy())
            array = array.astype(self._value_types[column_key].value.corresponding_numpy_dtype)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def set_column_values_from_united_array(self, column_key: CK, values: UnitedArray) -> None:
        """
        Set column values from a United_Array.
        
        Args:
            column_key (CK): The column key of the column
            values (United_Array): The values to set
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       unit incompatibility, or column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if values.display_unit != self._display_units[column_key]:
                raise United_Dataframe_Exception(f"Unit {values.display_unit} is not compatible with the display unit {self._display_units[column_key]} of the column {dataframe_column_name}.")
            array: np.ndarray = values._canonical_np_array
            array = array.astype(self._value_types[column_key].value.corresponding_numpy_dtype)
            self._internal_canonical_dataframe[dataframe_column_name] = array

    def add_empty_column(self, column_key: CK, si_quantity_or_display_unit: SI_QUANTITY|Unit, value_type: Value_Type) -> None:
        """
        Add an empty column to the dataframe.
        
        Args:
            column_key (CK): The key for the new column
            si_quantity_or_display_unit (SI_QUANTITY|Unit): The SI quantity or display unit for the column
            value_type (Value_Type): The value type for the column
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, the column key already exists,
                                       or the SI quantity and value type are incompatible
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} already exists in the dataframe.")
        
            if isinstance(si_quantity_or_display_unit, SI_QUANTITY):
                si_quantity: SI_QUANTITY = si_quantity_or_display_unit
                display_unit: Unit = si_quantity.si_base_unit
            else:
                display_unit: Unit = si_quantity_or_display_unit
                si_quantity: SI_QUANTITY = display_unit.si_quantity

            if si_quantity.is_numeric != value_type.value.is_numeric:
                raise United_Dataframe_Exception(f"The SI quantity {si_quantity} and the value type {value_type} are not compatible.")

            dataframe_column_name: str = self.create_internal_dataframe_column_name(column_key)
            if dataframe_column_name in self._internal_canonical_dataframe.columns:
                raise United_Dataframe_Exception(f"Column key {column_key} already exists in the dataframe.")
            self._internal_canonical_dataframe[dataframe_column_name] = pd.Series([pd.NA] * len(self._internal_canonical_dataframe), dtype=value_type.value.corresponding_pandas_type)
            self._column_keys.append(column_key)
            self._internal_dataframe_column_strings[column_key] = dataframe_column_name
            self._si_quantities[column_key] = si_quantity
            self._value_types[column_key] = value_type
            self._display_units[column_key] = display_unit

    def remove_column(self, column_key: CK) -> None:
        """
        Remove a column from the dataframe.
        
        Args:
            column_key (CK): The column key of the column to remove
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only or the column doesn't exist
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            self._internal_canonical_dataframe.drop(columns=[dataframe_column_name], inplace=True)
            self._internal_dataframe_column_strings.pop(column_key)
            self._column_keys.remove(column_key)
            self._si_quantities.pop(column_key)
            self._value_types.pop(column_key)
            self._display_units.pop(column_key)

    def remove_columns(self, column_keys: list[CK]|set[CK]) -> None:
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            for column_key in column_keys:
                self.remove_column(column_key)

    def add_column_from_list(self, column_key: CK, values: list, unit: Unit, value_type: Value_Type) -> None:
        """
        Add a new column with data from a list.
        
        Args:
            column_key (CK): The key for the new column
            values (list): The values for the column
            unit (Unit): The unit of the values
            value_type (Value_Type): The value type for the column
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            self.add_empty_column(column_key, unit, value_type)
            self.set_column_values_from_list(column_key, values, unit)
    
    def add_column_from_numpy_array(self, column_key: CK, values: np.ndarray, unit: Unit, value_type: Value_Type) -> None:
        """
        Add a new column with data from a numpy array.
        
        Args:
            column_key (CK): The key for the new column
            values (np.ndarray): The values for the column
            unit (Unit): The unit of the values
            value_type (Value_Type): The value type for the column
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            self.add_empty_column(column_key, unit, value_type)
            self.set_column_values_from_numpy_array(column_key, values, unit)
    
    def add_column_from_pandas_series(self, column_key: CK, values: pd.Series, unit: Unit, value_type: Value_Type) -> None:
        """
        Add a new column with data from a pandas Series.
        
        Args:
            column_key (CK): The key for the new column
            values (pd.Series): The values for the column
            unit (Unit): The unit of the values
            value_type (Value_Type): The value type for the column
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            self.add_empty_column(column_key, unit, value_type)
            self.set_column_values_from_pandas_series(column_key, values, unit)
    
    def add_column_from_united_array(self, column_key: CK, values: UnitedArray) -> None:
        """
        Add a new column with data from a United_Array.
        
        Args:
            column_key (CK): The key for the new column
            values (United_Array): The values for the column
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, length mismatch,
                                       or the column key already exists
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if len(values) != len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of rows ({len(self._internal_canonical_dataframe)}).")
            value_type: Value_Type = Value_Type.find_value_type_by_united_value_type(values.value_type)
            self.add_empty_column(column_key, values.display_unit, value_type)
            self.set_column_values_from_united_array(column_key, values)
            
    def get_iterator_for_column(self, column_key: CK) -> Iterator[United_Value]:
        """
        Get an iterator over the values of a column.

        Args:
            column_key (CK): The column key of the column to get the iterator for

        Returns:
            Iterator[United_Value]: An iterator over the values of the column
        """
        with self._rlock:
            return (self.get_cell_value(row_index, column_key) for row_index in range(len(self._internal_canonical_dataframe)))

    # ----------- Row operations ------------

    def row(self, row_index: int) -> dict[CK, United_Value]:
        """
        Get a row from the dataframe.

        Args:
            index (int): The index of the row to get

        Returns:
            dict[CK, United_Value]: A dictionary of column keys and values for the row
        """
        with self._rlock:
            return {column_key: self.get_cell_value(row_index, column_key) for column_key in self._column_keys}
    
    def iterrows(self) -> Iterator[tuple[int, dict[CK, United_Value]]]:
        """
        Iterate over dataframe rows as (row_index, row_dict) pairs.
        """
        with self._rlock:
            for row_index in range(len(self._internal_canonical_dataframe)):
                yield (row_index, self.row(row_index))

    def get_iterator_for_row(self, row_index: int) -> Iterator[United_Value]:
        """
        Get an iterator over the values of a row.

        Args:
            row_index (int): The index of the row to get the iterator to iterate over the values of that row

        Returns:
            Iterator[United_Value]: An iterator over the values of the row
        """
        with self._rlock:
            return (self.get_cell_value(row_index, column_key) for column_key in self._column_keys)

    def remove_row(self, row: int) -> None:
        """
        Remove a row from the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if row < 0 or row >= len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"Row index {row} is out of bounds.")
            self._internal_canonical_dataframe.drop(index=row, inplace=True)

    def remove_all_rows(self) -> None:
        """
        Remove all rows from the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            self._internal_canonical_dataframe = self._internal_canonical_dataframe.iloc[0:0]

    def remove_rows(self, row_indices: list[int]|set[int]|slice) -> None:
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if isinstance(row_indices, slice):
                self._internal_canonical_dataframe = self._internal_canonical_dataframe.iloc[row_indices]
            else:
                for row_index in sorted(row_indices, reverse=True):
                    self.remove_row(row_index)

    def insert_row(self, row: int, values: list[United_Value]|dict[CK, United_Value]) -> None:
        """
        Insert a row at the specified index.
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if row < 0 or row > len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"Row index {row} is out of bounds.")
            if len(values) != len(self._column_keys):
                raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of columns ({len(self._column_keys)}).")
            
            if isinstance(values, list):
                if len(values) != len(self._column_keys):
                    raise United_Dataframe_Exception(f"The number of values ({len(values)}) does not match the number of columns ({len(self._column_keys)}).")
                for index, value in enumerate(values):
                    dataframe_column_name: str = self.internal_dataframe_column_string(self._column_keys[index])
                    if self._check_compatibility(self._column_keys[index], value):
                        self._internal_canonical_dataframe.at[row, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                    else:
                        raise United_Dataframe_Exception(f"Value {value} is not compatible with the value type {self.value_type(self._column_keys[index])}.")
            if isinstance(values, dict):
                if any(not self.has_column(key) for key in values.keys()):
                    for key in values.keys():
                        if not self.has_column(key):
                            raise United_Dataframe_Exception(f"Column key {key} does not exist in the dataframe.")
                values_to_insert = [] * len(self._column_keys)
                for column_key in self._column_keys:
                    dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
                    if column_key in values:
                        value: United_Value = values[column_key]
                        if self._check_compatibility(column_key, value):
                            self._internal_canonical_dataframe.at[row, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                        else:
                            raise United_Dataframe_Exception(f"Value {value} is not compatible with the value type {self.value_type(column_key)}.")
                    else:
                        self._internal_canonical_dataframe.at[row, dataframe_column_name] = pd.NA

    def add_row(self, values: list[United_Value]|dict[CK, United_Value]) -> None:
        """
        Add a row to the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            self.insert_row(len(self._internal_canonical_dataframe), values)

    def add_empty_rows(self, number_of_rows: int=1) -> None:
        """
        Add empty rows to the dataframe.
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            self._internal_canonical_dataframe = pd.concat([self._internal_canonical_dataframe, pd.DataFrame(index=range(len(self._internal_canonical_dataframe), len(self._internal_canonical_dataframe) + number_of_rows))], ignore_index=True)

    # ----------- Cell operations ------------

    def get_cell_value(self, row_index: int, column_key: CK) -> United_Value:
        """
        Get the value of a specific cell as a United_Value.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            
        Returns:
            United_Value: The cell value with appropriate unit information
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            # Check if the cell is empty
            if pd.isna(self._internal_canonical_dataframe.at[row_index, dataframe_column_name]):
                if self.value_type(column_key).has_python_na_value():
                    return create_united_value_unsafely(self._internal_canonical_dataframe.at[row_index, dataframe_column_name], self._display_units[column_key])
                else:
                    raise United_Dataframe_Exception(f"The requested cell is empty, but no NA value is defined for the value type {self.value_type(column_key).__name__} of the column {column_key}.")
            else:
                match self.value_type(column_key):
                    case Value_Type.BOOLEAN:
                        return create_united_value_unsafely(self._internal_canonical_dataframe.at[row_index, dataframe_column_name], NO_NUMBER.no_number)
                    case Value_Type.FLOAT64 | Value_Type.FLOAT32 | Value_Type.INT64 | Value_Type.INT32 | Value_Type.INT16 | Value_Type.INT8:
                        return create_united_value_unsafely(self._internal_canonical_dataframe.at[row_index, dataframe_column_name], self._display_units[column_key])
                    case Value_Type.STRING:
                        return create_united_value_unsafely(self._internal_canonical_dataframe.at[row_index, dataframe_column_name], NO_NUMBER.no_number)
                    case Value_Type.DATETIME64:
                        return create_united_value_unsafely(self._internal_canonical_dataframe.at[row_index, dataframe_column_name], NO_NUMBER.no_number)
                    case _:
                        raise United_Dataframe_Exception(f"Invalid column value type: {self.value_type(column_key).__name__}")
                    
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
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[CK]|set[CK]) -> "United_Dataframe[CK]":
        """
        Get a new dataframe with the selected columns.
        
        Args:
            index_or_column_key_or_list_of_keys (list[int|CK]): The column indices or column keys
            
        Returns:
            United_Dataframe[CK]: A new dataframe with the selected columns as a shallow copy
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: list[int]|set[int]|slice) -> "United_Dataframe[CK]":
        """
        Get a new dataframe with the selected rows.
        """
        ...

    @overload
    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: tuple[int, CK]|tuple[CK, int]) -> United_Value:
        """
        Get a cell value for pandas-like cell access.
        """
        ...

    def __getitem__(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position: int|CK|list[int]|set[int]|slice|list[CK]|set[CK]|tuple[int, CK]|tuple[CK, int]) -> "_ColumnAccessor[CK] | _RowAccessor[CK] | United_Dataframe[CK] | United_Value":
        """
        Get a column accessor for pandas-like column access.
        
        Args:
            index_or_column_key_or_list_of_keys (int|CK|list[int|CK]): The column index or column key or a list of column keys
            
        Returns:
            ColumnAccessor[CK] | United_Dataframe[CK]: An accessor object for the specified column or a new dataframe with the selected columns as a shallow copy
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist
        """
        with self._rlock:

            match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position:
                case int():
                    return _RowAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case Column_Key()|str():
                    return _ColumnAccessor(self, column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                case slice():
                    new_united_dataframe = self.copy(deep=True)
                    new_united_dataframe.remove_rows(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)
                    return new_united_dataframe
                case list() | set():
                    if len(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) == 0:
                        return United_Dataframe[CK].create_empty_dataframe([], [], [], 0, self._internal_column_name_formatter)
                    if isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), int):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.remove_rows(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) # type: ignore[arg-type]
                        return new_united_dataframe
                    elif isinstance(next(iter(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position)), Column_Key|str):
                        new_united_dataframe = self.copy(deep=True)
                        new_united_dataframe.remove_columns(column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position) # type: ignore[arg-type]
                        return new_united_dataframe
                    else:
                        raise United_Dataframe_Exception(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                case tuple():
                    match column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0], column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1]:
                        case int(), Column_Key()|str():
                            row_index: int = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0] # type: ignore[assignment]
                            column_key: CK = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1] # type: ignore[assignment]
                            return self.get_cell_value(row_index, column_key)
                        case Column_Key()|str(), int():
                            column_key: CK = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[0] # type: ignore[assignment]
                            row_index: int = column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position[1] # type: ignore[assignment]
                            return self.get_cell_value(row_index, column_key)
                        case _:
                            raise United_Dataframe_Exception(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                case _:
                    raise United_Dataframe_Exception(f"Invalid key: {column_key_or_row_key_or_list_of_column_keys_or_list_of_row_indices_or_cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")

    def set_cell_value(self, row_index: int, column_key: CK, value: United_Value) -> None:
        """
        Set the value of a specific cell.
        
        Args:
            row_index (int): The row index
            column_key (CK): The column key
            value (United_Value): The value to set
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, the column doesn't exist,
                                       the row is out of bounds, or the value type is incompatible
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if not 0 <= row_index < len(self._internal_canonical_dataframe):
                raise United_Dataframe_Exception(f"Row {row_index} does not exist in the dataframe. The dataframe has {len(self._internal_canonical_dataframe)} rows.")

            match value.canonical_value, self.value_type(column_key):
                case bool() | int(), Value_Type.BOOLEAN:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                case float() | int(), Value_Type.FLOAT64 | Value_Type.FLOAT32 | Value_Type.INT64 | Value_Type.INT32 | Value_Type.INT16 | Value_Type.INT8:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                case str(), Value_Type.STRING:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                case datetime(), Value_Type.DATETIME64:
                    self._internal_canonical_dataframe.at[row_index, dataframe_column_name] = self._internal_canonical_dataframe[dataframe_column_name].dtype.type(value.canonical_value)
                case _:
                    raise United_Dataframe_Exception(f"Invalid united value type: {type(value.canonical_value).__name__} and column value type: {self.value_type(column_key).__name__}")
    
    def __setitem__(self, cell_position: tuple[int, CK]|tuple[CK, int], value: United_Value):
        """
        Set a cell value using pandas-like syntax.
        
        Args:
            cell_position (tuple[int, CK]|tuple[CK, int]): The cell position
            value (United_Value): The value to set
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, the column doesn't exist, or the row is out of bounds
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            match cell_position:
                case int(), Column_Key()|str():
                    row_index: int = cell_position[0]
                    column_key: CK = cell_position[1]
                case Column_Key()|str(), int():
                    column_key: CK = cell_position[0]
                    row_index: int = cell_position[1]
                case _:
                    raise United_Dataframe_Exception(f"Invalid key: {cell_position}. Use df[key][row] for column access or df.loc[row, key] for row/column access.")
                
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            if not (0 <= row_index < len(self._internal_canonical_dataframe)):
                raise United_Dataframe_Exception(f"The row index {row_index} does not exist. The dataframe has {len(self)} rows.")
            self.set_cell_value(row_index, column_key, value)

    # ----------- Column functions operations ------------

    def colfun_min(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> United_Value:
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
            United_Value: The minimum value with appropriate unit information
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if len(self._internal_canonical_dataframe) == 0:
                return United_Value.united_number(np.nan, self._display_units[column_key])

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
                    raise United_Dataframe_Exception(f"Invalid case: {case}")
            
            if len(values) == 0:
                display_unit: Unit = self._display_units[column_key]
                return United_Value.united_number(display_unit.from_canonical_unit(np.nan), display_unit)
            
            display_unit: Unit = self._display_units[column_key]
            return United_Value.united_number(display_unit.from_canonical_unit(np.min(values)), display_unit)

    def colfun_max(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> United_Value:
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
            United_Value: The maximum value with appropriate unit information
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if len(self._internal_canonical_dataframe) == 0:
                return United_Value.united_number(np.nan, self._display_units[column_key])
            
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
                    raise United_Dataframe_Exception(f"Invalid case: {case}")
        
            if len(values) == 0:
                display_unit: Unit = self._display_units[column_key]
                return United_Value.united_number(display_unit.from_canonical_unit(np.nan), display_unit)
            
            display_unit: Unit = self._display_units[column_key]
            return United_Value.united_number(display_unit.from_canonical_unit(np.max(values)), display_unit)

    def colfun_sum(self, column_key: CK) -> United_Value:
        """
        Calculate the sum of a numeric column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            United_Value: The sum with appropriate unit information
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                display_unit: Unit = self._display_units[column_key]
                return United_Value.united_number(display_unit.from_canonical_unit(np.sum(values)), display_unit)
            else:
                raise United_Dataframe_Exception(f"Column '{column_key}' is not numeric.")

    def colfun_mean(self, column_key: CK) -> United_Value:
        """
        Calculate the mean of a numeric column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            United_Value: The mean with appropriate unit information
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                display_unit: Unit = self._display_units[column_key]
                return United_Value.united_number(display_unit.from_canonical_unit(float(np.mean(values))), display_unit)
            else:
                raise United_Dataframe_Exception(f"Column '{column_key}' is not numeric.")

    def colfun_std(self, column_key: CK) -> United_Value:
        """
        Calculate the standard deviation of a numeric column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            United_Value: The standard deviation with appropriate unit information
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                display_unit: Unit = self._display_units[column_key]
                return United_Value.united_number(display_unit.from_canonical_unit(float(np.std(values))), display_unit)
            else:
                raise United_Dataframe_Exception(f"Column '{column_key}' is not numeric.")
            
    def colfun_unique(self, column_key: CK) -> list[United_Value]:
        """
        Get the unique values of a column.

        Args:
            index_or_column_key (int|CK): The index or column key of the column

        Returns:
            list[United_Value]: The unique values with appropriate unit information
        """
        with self._rlock:
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            unique_values: list[United_Value] = []
            
            match self.value_type(column_key):
                case Value_Type.FLOAT64 | Value_Type.FLOAT32 | Value_Type.INT64 | Value_Type.INT32 | Value_Type.INT16 | Value_Type.INT8:
                    display_unit: Unit = self._display_units[column_key]
                    for value in values.unique():
                        unique_values.append(United_Value.united_number(display_unit.from_canonical_unit(value), display_unit))
                case Value_Type.STRING:
                    for value in values.unique():
                        unique_values.append(United_Value.united_text(value))
                case Value_Type.BOOLEAN:
                    for value in values.unique():
                        unique_values.append(United_Value.united_boolean(value))
                case Value_Type.DATETIME64:
                    for value in values.unique():
                        unique_values.append(United_Value.united_datetime(value))
                case _:
                    raise United_Dataframe_Exception(f"Invalid value type: {self.value_type(column_key)}")
            
            return unique_values
        
    def colfun_smallest_positive_nonzero_value(self, column_key: CK) -> United_Value:
        """
        Get the smallest positive non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            positive_nonzero_values: pd.Series = values[(values > 0) & values.notna()]
            if len(positive_nonzero_values) == 0:
                return United_Value.united_number(np.nan, self._display_units[column_key])
            return United_Value.united_number(np.min(positive_nonzero_values), self._display_units[column_key])
        
    def colfun_largest_positive_nonzero_value(self, column_key: CK) -> United_Value:
        """
        Get the largest positive non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            positive_nonzero_values: pd.Series = values[(values > 0) & values.notna()]
            if len(positive_nonzero_values) == 0:
                return United_Value.united_number(np.nan, self._display_units[column_key])
            return United_Value.united_number(np.max(positive_nonzero_values), self._display_units[column_key])

    def colfun_largest_negative_nonzero_value(self, column_key: CK) -> United_Value:
        """
        Get the largest negative non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            negative_nonzero_values: pd.Series = values[(values < 0) & values.notna()]
            if len(negative_nonzero_values) == 0:
                return United_Value.united_number(np.nan, self._display_units[column_key])
            return United_Value.united_number(np.max(negative_nonzero_values), self._display_units[column_key])
        
    def colfun_smallest_negative_nonzero_value(self, column_key: CK) -> United_Value:
        """
        Get the smallest negative non-zero value of a column.
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
            negative_nonzero_values: pd.Series = values[(values < 0) & values.notna()]
            if len(negative_nonzero_values) == 0:
                return United_Value.united_number(np.nan, self._display_units[column_key])
            return United_Value.united_number(np.min(negative_nonzero_values), self._display_units[column_key])
    
    @overload
    def colfun_count_value_occurances(self, column_key: CK) -> dict[United_Value, int]:
        """
        Count the number of occurrences of each unique value in the column.
        """
        ...
    
    @overload
    def colfun_count_value_occurances(self, column_key: CK, value_to_count: United_Value) -> int:
        """
        Count the number of occurrences of the specified value in the column.
        """
        ...

    def colfun_count_value_occurances(self, column_key: CK, value_to_count: United_Value|None = None) -> dict[United_Value, int]|int:
        with self._rlock:

            if value_to_count is None:
                unique_values: np.ndarray = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)].unique()
                column_as_pd_series: pd.Series = self.column_values_as_canonical_pandas_series(column_key)
                occurance_counts_dict: dict[United_Value, int] = {}
                unique_values_type: type = self.value_type(column_key).value.corresponding_united_value_type
                match unique_values_type:
                    case bool():
                        for value in unique_values:
                            occurance_counts_dict[United_Value.united_boolean(value)] = column_as_pd_series.eq(value).sum()
                    case float():
                        for value in unique_values:
                            occurance_counts_dict[United_Value.united_number(value, self._display_units[column_key])] = column_as_pd_series.eq(value).sum()
                    case str():
                        for value in unique_values:
                            occurance_counts_dict[United_Value.united_text(value)] = column_as_pd_series.eq(value).sum()
                    case datetime():
                        for value in unique_values:
                            occurance_counts_dict[United_Value.united_datetime(value)] = column_as_pd_series.eq(value).sum()
                    case _:
                        raise United_Dataframe_Exception(f"Invalid value type: {unique_values_type}")
                return occurance_counts_dict
            else:
                canonical_value: UnitedValueValueType = value_to_count.canonical_value
                expected_type: type = self.value_type(column_key).value.corresponding_united_value_type
                if not isinstance(canonical_value, expected_type):
                    raise United_Dataframe_Exception(f"Expected value of type {expected_type}, got {type(canonical_value)}")
                column_values: pd.Series = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)]                
                occurance_count: int = column_values.eq(canonical_value).sum()

                return occurance_count
            
    def colfun_row_index(self, column_key: CK, value: United_Value, case: Literal["first", "last"] = "first") -> int:
        """
        Get the row index of the first occurrence of a value in a column.

        Args:
            column_key (CK): The column key of the column
            value (United_Value): The value to get the row index of
            case (Literal["first", "last"]): The case to get the row index of

        Returns:
            int: The row index of the first occurrence of the value in the column. Returns -1 if the value is not found.
        """
        with self._rlock:

            canonical_value: UnitedValueValueType = value.canonical_value
            expected_type: type = self.value_type(column_key).value.corresponding_united_value_type
            if not isinstance(canonical_value, expected_type):
                raise United_Dataframe_Exception(f"Expected value of type {expected_type}, got {type(canonical_value)}")
            column_values: pd.Series = self._internal_canonical_dataframe[self.internal_dataframe_column_string(column_key)]
            row_index: int|str
            try:
                match case:
                    case "first":
                        row_index = column_values.eq(canonical_value).idxmax()
                    case "last":
                        row_index = column_values.eq(canonical_value).idxmin()
                    case _:
                        raise United_Dataframe_Exception(f"Invalid case: {case}")
            except ValueError:
                return -1
            if isinstance(row_index, str):
                raise United_Dataframe_Exception(f"The row index of the value {value} in the column {column_key} is not a number. It appears the index of the internal dataframe has been set.")
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
                        raise United_Dataframe_Exception(f"Column key {col} does not exist in the dataframe.")
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
    
    class _United_Value_Proxy:
        def __init__(self, symbol: str = "x"):
            self.symbol = symbol

        def _wrap(self, op, other):
            if isinstance(other, United_Value):
                val = other.canonical_float
                # Return a numpy-compatible ufunc style lambda
                return lambda x: op(x, val)
            raise TypeError("Only comparisons against United_Value are supported.")

        def __gt__(self, other): return self._wrap(operator.gt, other)
        def __ge__(self, other): return self._wrap(operator.ge, other)
        def __lt__(self, other): return self._wrap(operator.lt, other)
        def __le__(self, other): return self._wrap(operator.le, other)
        def __eq__(self, other): return self._wrap(operator.eq, other)
        def __ne__(self, other): return self._wrap(operator.ne, other)

    def _convert_filter(self, filter_function: Callable[[United_Value], bool]) -> Callable[[np.ndarray], np.ndarray]:
        proxy: United_Dataframe._United_Value_Proxy = United_Dataframe._United_Value_Proxy()
        scalar_filter: bool = filter_function(proxy) # type: ignore[arg-type]

        if not callable(scalar_filter):
            raise TypeError("The filter_function must return a callable when evaluated with proxy input.")

        def ufunc_filter(arr: np.ndarray) -> np.ndarray:
            return scalar_filter(arr)

        return ufunc_filter

    def maskfun_get_from_filter(
        self,
        column_key_and_callable: dict[CK, Callable[[United_Value], bool] | Callable[[str], bool]]
    ) -> np.ndarray:
        """
        Return a boolean mask of rows that satisfy all filter functions on selected columns.

        Args:
            column_key_and_callable (dict[CK, Callable[[United_Value], bool] | Callable[[str], bool]]): A dictionary of column keys and filter functions.

        Returns:
            np.ndarray: A boolean mask of rows that satisfy all filter functions on selected columns.
        """

        with self._rlock:
            mask_of_dataframe = np.ones(self.rows, dtype=bool)

            for column_key, filter_function in column_key_and_callable.items():
                value_type = self.value_type(column_key)

                if value_type.value.is_numeric:
                    # Numeric column
                    column_array = self.column_values_as_numpy_array(
                        column_key,
                        self.si_quantity(column_key).si_base_unit,
                    )
                    float_mask_func = self._convert_filter(filter_function)  # type: ignore[arg-type]
                    mask_of_column = float_mask_func(column_array)

                elif value_type == Value_Type.STRING:
                    # String column
                    string_array = self._internal_canonical_dataframe[
                        self.internal_dataframe_column_string(column_key)
                    ].to_numpy().ravel()

                    vectorized = np.frompyfunc(filter_function, 1, 1)
                    mask_of_column = vectorized(string_array).astype(bool)  # type: ignore[assignment]

                else:
                    raise United_Dataframe_Exception(f"Unsupported value type for column {column_key}.")

                mask_of_dataframe &= mask_of_column

            return mask_of_dataframe
        
    def maskfun_apply_mask(self, mask: np.ndarray) -> "United_Dataframe[CK]":
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
                raise United_Dataframe_Exception(f"The mask must have the same number of columns as the dataframe. The mask has {mask.ndim} columns, but the dataframe has {self.cols} columns.")

            # Check the length of the mask
            if len(mask) != self.rows:
                raise United_Dataframe_Exception(f"The mask must have the same number of rows as the dataframe. The mask has {len(mask)} rows, but the dataframe has {self.rows} rows.")

            # Check the type of the mask
            if mask.dtype != bool:
                raise United_Dataframe_Exception(f"The mask must be a boolean array. The mask has dtype {mask.dtype}.")

            # Create a new dataframe with the rows that satisfy the mask
            return United_Dataframe[CK].create_from_dataframe_and_column_information_list(
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

    def rowfun_head(self, n: int = 1) -> "United_Dataframe[CK]":
        """
        Get the first n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return (default: 1)
            
        Returns:
            United_Dataframe: A new dataframe containing the first n rows
            
        Raises:
            United_Dataframe_Exception: If n is negative
            
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
                raise United_Dataframe_Exception(f"n must be non-negative, got {n}")
            
            # If n is larger than the dataframe size, return all rows
            actual_n = min(n, len(self._internal_canonical_dataframe))
            head_df = self._internal_canonical_dataframe.head(actual_n)
            
            return United_Dataframe[CK](
                head_df,
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)

    def rowfun_first(self) -> "United_Dataframe[CK]":
        """
        Get the first row of the dataframe.
        
        Returns:
            United_Dataframe: A new dataframe containing only the first row
            
        Raises:
            United_Dataframe_Exception: If the dataframe is empty
            
        Examples:
            # Get the first row
            first_row = df.first()
            
            # Access the first row's values
            first_row.loc[0, 'column_name']
        """
        with self._rlock:
            if self.empty:
                raise United_Dataframe_Exception("Cannot get first row from an empty dataframe")
            
            first_df = self._internal_canonical_dataframe.head(1)
            
            return United_Dataframe(
                first_df,
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)

    def rowfun_tail(self, n: int = 1) -> "United_Dataframe[CK]":
        """
        Get the last n rows of the dataframe.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe: A new dataframe containing the last n rows
            
        Raises:
            United_Dataframe_Exception: If n is negative
            
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
                raise United_Dataframe_Exception(f"n must be non-negative, got {n}")
            
            # If n is larger than the dataframe size, return all rows
            actual_n = min(n, len(self._internal_canonical_dataframe))
            tail_df = self._internal_canonical_dataframe.tail(actual_n)
            
            return United_Dataframe(
                tail_df,
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)

    def rowfun_last(self) -> "United_Dataframe[CK]":
        """
        Get the last row of the dataframe.
        
        Returns:
            United_Dataframe: A new dataframe containing only the last row
            
        Raises:
            United_Dataframe_Exception: If the dataframe is empty
            
        Examples:
            # Get the last row
            last_row = df.last()
            
            # Access the last row's values
            last_row.loc[0, 'column_name']
        """
        with self._rlock:
            if self.empty:
                raise United_Dataframe_Exception("Cannot get last row from an empty dataframe")
            
            last_df = self._internal_canonical_dataframe.tail(1)
            
            return United_Dataframe(
                last_df,
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)

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
                unit_info[column_key] = f"{column_key} [{unit}]" if unit != NO_NUMBER.no_number else f"{column_key} [-]"
            
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
                    unit_str = f" [{self._display_units[column_key]}]" if self._display_units[column_key] != NO_NUMBER.no_number else " [-]"
                    print(f" {i}  {self.column_key_as_str(column_key)}{unit_str}  {dtype}  {non_null_count} non-null")
            else:
                print(f" {len(self._column_keys)} columns")
            
            if memory_usage:
                memory_usage_bytes = self._internal_canonical_dataframe.memory_usage(deep=True).sum()
                print(f"memory usage: {memory_usage_bytes} bytes")

    def sample(self, n: int | None = None, frac: float | None = None, random_state: int | None = None) -> "United_Dataframe[CK]":
        """
        Get a random sample of rows from the dataframe.
        
        Args:
            n (int | None): Number of rows to sample
            frac (float | None): Fraction of rows to sample (0.0 to 1.0)
            random_state (int | None): Random seed for reproducibility
            
        Returns:
            United_Dataframe: A new dataframe containing the sampled rows
            
        Raises:
            United_Dataframe_Exception: If both n and frac are specified
        """
        with self._rlock:
            if n is None and frac is None:
                n = 1
            elif n is not None and frac is not None:
                raise United_Dataframe_Exception("Cannot specify both n and frac")
            
            sampled_df = self._internal_canonical_dataframe.sample(n=n, frac=frac, random_state=random_state)
            return United_Dataframe(
                sampled_df,
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)

    def dropna(self, subset: list[CK] | None = None) -> "United_Dataframe[CK]":
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
            
            return United_Dataframe(
                cleaned_df,
                None,
                self._column_keys,
                self._si_quantities,
                self._display_units,
                self._value_types,
                self._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN)

    def fillna(self, value: United_Value, subset: list[CK] | None = None) -> "United_Dataframe[CK]":
        raise NotImplementedError("Not implemented")

    @classmethod
    def dataframes_can_concatenate(cls, *dataframes: "United_Dataframe[CK]") -> bool:
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
    def concatenate_dataframes(cls, dataframe: "United_Dataframe[CK]", *dataframes: "United_Dataframe[CK]") -> "United_Dataframe[CK]":
        """
        Concatenate multiple dataframes vertically.
        
        Args:
            dataframe: The first dataframe
            *dataframes: Additional dataframes to concatenate
            
        Returns:
            United_Dataframe: A new dataframe containing all rows from the input dataframes
            
        Raises:
            United_Dataframe_Exception: If the dataframes have incompatible column structures
        """
        sorted_dataframes = sorted(dataframes, key=lambda df: id(df._rlock))
        with ExitStack() as stack:
            for df in sorted_dataframes:
                stack.enter_context(df._lock._rlock) # type: ignore
            first_keys = dataframes[0].column_keys
            for dataframe in dataframes:
                if dataframe.column_keys != first_keys:
                    raise United_Dataframe_Exception(f"The dataframes cannot be concatenated. The column names or the units are not compatible.")
        
        concatenated_dataframe: pd.DataFrame = dataframe.internal_dataframe_deepcopy
        for df in dataframes:
            concatenated_dataframe = pd.concat([concatenated_dataframe, df.internal_dataframe_deepcopy], ignore_index=True)
        return cls(
            concatenated_dataframe,
            None,
            dataframe._column_keys,
            dataframe._si_quantities,
            dataframe._display_units,
            dataframe._value_types,
            dataframe._internal_column_name_formatter,
            _INTERNAL_INIT_TOKEN)

    def mask_in_range(self, column_key: CK, min_val: United_Value, max_val: United_Value) -> np.ndarray:
        """
        Check which rows have values in the specified range.
        
        Args:
            column_key (CK): The column key to check
            min_val (United_Value): The minimum value (inclusive)
            max_val (United_Value): The maximum value (inclusive)
            
        Returns:
            np.ndarray: Boolean array where True indicates values in range [min_val, max_val]
            
        Raises:
            United_Dataframe_Exception: If the column doesn't exist or is not numeric
        """
        with self._rlock:
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                return values.between(min_val.canonical_value, max_val.canonical_value).to_numpy()
            else:
                raise United_Dataframe_Exception(f"Column '{column_key}' is not numeric.")

    def colfun_clamp_column(self, column_key: CK, min_val: United_Value, max_val: United_Value, inclusive: Literal["both", "neither", "left", "right"] = "both") -> None:
        """
        Clamp all values in a column to the specified range.
        
        Args:
            column_key (CK): The column key to clamp
            min_val (United_Value): The minimum value
            max_val (United_Value): The maximum value
            inclusive (str): Which bounds to include ("both", "neither", "left", "right")
            
        Raises:
            United_Dataframe_Exception: If the dataframe is read-only, the column doesn't exist,
                                       or the column is not numeric
        """
        with self._wlock:
            if self._read_only:
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            
            if not self.has_column(column_key):
                raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")
            dataframe_column_name: str = self.internal_dataframe_column_string(column_key)
            if self.is_numeric(column_key):
                values: pd.Series = self._internal_canonical_dataframe[dataframe_column_name]
                clamped_values = values.clip(lower=min_val.canonical_float, upper=max_val.canonical_float, inclusive=inclusive)
                self._internal_canonical_dataframe[dataframe_column_name] = clamped_values
            else:
                raise United_Dataframe_Exception(f"Column '{column_key}' is not numeric.")

    # ----------- Column and Row Filtering ------------

    def filterfun_get_by_column_key_types(self, *column_key_types: type[CK_CF]) -> "United_Dataframe[CK_CF]":
        """
        Filter the dataframe by column key type.
        """
        with self._rlock:

            column_information_of_type: list[Column_Information[CK_CF]] = self.column_information_of_type(*column_key_types)
            return United_Dataframe[CK_CF].create_from_dataframe_and_column_information_list(
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
                raise United_Dataframe_Exception("The dataframe is read-only. Please create a new dataframe instead.")
            
            column_keys_to_keep: list[CK] = self.column_keys_of_type(*column_key_types)
            column_keys_to_remove: list[CK] = [column_key for column_key in self._column_keys if column_key not in column_keys_to_keep]
            for column_key in column_keys_to_remove:
                self.remove_column(column_key)

    def filterfun_by_filterdict(self, filter_dict: 
                     dict[CK, United_Value|str|bool|datetime]|
                     dict[CK, United_Value]|
                     dict[CK, str]|
                     dict[CK, bool]|
                     dict[CK, datetime]) -> "United_Dataframe[CK]":
        """
        Filter the dataframe by a dictionary of column keys and values.

        Args:
            filter_dict (dict[CK, United_Value|str|bool]): A dictionary of column keys and values to filter by

        Returns:
            United_Dataframe[CK]: A new dataframe with the filtered rows.

        Raises:
            United_Dataframe_Exception: If the column key does not exist in the dataframe.
        """
        with self._rlock:
            filtered_df = self._internal_canonical_dataframe.copy()
            for column_key, value in filter_dict.items():
                if self.has_column(column_key):
                    value_type: Value_Type = self.value_type(column_key)
                    if isinstance(value, United_Value):
                        value_coerced: float|int|str|bool|datetime = value_type.coerce_to_type(value.canonical_value)
                    else:
                        value_coerced: float|int|str|bool|datetime = value_type.coerce_to_type(value)
                    filtered_df = filtered_df[filtered_df[self.internal_dataframe_column_string(column_key)] == value_coerced]
                else:
                    raise United_Dataframe_Exception(f"Column key {column_key} does not exist in the dataframe.")

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
    def from_json(cls, data: dict) -> "United_Dataframe":
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
    
    # ----------- Constructors ------------

    @classmethod
    def create_from_pandas_dataframe(
        cls,
        dataframe: pd.DataFrame,
        dataframe_column_names: dict[CK, str],
        column_keys: list[CK] = [],
        units: list[Unit]|dict[CK, Unit] = [],
        value_types: list[Value_Type]|dict[CK, Value_Type] = [],
        internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = SIMPLE_UNITED_FORMATTER,
        deepcopy_dataframe: bool = True,
        ) -> "United_Dataframe[CK]":
        """
        Create a United_Dataframe from a pandas DataFrame.
        
        This is the primary factory method for creating United_Dataframes. It handles
        unit conversion, column key assignment, and value type specification.
        
        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to convert
            dataframe_column_names (dict[CK, str]): Dictionary of column keys and their corresponding names in the provided dataframe. The columns will be renamed to the proper names using the internal_column_name_formatter.
            column_keys (list[CK]): List of column keys. If empty, string column names
                                          will be used as keys
            units (list[Unit]|dict[CK, Unit]): Units for each column. Can be specified
                                                      as a list (in column order) or dict (by column key).
                                                      If empty, NO_NUMBER units will be used
            value_types (list[Value_Type]|dict[CK, Value_Type]): Value types for each column.
                                                                        Can be specified as a list (in column order)
                                                                        or dict (by column key). If empty, types
                                                                        will be inferred from the DataFrame
            internal_column_name_formatter (Callable): Function to format internal column names
                                                      (default: SIMPLE_UNITED_FORMATTER)
            deepcopy_dataframe (bool): Whether to create a deep copy of the input DataFrame
                                      (default: True)
        
        Returns:
            United_Dataframe[CK]: A new United_Dataframe instance
            
        Raises:
            United_Dataframe_Exception: If the number of columns doesn't match the provided
                                       metadata, or if units/value_types are incompatible
                                       
        Examples:
            # Create with string column keys and inferred types
            df = pd.DataFrame({'A': [1, 2, 3], 'B': [4.0, 5.0, 6.0]})
            united_df = United_Dataframe.from_pandas_dataframe(df)
            
            # Create with custom units
            units = {'A': Unit.meter, 'B': Unit.second}
            united_df = United_Dataframe.from_pandas_dataframe(df, units=units)
            
            # Create with custom column keys and value types
            column_keys = [CustomKey('col1'), CustomKey('col2')]
            value_types = [Value_Type.INT64, Value_Type.FLOAT64]
            united_df = United_Dataframe.from_pandas_dataframe(
                df, column_keys=column_keys, value_types=value_types
            )
        """

        if len(column_keys) != len(units) or len(column_keys) != len(value_types):
            raise United_Dataframe_Exception(f"The number of column keys, units, and value types must be the same. Got {len(column_keys)}, {len(units)}, and {len(value_types)}.")
        
        display_units_dict: dict[CK, Unit] = {}
        value_type_dict: dict[CK, Value_Type] = {}
        si_quantity_dict: dict[CK, SI_QUANTITY] = {}
        if isinstance(units, list):
            display_units_dict = {column_key: unit for column_key, unit in zip(column_keys, units)}
        if isinstance(value_types, list):
            value_type_dict = {column_key: value_type for column_key, value_type in zip(column_keys, value_types)}
        for column_key, display_unit in display_units_dict.items():
            si_quantity_dict[column_key] = display_unit.si_quantity

        united_dataframe: "United_Dataframe[CK]" = cls(
            dataframe.copy(deep=deepcopy_dataframe),
            dataframe_column_names,
            column_keys,
            si_quantity_dict,
            display_units_dict,
            value_type_dict,
            internal_column_name_formatter,
            _INTERNAL_INIT_TOKEN)
        
        return united_dataframe
    
    @classmethod
    def create_empty_dataframe_from_column_information(
        cls,
        column_information: list[Column_Information[CK]],
        initial_number_of_rows: int = 0,
        internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = SIMPLE_UNITED_FORMATTER
        ) -> "United_Dataframe[CK]":

        column_keys: list[CK] = [column_information.column_key for column_information in column_information]
        units: list[Unit] = [column_information.display_unit for column_information in column_information]
        value_types: list[Value_Type] = [column_information.value_type for column_information in column_information]

        return cls.create_empty_dataframe(column_keys, units, value_types, initial_number_of_rows, internal_column_name_formatter)

    @classmethod
    def create_from_dataframe_and_column_information_list(
        cls,
        dataframe: pd.DataFrame,
        dataframe_column_names: dict[CK, str]|None,
        column_information_list: list[Column_Information[CK]],
        internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = SIMPLE_UNITED_FORMATTER,
        deepcopy_dataframe: bool = True
    ) -> "United_Dataframe[CK]":
        
        column_keys: list[CK] = [column_information.column_key for column_information in column_information_list]
        display_unit_dict: dict[CK, Unit] = {column_information.column_key: column_information.display_unit for column_information in column_information_list}
        si_quantity_dict: dict[CK, SI_QUANTITY] = {column_information.column_key: column_information.si_quantity for column_information in column_information_list}
        value_type_dict: dict[CK, Value_Type] = {column_information.column_key: column_information.value_type for column_information in column_information_list}

        if len(dataframe.columns) != len(column_keys):
            raise United_Dataframe_Exception(f"The number of columns in the dataframe and the number of column keys must be the same. Got {len(dataframe.columns)} and {len(column_keys)}.")

        united_dataframe: United_Dataframe[CK] = United_Dataframe[CK](
            dataframe.copy(deep=deepcopy_dataframe),
            dataframe_column_names,
            column_keys,
            si_quantity_dict,
            display_unit_dict,
            value_type_dict,
            internal_column_name_formatter,
            _INTERNAL_INIT_TOKEN
        )

        return united_dataframe

    @classmethod
    def create_empty_dataframe(
        cls,
        column_keys: list[CK] = [],
        units: list[Unit]|dict[CK, Unit] = [],
        value_types: list[Value_Type]|dict[CK, Value_Type] = [],
        initial_number_of_rows: int = 0,
        internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = SIMPLE_UNITED_FORMATTER
        ) -> "United_Dataframe[CK]":

        if len(column_keys) != len(units) or len(column_keys) != len(value_types):
            raise United_Dataframe_Exception(f"The number of column keys, units, and value types must be the same. Got {len(column_keys)}, {len(units)}, and {len(value_types)}.")

        display_unit_dict: dict[CK, Unit] = {}
        value_type_dict: dict[CK, Value_Type] = {}
        si_quantity_dict: dict[CK, SI_QUANTITY] = {}
        if isinstance(units, list):
            display_unit_dict = {column_key: unit for column_key, unit in zip(column_keys, units)}
        if isinstance(value_types, list):
            value_type_dict = {column_key: value_type for column_key, value_type in zip(column_keys, value_types)}
        for column_key, display_unit in display_unit_dict.items():
            si_quantity_dict[column_key] = display_unit.si_quantity

        dataframe_column_names: list[str] = [internal_column_name_formatter(United_Dataframe[CK].column_key_to_string(column_key), display_unit_dict[column_key], value_type_dict[column_key]) for column_key in column_keys]

        dataframe = pd.DataFrame(
            data={col: [pd.NA] * initial_number_of_rows for col in dataframe_column_names}
        )

        return cls(
            dataframe,
            None,
            column_keys,
            si_quantity_dict,
            display_unit_dict,
            value_type_dict,
            internal_column_name_formatter,
            _INTERNAL_INIT_TOKEN)
    
    @classmethod
    def create_from_row_value_dicts_and_column_information_list(
        cls,
        row_value_dicts: list[dict[CK, United_Value]],
        column_information_list: list[Column_Information[CK]],
        internal_column_name_formatter: Callable[[str, Unit, Value_Type], str] = SIMPLE_UNITED_FORMATTER
    ) -> "United_Dataframe[CK]":
        
        column_names: list[str] = [internal_column_name_formatter(United_Dataframe[CK].column_key_to_string(column_information.column_key), column_information.display_unit, column_information.value_type) for column_information in column_information_list]
        canonical_dataframe: pd.DataFrame = pd.DataFrame(row_value_dicts, columns=column_names, index=range(len(row_value_dicts)))

        united_dataframe: United_Dataframe[CK] = United_Dataframe[CK].create_from_dataframe_and_column_information_list(
            canonical_dataframe,
            None,
            column_information_list,
            internal_column_name_formatter)

        for row_index, row_value_dict in enumerate(row_value_dicts):
            for column_key, value in row_value_dict.items():
                united_dataframe.set_cell_value(row_index, column_key, value)

        return united_dataframe

    # ----------- GroupBy functionality ------------

    def groupby(self, by: CK|list[CK]|set[CK]) -> "GroupBy[CK]":
        """
        Group the dataframe by one or more columns.
        
        Args:
            by: Column index, column key, or list of columns to group by
            
        Returns:
            GroupBy: A GroupBy object for performing grouped operations
            
        Raises:
            United_Dataframe_Exception: If any of the grouping columns don't exist
            
        Examples:
            # Group by a single column
            grouped = df.groupby('category')
            
            # Group by multiple columns
            grouped = df.groupby(['category', 'region'])
            
            # Group by column index
            grouped = df.groupby(0)
        """
        with self._rlock:
            if isinstance(by, (Column_Key, str)):
                by = [by]
            
            # Validate all grouping columns exist
            for col in by:
                if not self.has_column(col):
                    raise United_Dataframe_Exception(f"Grouping column {col} does not exist in the dataframe.")
            
            return GroupBy(self, by)

class _ColumnAccessor(Generic[CK]):
    """
    Internal class for column-based access to cell values.
    
    Provides a pandas-like interface for accessing individual cells.
    """
    def __init__(self, parent: United_Dataframe[CK], column_key: CK):
        self._parent: United_Dataframe[CK] = parent
        self._column_key: CK = column_key
        
    def __getitem__(self, row: int) -> United_Value:
        return self._parent.get_cell_value(row, self._column_key)
    
    def __setitem__(self, row: int, value: United_Value):
        self._parent.set_cell_value(row, self._column_key, value)

    def __len__(self) -> int:
        return len(self._parent)
    
    def __iter__(self) -> Iterator[United_Value]:
        return self._parent.get_iterator_for_column(self._column_key)
    
    def __contains__(self, value: United_Value) -> bool:
        return value in self._parent.get_iterator_for_column(self._column_key)
    
    def as_numpy_array(self, unit: Unit) -> np.ndarray:
        return self._parent.column_values_as_numpy_array(self._column_key, unit)
    
    def as_pandas_series(self, unit: Unit) -> pd.Series:
        return self._parent.column_values_as_pandas_series(self._column_key, unit)
    
    def as_united_array(self, display_unit: Unit|None=None) -> UnitedArray:
        if display_unit is None:
            display_unit = self._parent.display_unit(self._column_key)
        return self._parent.column_values_as_united_array(self._column_key, display_unit)

    def sum(self) -> United_Value:
        return self._parent.colfun_sum(self._column_key)
    
    def mean(self) -> United_Value:
        return self._parent.colfun_mean(self._column_key)
    
    def std(self) -> United_Value:
        return self._parent.colfun_std(self._column_key)
    
    def min(self) -> United_Value:
        return self._parent.colfun_min(self._column_key)
    
    def max(self) -> United_Value:
        return self._parent.colfun_max(self._column_key)
    
    def unique(self) -> list[United_Value]:
        return self._parent.colfun_unique(self._column_key)
    
    def smallest_positive_nonzero_value(self) -> United_Value:
        return self._parent.colfun_smallest_positive_nonzero_value(self._column_key)
    
    def largest_positive_nonzero_value(self) -> United_Value:
        return self._parent.colfun_largest_positive_nonzero_value(self._column_key)
    
    def largest_negative_nonzero_value(self) -> United_Value:
        return self._parent.colfun_largest_negative_nonzero_value(self._column_key)
    
    def smallest_negative_nonzero_value(self) -> United_Value:
        return self._parent.colfun_smallest_negative_nonzero_value(self._column_key)
    
    @overload
    def count(self) -> dict[United_Value, int]: 
        """
        Count the number of occurrences of each unique value in the column.
        """
        return self._parent.colfun_count_value_occurances(self._column_key)

    @overload
    def count(self, value: United_Value) -> int:
        """
        Count the number of occurrences of the specified value in the column.
        """
        return self._parent.colfun_count_value_occurances(self._column_key, value)

    def count(self, value: United_Value|None=None) -> int|dict[United_Value, int]:
        if value is None:
            return self._parent.colfun_count_value_occurances(self._column_key)
        else:
            return self._parent.colfun_count_value_occurances(self._column_key, value)

    def isna(self) -> np.ndarray:
        """
        Return a boolean array indicating which values in this column are NA/NaN.
        
        This method works similarly to pandas Series.isna(), returning a numpy array
        with the same length as the column, where True indicates NA/NaN values.
        
        Returns:
            np.ndarray: Boolean array where True indicates NA values
            
        Examples:
            # Check which values in the column are NA
            na_mask = df['column_name'].isna()
            
            # Count NA values
            na_count = df['column_name'].isna().sum()
            
            # Filter to non-NA values
            non_na_values = df['column_name'][~df['column_name'].isna()]
        """
        dataframe_column_name: str = self._parent.internal_dataframe_column_string(self._column_key)
        
        # Get the pandas Series for this column
        column_series: pd.Series = self._parent._internal_canonical_dataframe[dataframe_column_name]
        
        # Check for NA values
        result_series: pd.Series = pd.isna(column_series)

        # Convert the pandas Series to a numpy array
        return result_series.to_numpy()
    
    def notna(self) -> np.ndarray:
        """
        Return a boolean array indicating which values in this column are not NA/NaN.
        
        This is the inverse of isna() - returns True for non-NA values.
        
        Returns:
            np.ndarray: Boolean array where True indicates non-NA values
            
        Examples:
            # Check which values in the column are not NA
            non_na_mask = df['column_name'].notna()
            
            # Count non-NA values
            non_na_count = df['column_name'].notna().sum()
            
            # Filter to non-NA values
            non_na_values = df['column_name'][df['column_name'].notna()]
        """
        return ~self.isna()
    
    def __ge__(self, other: United_Value) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (United_Value): The value to filter by

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x >= other})
    
    def __le__(self, other: United_Value) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (United_Value): The value to filter by

        Returns:    
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x <= other})
    
    def __gt__(self, other: United_Value) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (United_Value): The value to filter by

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x > other})  
    
    def __lt__(self, other: United_Value) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (United_Value): The value to filter by    

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x < other})
    
    @property
    def display_unit(self) -> Unit:
        return self._parent.display_unit(self._column_key)
    
    @property
    def si_quantity(self) -> SI_QUANTITY:
        return self._parent.si_quantity(self._column_key)
    
    @property
    def value_type(self) -> Value_Type:
        return self._parent.value_type(self._column_key)
    
    @property
    def column_key(self) -> CK:
        return self._column_key
    
    def row_index(self, value: United_Value, case: Literal["first", "last"] = "first") -> int:
        """
        Get the row index of the first occurrence of a value in a column.

        Args:
            value (United_Value): The value to get the row index of
            case (Literal["first", "last"]): The case to get the row index of

        Returns:
            int: The row index of the first occurrence of the value in the column. Returns -1 if the value is not found.
        """
        return self._parent.colfun_row_index(self._column_key, value, case)

class _RowAccessor(Generic[CK]):
    """
    Internal class for row-based access to cell values.
    """
    def __init__(self, parent: United_Dataframe[CK], row_index: int):
        self._parent: United_Dataframe[CK] = parent
        self._row_index: int = row_index

    def __getitem__(self, column_key: CK) -> United_Value:
        return self._parent.get_cell_value(self._row_index, column_key)
    
    def __setitem__(self, column_key: CK, value: United_Value):
        self._parent.set_cell_value(self._row_index, column_key, value)
    
    def __len__(self) -> int:
        return len(self._parent)
    
    def __iter__(self) -> Iterator[United_Value]:
        return self._parent.get_iterator_for_row(self._row_index)
    
    def __contains__(self, value: United_Value) -> bool:
        return value in self._parent.get_iterator_for_row(self._row_index)
    

class GroupBy(Generic[CK]):
    """
    A GroupBy object for performing grouped operations on United_Dataframe.
    
    This class provides pandas-like groupby functionality with unit awareness.
    """
    
    def __init__(self, dataframe: United_Dataframe[CK], by: list[CK]|set[CK]):
        """
        Initialize a GroupBy object.
        
        Args:
            dataframe: The United_Dataframe to group
            by: List of columns to group by
        """
        self._dataframe: United_Dataframe[CK] = dataframe
        self._by: list[CK] = list(by)
        self._groups: dict[tuple, United_Dataframe[CK]] | None = None
        self._group_keys: list[tuple] = []
    
    def _get_groups(self) -> dict[tuple, United_Dataframe[CK]]:
        """
        Get the grouped dataframes.
        
        Returns:
            dict: Dictionary mapping group keys to United_Dataframe instances
        """
        if self._groups is None:
            with self._dataframe._rlock:
                # Get the grouping columns as pandas Series
                group_columns = []
                for col in self._by:
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    group_columns.append(self._dataframe._internal_canonical_dataframe[col_name])
                
                # Create group keys
                group_keys = list(zip(*group_columns))
                unique_keys = list(set(group_keys))
                
                # Create groups
                self._groups = {}
                self._group_keys = unique_keys
                
                for key in unique_keys:
                    # Create boolean mask for this group
                    mask = pd.Series([gk == key for gk in group_keys], index=self._dataframe._internal_canonical_dataframe.index)
                    
                    # Get the subset of data for this group
                    subset_df = self._dataframe._internal_canonical_dataframe[mask].copy()
                    
                    # Create United_Dataframe for this group
                    group_df = United_Dataframe(
                        subset_df,
                        None,
                        self._dataframe._column_keys,
                        self._dataframe._si_quantities,
                        self._dataframe._display_units,
                        self._dataframe._value_types,
                        self._dataframe._internal_column_name_formatter,
                        _INTERNAL_INIT_TOKEN
                    )
                    
                    self._groups[key] = group_df
        
        return self._groups
    
    @property
    def groups(self) -> dict[tuple, United_Dataframe[CK]]:
        """
        Get the grouped dataframes.
        
        Returns:
            dict: Dictionary mapping group keys to United_Dataframe instances
        """
        return self._get_groups()
    
    @property
    def group_keys(self) -> list[tuple]:
        """
        Get the unique group keys.
        
        Returns:
            list: List of unique group key tuples
        """
        if self._group_keys is None:
            self._get_groups()
        return self._group_keys
    
    def size(self, result_column_key: CK) -> United_Dataframe[CK]:
        """
        Get the size of each group.
        
        Args:
            result_column_key (CK): The column key to use for the size results
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and group sizes
            
        Examples:
            # Get group sizes
            sizes = grouped.size('group_size')
            
            # Access the size for a specific group
            size_value = sizes.loc[0, 'group_size']
        """
        with self._dataframe._rlock:
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for i, col in enumerate(self._by):
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[col_name] = self._dataframe._internal_canonical_dataframe[col_name].iloc[0]
                
                # Add size result
                row_data[result_column_key] = len(group_df)
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys: list[CK] = self._dataframe.column_keys.copy()
            result_si_quantities: dict[CK, SI_QUANTITY] = {column_key: SI_QUANTITY.NO_NUMBER for column_key in result_column_keys}  # counts have no units
            result_display_units: dict[CK, Unit] = {column_key: NUMBER.one for column_key in result_column_keys}  # counts are numbers with unit "1"
            result_value_types: dict[CK, Value_Type] = {column_key: Value_Type.INT64 for column_key in result_column_keys}  # size is integer
            
            return United_Dataframe[CK](
                result_df,
                None,
                result_column_keys,
                result_si_quantities,
                result_display_units,
                result_value_types,
                self._dataframe._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN
            )
    
    def sum(self, numeric_only: bool = True, result_column_keys: list[CK] | None = None) -> United_Dataframe[CK]:
        """
        Calculate the sum of numeric columns for each group.
        
        Args:
            numeric_only (bool): If True, only sum numeric columns (default: True)
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and sum results
            
        Raises:
            United_Dataframe_Exception: If numeric_only is True and no numeric columns exist
            
        Examples:
            # Sum all numeric columns
            sums = grouped.sum()
            
            # Sum with custom result column keys
            sums = grouped.sum(result_column_keys=['sum_temp', 'sum_pressure'])
        """
        with self._dataframe._rlock:
            if numeric_only:
                numeric_columns = self._dataframe.get_numeric_column_keys()
                if not numeric_columns:
                    raise United_Dataframe_Exception("No numeric columns found for summation.")
            else:
                numeric_columns = self._dataframe.column_keys
            
            if result_column_keys is None:
                result_column_keys = numeric_columns.copy()
            elif len(result_column_keys) != len(numeric_columns):
                raise United_Dataframe_Exception(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to sum ({len(numeric_columns)}).")
            
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for i, col in enumerate(self._by):
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[col_name] = self._dataframe._internal_canonical_dataframe[col_name].iloc[0]
                
                # Add sum results
                for i, col in enumerate(numeric_columns):
                    result_key = result_column_keys[i]
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[result_key] = group_df._internal_canonical_dataframe[col_name].sum()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys_final: list[CK] = self._dataframe.column_keys.copy()
            result_si_quantities: dict[CK, SI_QUANTITY] = self._dataframe.si_quantities.copy()
            result_display_units: dict[CK, Unit] = self._dataframe.display_units.copy()
            result_value_types: dict[CK, Value_Type] = self._dataframe.value_types.copy()
            
            return United_Dataframe[CK](
                result_df,
                None,
                result_column_keys_final,
                result_si_quantities,
                result_display_units,
                result_value_types,
                self._dataframe._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN
            )
    
    def mean(self, numeric_only: bool = True, result_column_keys: list[CK] | None = None) -> United_Dataframe[CK]:
        """
        Calculate the mean of numeric columns for each group.
        
        Args:
            numeric_only (bool): If True, only calculate mean for numeric columns (default: True)
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and mean results
            
        Raises:
            United_Dataframe_Exception: If numeric_only is True and no numeric columns exist
            
        Examples:
            # Calculate mean of all numeric columns
            means = grouped.mean()
            
            # Calculate mean with custom result column keys
            means = grouped.mean(result_column_keys=['avg_temp', 'avg_pressure'])
        """
        with self._dataframe._rlock:
            if numeric_only:
                numeric_columns = self._dataframe.get_numeric_column_keys()
                if not numeric_columns:
                    raise United_Dataframe_Exception("No numeric columns found for mean calculation.")
            else:
                numeric_columns = self._dataframe.column_keys
            
            if result_column_keys is None:
                result_column_keys = numeric_columns.copy()
            elif len(result_column_keys) != len(numeric_columns):
                raise United_Dataframe_Exception(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to calculate mean ({len(numeric_columns)}).")
            
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for i, col in enumerate(self._by):
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[col_name] = self._dataframe._internal_canonical_dataframe[col_name].iloc[0]
                
                # Add mean results
                for i, col in enumerate(numeric_columns):
                    result_key = result_column_keys[i]
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[result_key] = group_df._internal_canonical_dataframe[col_name].mean()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys_final: list[CK] = self._dataframe.column_keys.copy()
            result_si_quantities: dict[CK, SI_QUANTITY] = self._dataframe.si_quantities.copy()
            result_display_units: dict[CK, Unit] = self._dataframe.display_units.copy()
            result_value_types: dict[CK, Value_Type] = self._dataframe.value_types.copy()
            
            return United_Dataframe[CK](
                result_df,
                None,
                result_column_keys_final,
                result_si_quantities,
                result_display_units,
                result_value_types,
                self._dataframe._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN
            )
    
    def count(self, result_column_keys: list[CK] | None = None) -> United_Dataframe[CK]:
        """
        Count non-null values in each group.
        
        Args:
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and count results
            
        Examples:
            # Count non-null values in all columns
            counts = grouped.count()
            
            # Count with custom result column keys
            counts = grouped.count(result_column_keys=['count_temp', 'count_pressure'])
        """
        with self._dataframe._rlock:
            columns_to_count = self._dataframe.column_keys
            
            if result_column_keys is None:
                result_column_keys = columns_to_count.copy()
            elif len(result_column_keys) != len(columns_to_count):
                raise United_Dataframe_Exception(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to count ({len(columns_to_count)}).")
            
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for i, col in enumerate(self._by):
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[col_name] = self._dataframe._internal_canonical_dataframe[col_name].iloc[0]
                
                # Add count results
                for i, col in enumerate(columns_to_count):
                    result_key = result_column_keys[i]
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[result_key] = group_df._internal_canonical_dataframe[col_name].count()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys_final: list[CK] = self._dataframe.column_keys.copy()
            result_si_quantities: dict[CK, SI_QUANTITY] = {column_key: SI_QUANTITY.NO_NUMBER for column_key in result_column_keys_final}  # counts have no units
            result_display_units: dict[CK, Unit] = {column_key: NUMBER.one for column_key in result_column_keys_final}  # counts are numbers with unit "1"
            result_value_types: dict[CK, Value_Type] = {column_key: Value_Type.INT64 for column_key in result_column_keys_final}  # counts are integers
            
            return United_Dataframe[CK](
                result_df,
                None,
                result_column_keys_final,
                result_si_quantities,
                result_display_units,
                result_value_types,
                self._dataframe._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN
            )
    
    def apply(self, func: Callable[[United_Dataframe[CK]], United_Value], result_column_key: CK) -> United_Dataframe[CK]:
        """
        Apply a function to each group for a specific column.
        
        Args:
            func: Function to apply to each group
            column: Column to apply the function to
            
        Returns:
            United_Dataframe: A dataframe with group keys and function results
        """
        with self._dataframe._rlock:
            if not self._dataframe.has_column(result_column_key):
                raise United_Dataframe_Exception(f"Column {result_column_key} does not exist in the dataframe.")
            
            group_results = []
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for i, col in enumerate(self._by):
                    col_name = self._dataframe.internal_dataframe_column_string(col)
                    row_data[col_name] = self._dataframe._internal_canonical_dataframe[col_name].iloc[0]
                
                # Apply function to the group
                result = func(group_df)
                if not isinstance(result, United_Value):
                    raise United_Dataframe_Exception(f"Function must return a United_Value, got {type(result)}")
                
                # Add result to row data
                col_name = self._dataframe.internal_dataframe_column_string(result_column_key)
                row_data[col_name] = result.canonical_value
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = self._dataframe.column_keys.copy()
            result_si_quantities = self._dataframe.si_quantities.copy()
            result_display_units = self._dataframe.display_units.copy()
            result_value_types = self._dataframe.value_types.copy()
            
            return United_Dataframe[CK](
                result_df,
                None,
                result_column_keys,
                result_si_quantities,
                result_display_units,
                result_value_types,
                self._dataframe._internal_column_name_formatter,
                _INTERNAL_INIT_TOKEN
            )

    def head(self, n: int = 1) -> United_Dataframe[CK]:
        """
        Get the first n rows from each group.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe: A dataframe containing the first n rows from each group
            
        Raises:
            United_Dataframe_Exception: If n is negative
            
        Examples:
            # Get first row from each group (default)
            grouped.head()
            
            # Get first 3 rows from each group
            grouped.head(3)
            
            # If a group has fewer than n rows, all rows from that group are returned
            grouped.head(10)  # Returns all rows if group has fewer than 10 rows
        """
        with self._dataframe._rlock:
            if n < 0:
                raise United_Dataframe_Exception(f"n must be non-negative, got {n}")
            
            all_head_rows = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                # Get the first n rows from this group (or all if group has fewer than n rows)
                actual_n = min(n, len(group_df))
                head_rows = group_df.rowfun_head(actual_n)
                all_head_rows.append(head_rows)
            
            # Concatenate all the head rows from all groups
            if all_head_rows:
                return United_Dataframe[CK].concatenate_dataframes(all_head_rows[0], *all_head_rows[1:])
            else:
                # Return empty dataframe with same structure
                return United_Dataframe[CK].create_empty_dataframe(
                    self._dataframe._column_keys,
                    self._dataframe._display_units,
                    self._dataframe._value_types
                )

    def first(self) -> United_Dataframe[CK]:
        """
        Get the first row from each group.
        
        Returns:
            United_Dataframe: A dataframe containing the first row from each group
            
        Raises:
            United_Dataframe_Exception: If any group is empty
            
        Examples:
            # Get the first row from each group
            first_rows = grouped.first()
            
            # Access the first row's values for a specific group
            first_rows.loc[0, 'column_name']
        """
        with self._dataframe._rlock:
            all_first_rows = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                if group_df.empty:
                    raise United_Dataframe_Exception(f"Cannot get first row from empty group {key}")
                
                first_row = group_df.rowfun_first()
                all_first_rows.append(first_row)
            
            # Concatenate all the first rows from all groups
            if all_first_rows:
                return United_Dataframe[CK].concatenate_dataframes(all_first_rows[0], *all_first_rows[1:])
            else:
                # Return empty dataframe with same structure
                return United_Dataframe[CK].create_empty_dataframe(
                    self._dataframe._column_keys,
                    self._dataframe._display_units,
                    self._dataframe._value_types
                )

    def tail(self, n: int = 1) -> United_Dataframe[CK]:
        """
        Get the last n rows from each group.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe: A dataframe containing the last n rows from each group
            
        Raises:
            United_Dataframe_Exception: If n is negative
            
        Examples:
            # Get last row from each group (default)
            grouped.tail()
            
            # Get last 3 rows from each group
            grouped.tail(3)
            
            # If a group has fewer than n rows, all rows from that group are returned
            grouped.tail(10)  # Returns all rows if group has fewer than 10 rows
        """
        with self._dataframe._rlock:
            if n < 0:
                raise United_Dataframe_Exception(f"n must be non-negative, got {n}")
            
            all_tail_rows = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                # Get the last n rows from this group (or all if group has fewer than n rows)
                actual_n = min(n, len(group_df))
                tail_rows = group_df.rowfun_tail(actual_n)
                all_tail_rows.append(tail_rows)
            
            # Concatenate all the tail rows from all groups
            if all_tail_rows:
                return United_Dataframe[CK].concatenate_dataframes(all_tail_rows[0], *all_tail_rows[1:])
            else:
                # Return empty dataframe with same structure
                return United_Dataframe[CK].create_empty_dataframe(
                    self._dataframe._column_keys,
                    self._dataframe._display_units,
                    self._dataframe._value_types
                )

    def last(self) -> United_Dataframe[CK]:
        """
        Get the last row from each group.
        
        Returns:
            United_Dataframe: A dataframe containing the last row from each group
            
        Raises:
            United_Dataframe_Exception: If any group is empty
            
        Examples:
            # Get the last row from each group
            last_rows = grouped.last()
            
            # Access the last row's values for a specific group
            last_rows.loc[0, 'column_name']
        """
        with self._dataframe._rlock:
            all_last_rows = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                if group_df.empty:
                    raise United_Dataframe_Exception(f"Cannot get last row from empty group {key}")
                
                last_row = group_df.rowfun_last()
                all_last_rows.append(last_row)
            
            # Concatenate all the last rows from all groups
            if all_last_rows:
                return United_Dataframe[CK].concatenate_dataframes(all_last_rows[0], *all_last_rows[1:])
            else:
                # Return empty dataframe with same structure
                return United_Dataframe[CK].create_empty_dataframe(
                    self._dataframe._column_keys,
                    self._dataframe._display_units,
                    self._dataframe._value_types
                )

    def get_filtered(self, filter_dict: dict[CK, United_Value|str|bool|datetime]) -> "GroupBy[CK]":
        """
        Filter each group by a dictionary of column keys and values.
        
        This method applies the same filtering criteria to each group individually,
        then returns a new GroupBy object with the filtered groups.
        
        Args:
            filter_dict (dict[CK, United_Value|str|bool|datetime]): A dictionary of column keys and values to filter by
            
        Returns:
            GroupBy[CK]: A new GroupBy object with filtered groups
            
        Raises:
            United_Dataframe_Exception: If any column key does not exist in the dataframe
            
        Examples:
            # Filter groups to only include rows where 'status' is 'active'
            filtered_groups = grouped.get_filtered({'status': 'active'})
            
            # Filter groups with multiple criteria
            filtered_groups = grouped.get_filtered({
                'status': 'active',
                'priority': United_Value.united_number(5, Unit.one)
            })
            
            # Apply operations to filtered groups
            result = filtered_groups.sum()
        """
        with self._dataframe._rlock:
            filtered_groups = {}
            
            for key in self.group_keys:
                group_df = self.groups[key]
                # Apply the same filtering to each group
                filtered_group = group_df.filterfun_by_filterdict(filter_dict)
                if not filtered_group.empty:
                    filtered_groups[key] = filtered_group
            
            # Create a new GroupBy object with the filtered groups
            # We need to reconstruct the original dataframe from the filtered groups
            if filtered_groups:
                all_filtered_rows = []
                for group_df in filtered_groups.values():
                    all_filtered_rows.append(group_df)
                
                # Concatenate all filtered groups into a single dataframe
                filtered_dataframe = United_Dataframe[CK].concatenate_dataframes(all_filtered_rows[0], *all_filtered_rows[1:])
                
                # Create a new GroupBy object with the same grouping columns
                return GroupBy[CK](filtered_dataframe, self._by)
            else:
                # If no groups remain after filtering, return an empty GroupBy
                empty_dataframe = United_Dataframe[CK].create_empty_dataframe(
                    self._dataframe._column_keys,
                    self._dataframe._display_units,
                    self._dataframe._value_types
                )
                return GroupBy[CK](empty_dataframe, self._by)

    def isna(self, subset: list[CK] | None = None) -> np.ndarray:
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
            na_mask = grouped.isna()
            
            # Check specific columns for NA values
            na_mask = grouped.isna(['column1', 'column2'])
            
            # Use the mask for filtering
            non_na_rows = grouped[~grouped.isna().any(axis=1)]
        """
        return self._dataframe.maskfun_isna(subset)
    
    def notna(self, subset: list[CK] | None = None) -> np.ndarray:
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
            non_na_mask = grouped.notna()
            
            # Check specific columns for non-NA values
            non_na_mask = grouped.notna(['column1', 'column2'])
            
            # Use the mask for filtering
            non_na_rows = grouped[grouped.notna().all(axis=1)]
        """
        return ~self.isna(subset)