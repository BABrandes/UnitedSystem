"""
Core functionality mixin for UnitedDataframe.

Contains basic properties, initialization helpers, and core utility methods.
"""

from typing import Any, Optional, Sequence, TYPE_CHECKING, Iterable, overload, Union, cast
from collections.abc import Sequence
import pandas as pd
import numpy as np

from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._dataframe.internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter
from ..._units_and_dimension.unit import Unit

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class CoreMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Core functionality mixin for UnitedDataframe.
    
    Provides basic properties, initialization helpers, and core utility methods
    that are used throughout the dataframe implementation.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """
    
    # Basic properties and information
    def __len__(self) -> int:
        """
        Return the number of rows in the dataframe.
        
        Returns:
            int: The number of rows in the dataframe
        """
        with self._rlock:
            return self._number_of_rows()
        
    def _number_of_rows(self) -> int:
        """
        Internal: Get the number of rows in the dataframe (no lock).
        """
        return len(self._internal_dataframe)
    
    def _number_of_columns(self) -> int:
        """
        Internal: Get the number of columns in the dataframe (no lock).
        """
        return len(self._column_keys)
    
    @property
    def internal_dataframe_column_name_formatter(self) -> InternalDataFrameColumnNameFormatter:
        """
        Get the internal dataframe column name formatter.
        """
        return self._internal_dataframe_column_name_formatter

    def has_unit(self, column_key: CK) -> bool:
        """
        Public: Check if a column has a unit (with lock).
        """
        with self._rlock:
            return self._unit_has(column_key)

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
            return len(self._internal_dataframe)

    @property
    def size(self) -> int:
        """
        Get the total number of elements in the dataframe.
        
        Returns:
            int: Total number of elements (rows × columns)
        """
        with self._rlock:
            return self._internal_dataframe.size

    @property
    def empty(self) -> bool:
        """
        Check if the dataframe is empty.
        
        Returns:
            bool: True if the dataframe has no rows, False otherwise
        """
        with self._rlock:
            return self._internal_dataframe.empty

    # Internal utilities
    def _create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Internal: Create the internal dataframe column name for a column (no lock).
        """
        return self._internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, self._column_units[column_key])

    def create_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Public: Create the internal dataframe column name for a column (with lock).

        Args:
            column_key (CK): The column key to create the internal dataframe column name for

        Returns:
            str: The internal dataframe column name for the column
        """
        with self._rlock:
            return self._create_internal_dataframe_column_name(column_key)

    def _get_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Internal: Get the internal dataframe column string for a column key (no lock).
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        return self._internal_dataframe_column_names[column_key]

    def get_internal_dataframe_column_name(self, column_key: CK) -> str:
        """
        Public: Get the internal dataframe column string for a column key (with lock).
        
        Args:
            column_key (CK): The column key
            
        Returns:
            str: The internal column string
        """
        with self._rlock:
            return self._get_internal_dataframe_column_name(column_key)
        
    def _get_internal_dataframe_column_names(self, column_key: CK|Sequence[CK]|None = None) -> list[str]:
        """
        Internal: Get the internal dataframe column strings for a list of column keys (no lock).
        """
        if column_key is None:
            column_key = self._column_keys
        if isinstance(column_key, Sequence):
            return [self._get_internal_dataframe_column_name(column_key) for column_key in column_key] # type: ignore
        else:
            return [self._get_internal_dataframe_column_name(column_key)]

    def get_internal_dataframe_column_names(self, column_key: CK|Sequence[CK]|None = None) -> list[str]:
        """
        Public: Get the internal dataframe column strings for a list of column keys (with lock).
        """
        with self._rlock:
            return self._get_internal_dataframe_column_names(column_key)

    @staticmethod
    def column_key_to_string(column_key: CK, internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter, column_unit: Optional[Unit]) -> str:
        """
        Public: Convert a column key to string. (with lock)
        """
        return internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_unit)
    
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

    # Utility methods
    def get_numeric_column_keys(self) -> list[CK]:
        """
        Get a list of column keys for numeric columns only.
        Returns:
            list[CK]: List of column keys for numeric columns
        """
        with self._rlock:
            return [column_key for column_key in self._column_keys if self._colkey_is_numeric(column_key)]
        
    def __repr__(self) -> str:
        """
        Return a string representation of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.__repr__()
        
    def _repr_html_(self) -> str:
        """
        Return an HTML representation of the dataframe for Jupyter notebook display.
        
        The internal dataframe already contains unit information in column names,
        so we can directly expose the pandas HTML representation.
        """
        with self._rlock:
            return self._internal_dataframe._repr_html_() # type: ignore
        
    def to_html(self, **kwargs: Any) -> str:
        """
        Return an HTML representation of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.to_html(**kwargs) # type: ignore
        
    def __contains__(self, item: Any) -> bool:
        """
        Check if the dataframe contains a column key.
        """
        with self._rlock:
            return item in self._column_keys
        
    @property
    def shape(self) -> tuple[int, int]:
        """
        Return the shape of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.shape
        
    @property
    def dtypes(self) -> pd.Series: # type: ignore
        """
        Return the dtypes of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.dtypes # type: ignore
        
    @property
    def index(self) -> pd.Index: # type: ignore[reportUnknownReturnType]
        """
        Return the index of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.index # type: ignore
        
    @property
    def columns(self) -> pd.Index: # type: ignore[reportUnknownReturnType]
        """
        Return the columns of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.columns # type: ignore

    @property
    def values(self) -> np.ndarray:
        """
        Return the values of the dataframe.
        """
        with self._rlock:
            return self._internal_dataframe.values
        
    def head(self, n: int = 5) -> "UnitedDataframe[CK]":
        """
        Return the first n rows of the dataframe.
        """
        with self._rlock:
            return self._create_with_replaced_internal_dataframe(self._internal_dataframe.head(n), copy_dataframe=False)
        
    def tail(self, n: int = 5) -> "UnitedDataframe[CK]":
        """
        Return the last n rows of the dataframe.
        """
        with self._rlock:
            return self._create_with_replaced_internal_dataframe(self._internal_dataframe.tail(n), copy_dataframe=False)
        
    @overload
    def get_pandas_dataframe(self, deepcopy: bool = True, column_keys: dict[CK, Union[str, Unit, tuple[str, Unit], tuple[Unit, str]]] = {}) -> pd.DataFrame:
        ...
    @overload
    def get_pandas_dataframe(self, deepcopy: bool = True, column_keys: Iterable[CK] = ()) -> pd.DataFrame:
        ...
    def get_pandas_dataframe(self, deepcopy: bool = True, column_keys: dict[CK, str]|Iterable[CK] = {}) -> pd.DataFrame:
        """
        Return a pandas dataframe with the specified column keys. The target column names can be provided as a dictionary.

        Args:
            deepcopy (bool): Whether to make a deep copy of the dataframe (recommended).
            column_keys (Dict[CK, str]|Iterable[CK]): A dictionary of column keys to column names or an iterable of column keys.

        Returns:
            pd.DataFrame: A pandas dataframe with the specified column keys.
        """
        with self._rlock:

            if next(iter(column_keys), None) is None:
                return self._internal_dataframe.copy() if deepcopy else self._internal_dataframe
            else:
                if isinstance(column_keys, dict):
                    internal_column_names_to_extract: list[str] = []
                    rename_dict: dict[str, str] = {}
                    unit_transformation_dict: dict[str, tuple[Unit, Unit]] = {}
                    for column_key, target_information in column_keys.items(): # type: ignore
                        if not self._colkey_exists(column_key):
                            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
                        if isinstance(target_information, str):
                            target_column_name: str = target_information
                            unit: Optional[Unit] = self._column_units[column_key]
                        elif isinstance(target_information, Unit):
                            target_column_name: str = self._get_internal_dataframe_column_name(column_key)
                            unit: Optional[Unit] = target_information
                        else:
                            if isinstance(target_information[0], str) and isinstance(target_information[1], Unit):
                                target_column_name: str = cast(str, target_information[0])
                                unit: Optional[Unit] = cast(Unit, target_information[1])
                            elif isinstance(target_information[0], Unit) and isinstance(target_information[1], str):
                                target_column_name: str = cast(str, target_information[1])
                                unit: Optional[Unit] = cast(Unit, target_information[0])
                            else:
                                raise ValueError(f"Invalid target information for column key {column_key}: {target_information}")
                        internal_column_name: str = self._get_internal_dataframe_column_name(column_key)
                        internal_column_names_to_extract.append(internal_column_name)
                        rename_dict[internal_column_name] = target_column_name
                        if unit is not None:
                            if not self._unit_has(column_key):
                                raise ValueError(f"Column key {column_key} has no unit, but a target unit was provided.")
                            unit_transformation_dict[target_column_name] = (self._unit_get(column_key), unit)

                    dataframe_to_return: pd.DataFrame = self._internal_dataframe[internal_column_names_to_extract].copy(deep=deepcopy).rename(columns=rename_dict)
                    for column_name, (current_unit, target_unit) in unit_transformation_dict.items():
                        if not Unit.effectively_equal(current_unit, target_unit) and deepcopy == False:
                            raise ValueError(f"One cannot do a unit conversion without deepcopy=False as this would change the original dataframe.")
                        dataframe_to_return[column_name] = Unit.convert(dataframe_to_return[column_name], current_unit, target_unit) #type: ignore
                    return dataframe_to_return
                else:
                    internal_column_names_to_extract: list[str] = []
                    for column_key in column_keys:
                        if not self._colkey_exists(column_key):
                            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
                        internal_column_name: str = self._get_internal_dataframe_column_name(column_key)
                        internal_column_names_to_extract.append(internal_column_name)
                    return self._internal_dataframe[internal_column_names_to_extract].copy(deep=deepcopy)
                

    def append(self, *others: "UnitedDataframe[CK]") -> "UnitedDataframe[CK]":
        """
        Appends dataframes to the current dataframe. The units of the columns may be different, but they must be compatible.
        If the units are effectively the same, this method will be much faster than if the units are different.

        Args:
            *others (UnitedDataframe[CK]): The dataframes to append to the current dataframe.

        Returns:
            UnitedDataframe[CK]: The dataframe with the appended dataframes.
        """

        def helper_append(other: "UnitedDataframe[CK]") -> pd.DataFrame:
            """
            Internal: Helper method for appending a dataframe. It returns a dataframe ready to be appended to the current dataframe.
            """

            # Check if the column keys are the same
            if set(self._column_keys) != set(other._column_keys):
                raise ValueError("The column keys of the two dataframes must be the same.")
            
            # Check if the column dimensions and types are the same for each column
            columns_that_need_unit_conversion: set[CK] = set()
            columns_that_only_need_renaming: set[CK] = set()
            for column_key in self._column_keys:
                if self._column_types[column_key] != other._column_types[column_key]:
                    raise ValueError(f"The column {column_key} has different types in the two dataframes.")
                if self._unit_has(column_key) != other._unit_has(column_key): # type: ignore
                    raise ValueError(f"The column {column_key} and {other._get_internal_dataframe_column_name(column_key)} must both have units or neither have units.")
                if self._unit_has(column_key):
                    unit: Unit = self._unit_get(column_key)
                    other_unit: Unit = other._unit_get(column_key) #type: ignore
                    if not Unit.compatible_to(unit, other_unit):
                        raise ValueError(f"The column {column_key} and {other._get_internal_dataframe_column_name(column_key)} must both have units that are compatible.")
                    if not unit == other_unit:
                        if Unit.effectively_equal_to(unit, other_unit):
                            columns_that_only_need_renaming.add(column_key)
                        else:
                            columns_that_need_unit_conversion.add(column_key)

            if len(columns_that_need_unit_conversion) == 0:
                # This is the cheap case, we can just rename the columns and concatenate

                # Copy the other dataframe and rename its columns to match self
                dataframe_to_append: pd.DataFrame = other._internal_dataframe.copy(deep=False)
                rename_dict: dict[str, str] = {}
                for column_key in columns_that_only_need_renaming:
                    rename_dict[other._get_internal_dataframe_column_name(column_key)] = self._get_internal_dataframe_column_name(column_key)
                
                # Only rename if there are columns to rename
                if rename_dict:
                    dataframe_to_append = dataframe_to_append.rename(columns=rename_dict, inplace=False)

                # Check if the renamed dataframe has the same column names as self
                if set(dataframe_to_append.columns) != set(self._internal_dataframe.columns):
                    raise AssertionError("The internal column names of the two dataframes must be the same.")
                
                return dataframe_to_append
            
            else:
                # This is the expensive case, we need to convert the units and concatenate

                # Copy the current dataframe
                united_dataframe_to_append: "UnitedDataframe[CK]" = other._copy(deep=False) # type: ignore

                # Convert the columns that need unit conversion
                for column_key in columns_that_need_unit_conversion:
                    united_dataframe_to_append._unit_change(column_key, self._unit_get(column_key)) #type: ignore

                # After unit conversion, the column names should match
                if set(united_dataframe_to_append._internal_dataframe.columns) != set(self._internal_dataframe.columns):
                    raise AssertionError("After unit conversion, the internal column names of the two dataframes must be the same.")
                
                return united_dataframe_to_append._internal_dataframe

        with self._rlock:
            list_of_dataframes_to_append: list[pd.DataFrame] = [self._internal_dataframe]
            for other in others:
                list_of_dataframes_to_append.append(helper_append(other))
            concatenated_df = pd.concat(list_of_dataframes_to_append, axis=0)
            return self._create_with_replaced_internal_dataframe(concatenated_df, copy_dataframe=False)