"""
Constructor operations mixin for UnitedDataframe.

Contains all class factory methods and constructor operations,
including creating empty dataframes, from arrays, and other construction patterns.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Dict, List, TYPE_CHECKING, Union, cast
from collections.abc import Sequence
import pandas as pd
from pandas._typing import Dtype

from .dataframe_protocol import CK, UnitedDataframeProtocol
from ....column_type import ColumnType
from ....dimension import Dimension
from ....unit import Unit
from ...units.united import United
from ...dataframe.internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter, SimpleInternalDataFrameNameFormatter
from ....column_type import ARRAY_TYPE, LOWLEVEL_TYPE

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class ConstructorMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Constructor operations mixin for UnitedDataframe.
    
    Provides all class factory methods and constructor operations,
    including creating empty dataframes, from arrays, and other construction patterns.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Class Factory Methods ------------

    @classmethod
    def create_empty_dataframe(
        cls,
        column_keys: List[CK],
        column_types: Dict[CK, ColumnType],
        column_units_or_dimensions: Dict[CK, Union[Unit, Dimension, None]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter
    ) -> "UnitedDataframe[CK]":
        """
        Create an empty UnitedDataframe with the specified structure.
        
        Args:
            column_keys (List[CK]): List of column keys
            column_types (Dict[CK, ColumnType]): Column type mapping
            column_units_or_dimensions (Dict[CK, Union[Unit, Dimension, None]]): Column unit or dimension mapping
            internal_dataframe_name_formatter (InternalDataFrameNameFormatter[CK]): Name formatter object
            
        Returns:
            UnitedDataframe: Empty dataframe with specified structure
        """
        
        # Step 1: Check that all column keys are present in all dictionaries
        if set(column_keys) != set(column_types.keys()):
            raise ValueError("Column keys must be present in the column_types dictionary")
        if set(column_keys) != set(column_units_or_dimensions.keys()):
            raise ValueError("Column keys must be present in the units_or_dimensions dictionary")

        # Step 2: Set column units
        column_units: dict[CK, Unit | None] = {}
        for column_key in column_keys:
            if not column_key in column_units_or_dimensions:
                raise ValueError(f"Column key {column_key} not found in unit_or_dimension")
            unit_or_dimension: Unit | Dimension | None = column_units_or_dimensions[column_key]
            if isinstance(unit_or_dimension, Unit):
                column_units[column_key] = unit_or_dimension
            elif isinstance(unit_or_dimension, Dimension):
                column_units[column_key] = unit_or_dimension.canonical_unit
            else:
                column_units[column_key] = None
        
        # Step 3: Create empty pandas dataframe with proper column names
        internal_column_strings: dict[CK, str] = {}
        for column_key in column_keys:
            internal_column_strings[column_key] = internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_units[column_key])
        empty_df: pd.DataFrame = pd.DataFrame(columns=list(internal_column_strings.values()))

        # Step 4: Set the column types
        for column_key in column_keys:
            if not column_key in column_types:
                raise ValueError(f"Column key {column_key} not found in column_types dictionary")
            dtype: Dtype = column_types[column_key].value.dataframe_storage_type
            empty_df[internal_column_strings[column_key]] = pd.Series(dtype=dtype)
        
        # Step 5: Create UnitedDataframe instance
        return cls._construct( # type: ignore
            dataframe=empty_df,
            column_keys=column_keys,
            column_types=column_types,
            column_units=column_units,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            read_only= False,
            copy_dataframe= False,
            rename_dataframe_columns= False
        )

    @classmethod
    def create_dataframe_from_data(
        cls,
        arrays: Dict[CK, Union[ARRAY_TYPE, List[LOWLEVEL_TYPE]]],
        column_types: Dict[CK, ColumnType],
        column_units_or_dimensions: Dict[CK, Union[Unit, Dimension, None]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter()
    ) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a dictionary of arrays.
        
        Args:
            arrays (Dict[CK, BaseArray]): Dictionary mapping column keys to arrays
            column_types (Dict[CK, ColumnType]): Column type mapping
            column_units_or_dimensions (Dict[CK, Union[Unit, Dimension, None]]): Column unit or dimension mapping
            internal_dataframe_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: New dataframe with data from arrays
        """
        # Step 1: Create the column keys
        column_keys: list[CK] = list(arrays.keys())

        # Step 2: Determine the column units and types from the arrays or, if a list is given, from column_types and column_units_or_dimensions dictionaries
        column_units: dict[CK, Unit | None] = {}
        final_column_types: dict[CK, ColumnType] = {}
        for column_key in column_keys:
            array_or_list: Union[ARRAY_TYPE, List[LOWLEVEL_TYPE]] = arrays[column_key]
            if isinstance(array_or_list, ARRAY_TYPE):
                final_column_types[column_key] = ColumnType.infer_approbiate_column_type(type(array_or_list)) # type: ignore
                if isinstance(array_or_list, United):
                    column_units[column_key] = array_or_list.active_unit
                else:
                    column_units[column_key] = None
            elif column_key in column_types:
                final_column_types[column_key] = column_types[column_key]
                column_units_or_dimension: Unit | Dimension | None = column_units_or_dimensions[column_key]
                if isinstance(column_units_or_dimension, Unit):
                    column_units[column_key] = column_units_or_dimension
                elif isinstance(column_units_or_dimension, Dimension):
                    column_units[column_key] = column_units_or_dimension.canonical_unit
                else:
                    column_units[column_key] = None
            else:
                raise ValueError(f"Column key {column_key} not found in arrays or column_types dictionary")

        # Step 3: Check that all column keys are present in all dictionaries
        if set(column_keys) != set(final_column_types.keys()):
            raise ValueError("Column keys must be present in the column_types dictionary")
        if set(column_keys) != set(column_units.keys()):
            raise ValueError("Column keys must be present in the units_or_dimensions dictionary")
        
        # Step 4: Check that the column types are compatible with the column units
        for column_key in column_keys:
            column_type: ColumnType = final_column_types[column_key]
            column_unit: Unit | None = column_units[column_key]
            if column_type.has_unit == column_unit is not None:
                raise ValueError(f"Column type {column_type} is not compatible with column unit {column_unit}")
            
        # Step 5: Check that all arrays and list have the same length
        length: int = len(next(iter(arrays.values())))
        for column_key in column_keys:
            array_or_list: Union[ARRAY_TYPE, List[LOWLEVEL_TYPE]] = arrays[column_key]
            if isinstance(array_or_list, ARRAY_TYPE):
                if len(array_or_list) != length:
                    raise ValueError(f"Array or list {array_or_list} has length {len(array_or_list)}, but {length} is expected")

        # Step 6: Create empty pandas dataframe with proper column names
        internal_column_strings: dict[CK, str] = {}
        for column_key in column_keys:
            internal_column_strings[column_key] = internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_units[column_key])
        dataframe: pd.DataFrame = pd.DataFrame(columns=list(internal_column_strings.values()), index=range(length))

        # Step 6: Set the column types
        for column_key in column_keys:
            if not column_key in final_column_types:
                raise ValueError(f"Column key {column_key} not found in column_types dictionary")
            dtype: Dtype = final_column_types[column_key].value.dataframe_storage_type
            dataframe[internal_column_strings[column_key]] = pd.Series(dtype=dtype)

        # Step 7: Fill the dataframe with the data
        for column_key in column_keys:
            array_or_list: Union[ARRAY_TYPE, List[LOWLEVEL_TYPE]] = arrays[column_key]
            dataframe_storage_type: Dtype = final_column_types[column_key].value.dataframe_storage_type
            unit: Unit | None = column_units[column_key]
            if isinstance(array_or_list, ARRAY_TYPE):
                if isinstance(array_or_list, United):
                    pandas_series: pd.Series = array_or_list.get_pandas_series(dtype=dataframe_storage_type, target_unit=unit) # type: ignore
                else:
                    pandas_series: pd.Series = array_or_list.get_pandas_series(dtype=dataframe_storage_type) # type: ignore
            elif isinstance(arrays[column_key], list):
                pandas_series: pd.Series = pd.Series(array_or_list, dtype=dataframe_storage_type) # type: ignore
            else:
                raise ValueError(f"Array or list {arrays[column_key]} is not a valid array or list")
            dataframe[internal_column_strings[column_key]] = pandas_series
        
        # Step 8: Create UnitedDataframe instance
        return cls._construct( # type: ignore
            dataframe=dataframe,
            column_keys=column_keys,
            column_types=final_column_types,
            column_units=column_units,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            read_only= False,
            copy_dataframe= False,
            rename_dataframe_columns= False
        )

    @classmethod
    def create_dataframe_from_pandas_with_correct_column_names(
        cls,
        pandas_dataframe: pd.DataFrame,
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter,
        deep_copy: bool
    ) -> "UnitedDataframe[CK]":
        
        """
        Create a UnitedDataframe from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Source pandas DataFrame
            internal_dataframe_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: New dataframe with data from pandas DataFrame
        """

        # Step 1: Extract the column keys, units, and types from the pandas dataframe
        column_keys: list[CK] = []
        column_units: dict[CK, Unit | None] = {}
        column_types: dict[CK, ColumnType] = {}
        for column_name in pandas_dataframe.columns:
            _column_key, column_unit = internal_dataframe_column_name_formatter.retrieve_from_internal_dataframe_column_name(column_name)
            column_key: CK = cast(CK, _column_key)
            column_keys.append(column_key)
            column_units[column_key] = column_unit
            pandas_series: pd.Series = pandas_dataframe[column_name] # type: ignore
            # Infer column type based on dtype and whether unit was found in column name
            has_unit = column_unit is not None
            column_types[column_key] = ColumnType.from_dtype(pandas_series.dtype, has_unit=has_unit) # type: ignore
            if column_types[column_key].has_unit != has_unit:
                raise ValueError(f"Column type {column_types[column_key]} is not compatible with column unit {column_unit}")

        # Step 2: Create UnitedDataframe instance
        return cls._construct( # type: ignore
            dataframe=pandas_dataframe,
            column_keys=column_keys,
            column_types=column_types,
            column_units=column_units,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            read_only= False,
            copy_dataframe= deep_copy,
            rename_dataframe_columns= False
        )
    
    @classmethod
    def create_dataframe_from_pandas_with_incorrect_column_names(
        cls,
        pandas_dataframe: pd.DataFrame,
        column_key_mapping: dict[str, CK],
        column_types: Dict[CK, ColumnType],
        column_units_or_dimensions: Dict[CK, Union[Unit, Dimension, None]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter,
        deep_copy: bool
    ) -> "UnitedDataframe[CK]":
        
        """
        Create a UnitedDataframe from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Source pandas DataFrame
            column_key_mapping (bidict[str, CK]): Mapping from pandas column names to UnitedDataframe column keys
            column_types (Dict[CK, ColumnType]): Column type mapping
            column_units_or_dimensions (Dict[CK, Union[Unit, Dimension, None]]): Column unit or dimension mapping
            internal_dataframe_column_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: New dataframe with data from pandas DataFrame
        """

        # Step 1: Check that all column keys are present in all dictionaries
        column_keys: list[CK] = list(column_key_mapping.values())
        if set(column_keys) != set(column_types.keys()):
            raise ValueError("Column keys must be present in the column_types dictionary")
        if set(column_keys) != set(column_units_or_dimensions.keys()):
            raise ValueError("Column keys must be present in the units_or_dimensions dictionary")

        # Step 2: Set column units
        column_units: dict[CK, Unit | None] = {}
        for column_key in column_keys:
            if not column_key in column_units_or_dimensions:
                raise ValueError(f"Column key {column_key} not found in unit_or_dimension")
            unit_or_dimension: Unit | Dimension | None = column_units_or_dimensions[column_key]
            if isinstance(unit_or_dimension, Unit):
                column_units[column_key] = unit_or_dimension
            elif isinstance(unit_or_dimension, Dimension):
                column_units[column_key] = unit_or_dimension.canonical_unit
            else:
                column_units[column_key] = None

        # Step 3: Create the correct column names
        correct_column_names: dict[str, str] = {}
        # Check that column_key_mapping is a bijection
        if len(column_key_mapping) != len(set(column_key_mapping.values())):
            raise ValueError("column_key_mapping is not a bijection")
        for column_name in pandas_dataframe.columns:
            correct_column_names[column_name] = internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key_mapping[column_name], column_units[column_key_mapping[column_name]])
        if deep_copy:
            pandas_dataframe_with_correct_column_names: pd.DataFrame = pandas_dataframe.copy(deep=True)
        else:
            pandas_dataframe_with_correct_column_names = pandas_dataframe
        pandas_dataframe.rename(columns=correct_column_names, inplace=True)

        # Step 4: Create UnitedDataframe instance
        return cls._construct( # type: ignore
            dataframe=pandas_dataframe_with_correct_column_names,
            column_keys=column_keys,
            column_types=column_types,
            column_units=column_units,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            read_only= False,
            copy_dataframe= False,
            rename_dataframe_columns= False
        )
    
    def _create_with_replaced_dataframe(self, dataframe: pd.DataFrame) -> "UnitedDataframe[CK]":
        """
        Internal: Create a UnitedDataframe from a pandas DataFrame with incorrect column names. (No locks, no read-only check)
        """

        return self.__class__._construct( # type: ignore
            dataframe=dataframe,
            column_keys=self._column_keys,
            column_types=self._column_types,
            column_units=self._column_units,
            internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter,
            read_only=self._read_only,
            copy_dataframe=False,
            rename_dataframe_columns=False
        )
    
    def crop_dataframe(self, column_keys: Sequence[CK]|None = None, row_indices: slice|Sequence[int]|None = None) -> "UnitedDataframe[CK]":
        """
        Crop the dataframe to the number of rows in the dataframe.
        """
        with self._rlock:
            return self._crop_dataframe(column_keys, row_indices)

    def _crop_dataframe(self, column_keys: Sequence[CK]|None = None, row_indices: slice|Sequence[int]|None = None) -> "UnitedDataframe[CK]":
        """
        Internal: Crop the dataframe to the number of rows in the dataframe. (No locks, no read-only check)
        """

        if column_keys is None or len(column_keys) == 0:
            column_keys = self._column_keys

        internal_column_names_to_keep: dict[CK, str] = {}
        for column_key in column_keys:
            internal_column_names_to_keep[column_key] = self._get_internal_dataframe_column_name(column_key)
        dataframe_cropped: pd.DataFrame = self._internal_dataframe[internal_column_names_to_keep.values()].copy(deep=True)

        column_types_to_keep: dict[CK, ColumnType] = {}
        column_units_to_keep: dict[CK, Unit | None] = {}
        for column_key in column_keys:
            column_types_to_keep[column_key] = self._column_types[column_key]
            column_units_to_keep[column_key] = self._column_units[column_key]

        if isinstance(row_indices, slice):
            dataframe_cropped: pd.DataFrame = dataframe_cropped.iloc[row_indices]
        elif isinstance(row_indices, Sequence):
            for row_index in sorted(row_indices, reverse=True):
                dataframe_cropped = dataframe_cropped.drop(row_index)

        return cls._construct( # type: ignore
            dataframe=dataframe_cropped,
            column_keys=column_keys,
            column_types=column_types_to_keep,
            column_units=column_units_to_keep,
            internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter,
            read_only=self._read_only,
            copy_dataframe=False,
            rename_dataframe_columns=False
        )

    # ----------- Copy Operations ------------

    def copy(self, deep: bool = True) -> "UnitedDataframe[CK]":
        """
        Create a copy of the dataframe.
        
        Args:
            deep: Whether to create a deep copy (default: True)
            
        Returns:
            UnitedDataframe: Copy of the dataframe
        """
        with self._rlock:  # Full IDE support!
            
            # For copy, preserve the original column metadata instead of inferring from pandas dtypes
            dataframe_copy = self._internal_dataframe.copy(deep=deep)
            
            return self.__class__._construct( # type: ignore
                dataframe=dataframe_copy,
                column_keys=self._column_keys,
                column_types=self._column_types.copy(),
                column_units=self._column_units.copy(),
                internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter,
                read_only=False  # Copy should not be read-only by default
            )

    def copy_structure(self) -> "UnitedDataframe[CK]":
        """
        Create a copy of the dataframe structure (metadata) with empty data.
        
        Returns:
            UnitedDataframe[CK]: New empty dataframe with same structure
        """
        with self._rlock:
            column_units_or_dimensions: dict[CK, Union[Unit, Dimension, None]] = {}
            for column_key in self._column_keys:
                column_units_or_dimensions[column_key] = self._column_units[column_key]
            return self.create_empty_dataframe(
                column_keys=self._column_keys,
                column_types=self._column_types,
                column_units_or_dimensions=column_units_or_dimensions,
                internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter
            )