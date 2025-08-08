"""
Constructor operations mixin for UnitedDataframe.

Contains all class factory methods and constructor operations,
including creating empty dataframes, from arrays, and other construction patterns.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import TYPE_CHECKING, Union, cast, Optional, Any, overload, Type
from collections.abc import Sequence
import pandas as pd
import numpy as np
from pandas._typing import Dtype

from .dataframe_protocol import CK, UnitedDataframeProtocol
from ..._dataframe.column_type import ColumnType
from ..._units_and_dimension.dimension import Dimension
from ..._units_and_dimension.unit import Unit
from ..._dataframe.internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter, SimpleInternalDataFrameNameFormatter
from ..._arrays.base_united_array import BaseUnitedArray
from ..._scalars.united_scalar import UnitedScalar
from ..._utils.value_type import VALUE_TYPE
from ..._utils.array_type import ARRAY_TYPE, ARRAY_TYPE_RUNTIME

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

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
    def create_from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        columns: Optional[dict[CK, tuple[ColumnType, str, Optional[Unit|Dimension]]|tuple[ColumnType, str]]],
        column_key_type: Type[CK],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        deepcopy: bool = True,
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a pandas DataFrame.

        If columns is None, the column keys, source column names, column types, and column units are inferred from the dataframe.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to create the UnitedDataframe from.
            columns (dict[CK, Tuple[ColumnType, str, Optional[Unit|Dimension]|Tuple[ColumnType, str]]]): The columns to create the UnitedDataframe from. The column key is the key of the column in the UnitedDataframe, the tuple is the source column name, the column type, and the column unit.
            internal_dataframe_column_name_formatter (InternalDataFrameColumnNameFormatter): The internal dataframe column name formatter.
            deepcopy (bool): Whether to deepcopy the dataframe.
            read_only (bool): Whether the UnitedDataframe should be read-only.

        Returns:
            UnitedDataframe: The UnitedDataframe created from the pandas DataFrame.
        """

        # Generate the column keys, source column names, column types, and column units
        column_keys: list[CK] = []
        source_column_names: dict[CK, str] = {}
        column_types: dict[CK, ColumnType] = {}
        column_units: dict[CK, Optional[Unit]] = {}
        if columns is None:
            for column_name in dataframe.columns:
                _column_key, column_unit = internal_dataframe_column_name_formatter.retrieve_from_internal_dataframe_column_name(column_name, column_key_type)
                column_key: CK = cast(CK, _column_key) # type: ignore
                column_types[column_key] = ColumnType.from_dtype(dataframe[column_name].dtype, has_unit=column_unit is not None) # type: ignore
                source_column_names[column_key] = column_name
                column_units[column_key] = column_unit
        else:
            column_keys = list(columns.keys())
            for column_key, value in columns.items():
                if len(value) == 3:
                    column_types[column_key] = value[0]
                    source_column_names[column_key] = value[1]
                    if isinstance(value[2], Unit):
                        column_units[column_key] = value[2]
                    elif isinstance(value[2], Dimension):
                        column_units[column_key] = value[2].canonical_unit
                    else:
                        column_units[column_key] = None
                elif len(value) == 2:
                    column_types[column_key] = value[0]
                    source_column_names[column_key] = value[1]
                    column_units[column_key] = None
                else:
                    raise ValueError(f"Invalid column specification for column key {column_key}: {value}")
            
        # Copy the dataframe
        dataframe = dataframe[source_column_names.values()].copy(deep=deepcopy)

        # Rename the columns
        target_column_names: list[str] = []
        for index, column_key in enumerate(column_keys):
            if not source_column_names[column_key] in dataframe.columns:
                raise ValueError(f"Source column name {source_column_names[column_key]} not found in dataframe")
            target_column_names.append(internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_units[column_key]))
            dataframe.rename(columns={source_column_names[column_key]: target_column_names[index]}, inplace=True)

        # Rearrange the columns
        dataframe = dataframe[target_column_names]

        # Create the UnitedDataframe
        return cls._construct( # type: ignore
            dataframe=dataframe,
            column_keys=column_keys,
            column_types=column_types,
            column_units=column_units,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            read_only=read_only,
        )

    @classmethod
    def create_empty(
        cls,
        column_keys: Sequence[CK],
        column_types: dict[CK, ColumnType],
        column_units_or_dimensions: dict[CK, Union[Unit, Dimension, None]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter()
    ) -> "UnitedDataframe[CK]":
        """
        Create an empty UnitedDataframe with the specified structure.
        
        Args:
            column_keys (Sequence[CK]): Sequence of column keys
            column_types (dict[CK, ColumnType]): Column type mapping
            column_units_or_dimensions (dict[CK, Union[Unit, Dimension, None]]): Column unit or dimension mapping
            internal_dataframe_column_name_formatter (InternalDataFrameColumnNameFormatter): Name formatter object
            
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
    
    
    @overload
    @classmethod
    def create_from_data(
        cls,
        columns: dict[CK, tuple[ColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]]|tuple[ColumnType, Sequence[VALUE_TYPE]]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        ...
    @overload
    @classmethod
    def create_from_data(
        cls,
        columns: dict[CK, tuple[ColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]]],   
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        ...
    @overload
    @classmethod
    def create_from_data(
        cls,
        columns: dict[CK, tuple[ColumnType, Sequence[VALUE_TYPE]]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        ...    
    @overload
    @classmethod
    def create_from_data(
        cls,
        columns: dict[CK, 
                      tuple[ColumnType, Optional[Unit|Dimension], np.ndarray]|
                      tuple[ColumnType, np.ndarray]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        ...
    @overload
    @classmethod
    def create_from_data(
        cls,
        columns: dict[CK, 
                      tuple[ColumnType, Optional[Unit|Dimension], "pd.Series[Any]"]|
                      tuple[ColumnType, "pd.Series[Any]"]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        ...    
    @classmethod
    def create_from_data(
        cls,
        columns: 
        dict[CK, tuple[ColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[ColumnType, Sequence[VALUE_TYPE]]] |
        dict[CK, tuple[ColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]]] |
        dict[CK, tuple[ColumnType, Sequence[VALUE_TYPE]]] |
        dict[CK, tuple[ColumnType, Optional[Unit|Dimension], np.ndarray] | tuple[ColumnType, np.ndarray]] |
        dict[CK, tuple[ColumnType, Optional[Unit|Dimension], "pd.Series[Any]"] | tuple[ColumnType, "pd.Series[Any]"]] |
        dict[CK, tuple[ColumnType, Optional[Unit|Dimension], Union[ARRAY_TYPE, Sequence[VALUE_TYPE], np.ndarray, "pd.Series[Any]"]] | tuple[ColumnType, Union[ARRAY_TYPE, Sequence[VALUE_TYPE], np.ndarray, "pd.Series[Any]"]]] |
        dict[CK, Union[ARRAY_TYPE, Sequence[VALUE_TYPE], np.ndarray, "pd.Series[Any]"]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter=SimpleInternalDataFrameNameFormatter(),
        read_only: bool = False
    ) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a dictionary of information and values.

        columns: Variant 1) dict[CK, {ColumnType, Unit|Dimension|None, column_values}]
        columns: Variant 2) dict[CK, {ColumnType, column_values}]
        columns: Variant 3) dict[CK, ARRAY|list[SCALAR]]

        column_values can be: 
        - ARRAY_TYPE
        - list[VALUE_TYPE]
        - np.ndarray
        - pd.Series[Any]
        
        Args:
            columns (dict[CK, Tuple[ColumnType, Optional[Unit|Dimension], Union[ARRAY_TYPE, Sequence[VALUE_TYPE], np.ndarray, pd.Series[Any]]|Tuple[ColumnType, Union[ARRAY_TYPE, Sequence[VALUE_TYPE], np.ndarray, pd.Series[Any]]]]]): Dictionary mapping column keys to arrays
            internal_dataframe_column_name_formatter (InternalDataFrameColumnNameFormatter): Name formatter function
            read_only (bool): Whether the UnitedDataframe should be read-only.
            
        Returns:
            UnitedDataframe: New dataframe with data from arrays
        """
        # Step 1: Create the column keys
        column_keys: list[CK] = list(columns.keys())

        # Step 2: Determine the column units and types from the arrays or, if a list is given, from column_types and column_units_or_dimensions dictionaries
        column_units: dict[CK, Unit | None] = {}
        column_types: dict[CK, ColumnType] = {}
        column_values: dict[CK, Union[np.ndarray, "pd.Series[Any]"]] = {}
        def get_numpy_from_array(column_key: CK, array_or_list_or_numpy_array_or_pandas_series: ARRAY_TYPE) -> np.ndarray:
            unit: Optional[Unit] = column_units[column_key]
            if isinstance(array_or_list_or_numpy_array_or_pandas_series, BaseUnitedArray):
                if not column_types[column_key].check_item_compatibility(array_or_list_or_numpy_array_or_pandas_series, unit):
                    raise ValueError(f"Array {array_or_list_or_numpy_array_or_pandas_series} is not compatible with column unit {unit}")
                return array_or_list_or_numpy_array_or_pandas_series.get_as_numpy_array(target_unit=unit)
            else:
                # Non-UnitedArray
                return array_or_list_or_numpy_array_or_pandas_series.canonical_np_array
        def get_numpy_from_list_of_scalars(column_key: CK, array_or_list_or_numpy_array_or_pandas_series: list[VALUE_TYPE]) -> np.ndarray:
            list_of_values: list[Any] = []
            for item in array_or_list_or_numpy_array_or_pandas_series:
                if issubclass(type(item), UnitedScalar):
                    assert isinstance(item, UnitedScalar)
                    # UnitedScalar
                    scalar: UnitedScalar[Any, Any] = item # type: ignore
                    if not column_types[column_key].check_item_compatibility(scalar, column_units[column_key]):
                        raise ValueError(f"List item {item} is not compatible with column unit {column_units[column_key]}")
                    if column_units[column_key] is not None:
                        list_of_values.append(column_units[column_key].from_canonical_value(scalar.canonical_value)) # type: ignore
                    else:
                        list_of_values.append(scalar.canonical_value) # type: ignore
                else:
                    # Non-UnitedScalar
                    list_of_values.append(item)
            return np.array(list_of_values)
        for column_key, value in columns.items():
            if isinstance(value, tuple):
                # Tuple value
                tuple_value: tuple[ColumnType, Optional[Unit|Dimension], Union[ARRAY_TYPE, list[VALUE_TYPE], np.ndarray, "pd.Series[Any]"]]|tuple[ColumnType, Union[ARRAY_TYPE, list[VALUE_TYPE], np.ndarray, "pd.Series[Any]"]] = value # type: ignore
                if len(tuple_value) == 3:
                    column_types[column_key] = tuple_value[0]
                    if isinstance(tuple_value[1], Unit):
                        column_units[column_key] = tuple_value[1]
                    elif isinstance(tuple_value[1], Dimension):
                        column_units[column_key] = tuple_value[1].canonical_unit
                    else:
                        column_units[column_key] = None
                    array_or_list_or_numpy_array_or_pandas_series: Union[ARRAY_TYPE, list[VALUE_TYPE], np.ndarray, "pd.Series[Any]"] = tuple_value[2]
                elif len(tuple_value) == 2:
                    column_types[column_key] = tuple_value[0]
                    column_units[column_key] = None
                    array_or_list_or_numpy_array_or_pandas_series: Union[ARRAY_TYPE, list[VALUE_TYPE], np.ndarray, "pd.Series[Any]"] = tuple_value[1]
                else:
                    raise ValueError(f"Invalid column specification for column key {column_key}: {value}")
                
                if isinstance(array_or_list_or_numpy_array_or_pandas_series, ARRAY_TYPE_RUNTIME):
                    assert isinstance(array_or_list_or_numpy_array_or_pandas_series, ARRAY_TYPE)
                    column_values[column_key] = get_numpy_from_array(column_key, array_or_list_or_numpy_array_or_pandas_series)
                elif isinstance(array_or_list_or_numpy_array_or_pandas_series, list):
                    column_values[column_key] = get_numpy_from_list_of_scalars(column_key, array_or_list_or_numpy_array_or_pandas_series)
                elif isinstance(array_or_list_or_numpy_array_or_pandas_series, np.ndarray):
                    # NumpyArray
                    column_values[column_key] = array_or_list_or_numpy_array_or_pandas_series
                elif isinstance(array_or_list_or_numpy_array_or_pandas_series, pd.Series): # type: ignore
                    # PandasSeries
                    column_values[column_key] = array_or_list_or_numpy_array_or_pandas_series
                else:
                    raise ValueError(f"Invalid array or list or numpy array or pandas series for column key {column_key}: {array_or_list_or_numpy_array_or_pandas_series}")

            else:
                # Non-Tuple value
                if isinstance(value, ARRAY_TYPE_RUNTIME):
                    assert isinstance(value, ARRAY_TYPE)
                    column_types[column_key] = ColumnType.infer_approbiate_column_type(type(value)) # type: ignore
                    if isinstance(value, BaseUnitedArray):
                        column_units[column_key] = value.unit
                    else:
                        column_units[column_key] = None
                    column_values[column_key] = get_numpy_from_array(column_key, value)
                elif isinstance(value, list):
                    list_of_scalars: list[VALUE_TYPE] = value
                    if len(list_of_scalars) == 0:
                        raise ValueError(f"List {value} is empty")
                    column_types[column_key] = ColumnType.infer_approbiate_column_type(type(list_of_scalars[0])) # type: ignore
                    if isinstance(list_of_scalars[0], UnitedScalar):
                        united_scalar: UnitedScalar[Any, Any] = cast(UnitedScalar[Any, Any], list_of_scalars[0])
                        column_units[column_key] = united_scalar.unit
                    else:
                        column_units[column_key] = None
                    column_values[column_key] = get_numpy_from_list_of_scalars(column_key, value)
                else:
                    raise ValueError(f"Invalid array or list or numpy array or pandas series for column key {column_key}: {value}")
                
        # Step 3: Check that all column values have the same length
        length: int = 0
        for column_key in column_keys:
            if length == 0:
                length = len(column_values[column_key])
            elif len(column_values[column_key]) != length:
                raise ValueError(f"Column {column_key} has a different length than column {column_keys[0]}")
            
        # Step 4: Create empty pandas dataframe with proper column names
        internal_column_strings: dict[CK, str] = {}
        for column_key in column_keys:
            internal_column_strings[column_key] = internal_dataframe_column_name_formatter.create_internal_dataframe_column_name(column_key, column_units[column_key])
        dataframe: pd.DataFrame = pd.DataFrame(columns=list(internal_column_strings.values()), index=range(length))

        # Step 5: Set the column types
        for column_key in column_keys:
            dtype: Dtype = column_types[column_key].value.dataframe_storage_type
            dataframe[internal_column_strings[column_key]] = pd.Series(dtype=dtype)

        # Step 6: Fill the dataframe with the data
        for column_key in column_keys:
            dtype: Dtype = column_types[column_key].value.dataframe_storage_type
            dataframe[internal_column_strings[column_key]] = pd.Series(column_values[column_key], dtype=dtype)
        
        # Step 7: Create UnitedDataframe instance
        return cls._construct( # type: ignore
            dataframe=dataframe,
            column_keys=column_keys,
            column_types=column_types,
            column_units=column_units,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            read_only= read_only,
            copy_dataframe= False,
            rename_dataframe_columns= False
        )

    def _create_with_replaced_internal_dataframe(self, dataframe: pd.DataFrame, copy_dataframe: bool) -> "UnitedDataframe[CK]":
        """
        Internal: Create a UnitedDataframe from a pandas DataFrame. (no lock, no read-only check)
        The column names of the dataframe must match the column names of the UnitedDataframe.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to create the UnitedDataframe from.
            copy_dataframe (bool): Whether to copy the dataframe.

        Returns:
            UnitedDataframe: The UnitedDataframe created from the pandas DataFrame.
        """

        if not dataframe.columns.equals(self._internal_dataframe.columns): # type: ignore
            raise ValueError("The column names of the dataframe do not match the column names of the UnitedDataframe.")

        return self._construct(
            dataframe=dataframe,
            column_keys=self._column_keys,
            column_types=self._column_types.copy(),
            column_units=self._column_units.copy(),
            internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter,
            read_only=self._read_only,
            copy_dataframe=False,
            rename_dataframe_columns=False
        )

    # ----------- Copy Operations ------------

    def _copy(self, deep: bool = True) -> "UnitedDataframe[CK]":
        """
        Internal: Create a copy of the dataframe. (no lock, no read-only check)
        
        Args:
            deep: Whether to create a deep copy (default: True)
            
        Returns:
            UnitedDataframe: Copy of the dataframe
        """
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
        
    def copy(self, deep: bool = True) -> "UnitedDataframe[CK]":
        """
        Create a copy of the dataframe.
        
        Args:
            deep: Whether to create a deep copy (default: True)
            
        Returns:
            UnitedDataframe: Copy of the dataframe
        """
        with self._rlock:
            return self._copy(deep=deep)

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
            return self.create_empty(
                column_keys=self._column_keys,
                column_types=self._column_types,
                column_units_or_dimensions=column_units_or_dimensions,
                internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter
            )