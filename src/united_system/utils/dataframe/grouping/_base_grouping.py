from typing import Generic, TypeVar, Tuple, Callable, cast, TYPE_CHECKING
from collections.abc import Sequence
from bidict import bidict
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from enum import Enum, auto

from ....column_key import ColumnKey
from ....column_type import SCALAR_TYPE, ColumnType, LOWLEVEL_TYPE
from ..accessors._row_accessor import RowAccessor
from ....unit import Unit
from ...scalars.united_scalar import UnitedScalar

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe


CK = TypeVar("CK", bound="ColumnKey|str")

@dataclass
class GroupingContainer(Generic[CK]):
    parent_united_dataframe: "UnitedDataframe[CK]"
    dataframe: pd.DataFrame
    internal_dataframe_column_names: bidict[CK, str]
    available_column_keys: list[CK]
    available_column_types: dict[CK, ColumnType]
    available_column_units: dict[CK, Unit|None]
    categorical_column_keys: list[CK]
    categorical_key_values: Tuple[LOWLEVEL_TYPE, ...]
    _united_dataframe: "UnitedDataframe[CK]|None" = field(init=False, repr=False, default=None)

    def united_dataframe(self, column_keys: Sequence[CK] | None = None) -> "UnitedDataframe[CK]":
        """
        Get the UnitedDataframe for this group.

        Args:
            column_keys: List of column keys to include in the UnitedDataframe, if None, all column keys are included

        Returns:
            "UnitedDataframe[CK]": The UnitedDataframe for this group
        """
        if column_keys is None:
            _column_keys: list[CK] = self.available_column_keys
        else:
            if not all(col in self.available_column_keys for col in column_keys):
                raise ValueError(f"Column keys {column_keys} not found in the available column keys {self.available_column_keys}")
            _column_keys: list[CK] = [col for col in column_keys if col in self.available_column_keys]

        _column_types: dict[CK, ColumnType] = {col: self.available_column_types[col] for col in _column_keys}
        _column_units: dict[CK, Unit|None] = {col: self.available_column_units[col] for col in _column_keys}

        if self._united_dataframe is None:
            self._united_dataframe = UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=self.dataframe,
                column_keys=_column_keys,
                column_types=_column_types,
                column_units=_column_units,
                internal_dataframe_column_name_formatter=self.parent_united_dataframe.internal_dataframe_column_name_formatter)
        return self._united_dataframe
    
    def count(self) -> int:
        return len(self.dataframe)
    
    def mean(self, column_keys_and_types: dict[CK, ColumnType]) -> dict[CK, float]:
        # Calculate the mean of the dataframe for each numeric column
        return {col: self.dataframe[col].mean() for col in column_keys_and_types if column_keys_and_types[col].is_numeric} # type: ignore
    
    def sum(self, column_keys_and_types: dict[CK, ColumnType]) -> dict[CK, float]:
        return {col: self.dataframe[col].sum() for col in column_keys_and_types if column_keys_and_types[col].is_numeric} # type: ignore
    
    def std(self, column_keys_and_types: dict[CK, ColumnType]) -> dict[CK, float]:
        return {col: self.dataframe[col].std() for col in column_keys_and_types if column_keys_and_types[col].is_numeric} # type: ignore

@dataclass
class ColumnInformation:
    column_type: ColumnType
    column_unit: Unit|None
    internal_dataframe_column_name: str

class BaseGrouping(Generic[CK]):
    
    def __init__(
            self,
            dataframe: "UnitedDataframe[CK]",
            by_unique_values_of_columns: Sequence[CK] = [],
            by_unique_results_of_row_functions: Sequence[Tuple[CK, Callable[["RowAccessor[CK]"], SCALAR_TYPE]]] = []):
        """
        Initialize a BaseGrouping object.
        
        Args:
            dataframe: The United_Dataframe for the grouping
            by_unique_values_of_columns: List of columns for the groupings
            by_unique_results_of_row_functions: List of tuples of column keys for the results of the row functions and functions to apply to each row to get a unique result for each grouping
        """

        # Store original dataframe
        self._dataframe: "UnitedDataframe[CK]" = dataframe
        
        # Initialize grouping infrastructure
        self._grouping_containers: list[GroupingContainer[CK]] = []
        self._available_column_keys: list[CK] = []
        self._rowfun_result_column_information: dict[CK, ColumnInformation] = {}
        self._categorical_column_information: dict[CK, ColumnInformation] = {}
        self._available_column_information: dict[CK, ColumnInformation] = {}
        
        # Create working copy of dataframe
        self._working_df: pd.DataFrame = dataframe._internal_dataframe.copy(deep=False) # type: ignore
        
        # Setup grouping columns
        self._setup_grouping_columns(by_unique_values_of_columns, by_unique_results_of_row_functions)
        
        # Evaluate row functions if any
        if by_unique_results_of_row_functions:
            self._evaluate_row_functions(by_unique_results_of_row_functions)

    def _setup_grouping_columns(
            self, 
            by_unique_values_of_columns: Sequence[CK], 
            by_unique_results_of_row_functions: Sequence[Tuple[CK, Callable[["RowAccessor[CK]"], SCALAR_TYPE]]]
    ) -> None:
        """Setup columns used for grouping."""
        # Add existing columns to group by
        for col in by_unique_values_of_columns:
            if not self._dataframe.colkey_exists(col):
                raise ValueError(f"Column key {col} not found in the dataframe.")
            self._available_column_keys.append(col)
            self._available_column_information[col] = ColumnInformation(
                column_type=self._dataframe.coltypes[col],
                column_unit=self._dataframe.units[col],
                internal_dataframe_column_name=self._dataframe.get_internal_dataframe_column_name(col)
            )
        
        # Add result columns for row functions
        temp_rowfun_column_keys: list[CK] = []
        temp_rowfun_internal_dataframe_column_names: list[str] = []
        for col, _ in by_unique_results_of_row_functions:
            if not self._dataframe.colkey_exists(col):
                raise ValueError(f"Column key {col} not found in the dataframe.")
            self._available_column_keys.append(col)
            temp_rowfun_column_keys.append(col)

        self._rowfun_result_column_information = {
            col: ColumnInformation(
                column_type=self._dataframe.coltypes[col],
                column_unit=self._dataframe.units[col],
                internal_dataframe_column_name=column_name
            ) for col, column_name in zip(temp_rowfun_column_keys, temp_rowfun_internal_dataframe_column_names)
        }

    def _evaluate_row_functions(
            self, 
            by_unique_results_of_row_functions: Sequence[Tuple[CK, Callable[["RowAccessor[CK]"], SCALAR_TYPE]]]
    ) -> None:
        """Evaluate row functions and store results."""

        result_types: dict[CK, type[SCALAR_TYPE]] = {}        
        first_evaluation: bool = False
        
        # Evaluate functions for each row
        for row_index in range(len(self._dataframe)):
            row_accessor: "RowAccessor[CK]" = RowAccessor(self._dataframe, row_index)
            
            for column_key, row_function in by_unique_results_of_row_functions:
                # Evaluate function
                result: SCALAR_TYPE = row_function(row_accessor)
                
                # Validate consistency on first evaluation
                if not first_evaluation:
                    self._validate_and_record_first_result(column_key, result, result_types)
                    # Set proper dtype for the column
                    column_name = self._rowfun_result_column_information[column_key].internal_dataframe_column_name
                    self._working_df[column_name] = pd.Series(dtype=self._rowfun_result_column_information[column_key].column_type.value.dataframe_storage_type)
                else:
                    self._validate_result_consistency(column_key, result, result_types)
                
                # Convert to low-level value and store
                column_information: ColumnInformation = self._rowfun_result_column_information[column_key]
                low_level_value: LOWLEVEL_TYPE = column_information.column_type.get_value_for_dataframe(result, column_information.column_unit) # type: ignore
                self._working_df.at[row_index, column_information.internal_dataframe_column_name] = low_level_value # type: ignore
            
            if not first_evaluation:
                first_evaluation = True

    def _validate_and_record_first_result(
            self, 
            column_key: CK, 
            result: SCALAR_TYPE, 
            result_types: dict[CK, type[SCALAR_TYPE]]
    ) -> None:
        """Validate and record the first result of a function."""

        # Record result type
        result_type: type[SCALAR_TYPE] = type(result) # type: ignore
        result_types[column_key] = result_type
        
        # Get unit
        unit: Unit|None = None
        if isinstance(result, UnitedScalar):
            unit = cast(Unit, result.unit) # type: ignore

        # Get internal dataframe column name
        internal_dataframe_column_name: str = self._dataframe.get_internal_dataframe_column_name(column_key)
        
        # Create column information
        column_information: ColumnInformation = ColumnInformation(
            column_type=ColumnType.infer_approbiate_column_type(result_type), # type: ignore[reportUnknownReturnType]
            column_unit=unit,
            internal_dataframe_column_name=internal_dataframe_column_name
        )
        self._rowfun_result_column_information[column_key] = column_information
        self._available_column_information[column_key] = column_information
        self._categorical_column_information[column_key] = column_information

    def _validate_result_consistency(
            self, 
            column_key: CK, 
            result: SCALAR_TYPE, 
            result_types: dict[CK, type[SCALAR_TYPE]]
    ) -> None:
        """Validate that subsequent results match the first result."""
        if not isinstance(result, result_types[column_key]):
            raise ValueError(f"Result type mismatch for function on column {column_key}")
        
        if isinstance(result, UnitedScalar):
            expected_unit = self._rowfun_result_column_information[column_key].column_unit
            if expected_unit is None or expected_unit.dimension != result.dimension: # type: ignore
                raise ValueError(f"Unit/dimension mismatch for function on column {column_key}")
        elif self._rowfun_result_column_information[column_key].column_unit is not None:
            raise ValueError(f"Expected non-united scalar for function on column {column_key}")

    ######################### Properties #########################
    
    def groupings(self, column_keys: Sequence[CK] | None = None) -> list["UnitedDataframe[CK]"]:
        """
        Get the grouped dataframes.
        
        Args:
            column_keys: List of column keys to include in the UnitedDataframe, if None, all column keys are included

        Returns:
            list: List of United_Dataframe instances for each group
        """
        return [group.united_dataframe(column_keys) for group in self._grouping_containers]
    
    @property
    def categorical_key_values(self) -> list[tuple[LOWLEVEL_TYPE, ...]]:
        """
        Get the group keys.
        
        Returns:
            list: List of group key tuples
        """
        return [group.categorical_key_values for group in self._grouping_containers]
    
    @property
    def categorical_column_keys(self) -> list[CK]:
        """
        Get the column keys of the group keys.
        
        Returns:
            list: List of column keys of the group keys
        """
        return list(self._categorical_column_information.keys())

    ######################### Core Aggregation Methods #########################

    class _NumericOperation(Enum):
        SUM = auto()
        MEAN = auto()
        STD = auto()

    def _aggregate_numeric(
            self, 
            operation: _NumericOperation, 
            keep_non_numeric_columns_set_to_nan: bool = False,
            ignore_rowfun_columns: bool = True,
            column_keys_to_aggregate: Sequence[CK] | None = None
    ) -> "UnitedDataframe[CK]":
        """
        Helper method for numeric aggregation operations (sum, mean, etc.).
        
        Args:
            operation (_NumericOperation): The pandas operation to apply
            keep_non_numeric_columns_set_to_nan (bool): If True, include non-numeric columns in result with NaN values
            ignore_rowfun_columns (bool): If True, skip columns created by row functions during grouping
            result_column_keys (Sequence[CK] | None): List of column keys for the result columns
            
        Returns:
            UnitedDataframe[CK]: A dataframe with group keys and aggregation results
        """
        with self._dataframe._rlock: # type: ignore


            # Step 1: Get the column keys to aggregate

            _column_keys_to_aggregate: list[CK] = []            
            for col, col_info in self._available_column_information.items():

                # Skip columns not in column_keys_to_aggregate if provided
                if column_keys_to_aggregate is not None:
                    if col not in column_keys_to_aggregate:
                        continue

                # Skip row function columns if flag is set
                if ignore_rowfun_columns and col in self._rowfun_result_column_information:
                    continue
                
                # Include numeric columns
                if col_info.column_type.is_numeric:
                    _column_keys_to_aggregate.append(col)
                # Include non-numeric columns if flag is set
                elif keep_non_numeric_columns_set_to_nan:
                    _column_keys_to_aggregate.append(col)
            
            # Step 2: Aggregate the data

            group_data: list[dict[CK, LOWLEVEL_TYPE]] = []
            
            for container in self._grouping_containers:
                row_data: dict[CK, LOWLEVEL_TYPE] = {}
                
                # Add group key values
                for col, col_info in self._categorical_column_information.items():
                    row_data[col] = container.categorical_key_values[list(self._categorical_column_information.keys()).index(col)]
                
                # Add aggregation results using GroupingContainer methods
                for i, col in enumerate(_column_keys_to_aggregate):
                    result_key = _column_keys_to_aggregate[i]
                    col_info = self._available_column_information[col]
                    
                    # Handle numeric vs non-numeric columns
                    if col_info.column_type.is_numeric:
                        # Use GroupingContainer methods
                        match operation:
                            case self._NumericOperation.SUM:
                                sum_results = container.sum({col: col_info.column_type})
                                row_data[result_key] = sum_results[col]
                            case self._NumericOperation.MEAN:
                                mean_results = container.mean({col: col_info.column_type})
                                row_data[result_key] = mean_results[col]
                            case self._NumericOperation.STD:
                                std_results = container.std({col: col_info.column_type})
                                row_data[result_key] = std_results[col]
                    else:
                        # Non-numeric column - set to NaN
                        row_data[result_key] = np.nan
                
                group_data.append(row_data)

            # Step 3: Create the result dataframe
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result using column information
            result_column_keys = list(self._categorical_column_information.keys()) + list(_column_keys_to_aggregate)
            result_column_types = {col: col_info.column_type for col, col_info in self._categorical_column_information.items()}
            result_column_units = {col: col_info.column_unit for col, col_info in self._categorical_column_information.items()}
            
            for i, col in enumerate(_column_keys_to_aggregate):
                result_key = result_column_keys[len(self._categorical_column_information) + i]
                col_info = self._available_column_information[col]
                
                if col_info.column_type.is_numeric:
                    result_column_types[result_key] = col_info.column_type
                    result_column_units[result_key] = col_info.column_unit
                else:
                    # Non-numeric column - use float type for NaN values
                    result_column_types[result_key] = ColumnType.FLOAT_64
                    result_column_units[result_key] = None

            return UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=result_df,
                column_keys=result_column_keys,
                column_types=result_column_types,
                column_units=result_column_units,
                internal_dataframe_column_name_formatter=self._dataframe.internal_dataframe_column_name_formatter,
                read_only=False,
                copy_dataframe=False,
                rename_dataframe_columns=False
            )

    def size(self, size_column_key: CK, size_column_type: ColumnType = ColumnType.INTEGER_64) -> "UnitedDataframe[CK]":
        """
        Get the size of each group.
        
        Args:
            size_column_key (CK): The column key to use for the size results
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and group sizes
        """
        with self._dataframe._rlock: # type: ignore
            group_data: list[dict[CK, LOWLEVEL_TYPE]] = []
            
            # Get the group data
            for container in self._grouping_containers:
                row_data: dict[CK, LOWLEVEL_TYPE] = {}
                
                # Add group key values
                for col in self._categorical_column_information.keys():
                    row_data[col] = container.categorical_key_values[list(self._categorical_column_information.keys()).index(col)]
                
                # Add size result using GroupingContainer method
                row_data[size_column_key] = container.count()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result using column information
            result_column_keys = list(self._categorical_column_information.keys()) + [size_column_key]
            result_column_types = {col: col_info.column_type for col, col_info in self._categorical_column_information.items()}
            result_column_types[size_column_key] = ColumnType.INTEGER_64
            result_column_units = {col: col_info.column_unit for col, col_info in self._categorical_column_information.items()}
            result_column_units[size_column_key] = None

            return UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=result_df,
                column_keys=result_column_keys,
                column_types=result_column_types,
                column_units=result_column_units,
                internal_dataframe_column_name_formatter=self._dataframe.internal_dataframe_column_name_formatter,
                read_only=False,
                copy_dataframe=False,
                rename_dataframe_columns=False
            )

    def sum(self, column_keys_to_aggregate: Sequence[CK] | None = None, keep_non_numeric_columns_set_to_nan: bool = False, ignore_rowfun_columns: bool = True) -> "UnitedDataframe[CK]":
        """
        Calculate the sum of numeric columns for each group.
        
        Args:
            keep_non_numeric_columns_set_to_nan (bool): If True, include non-numeric columns in result with NaN values
            ignore_rowfun_columns (bool): If True, skip columns created by row functions during grouping
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and sum results
        """
        return self._aggregate_numeric(self._NumericOperation.SUM, keep_non_numeric_columns_set_to_nan, ignore_rowfun_columns, column_keys_to_aggregate)
    
    def mean(self, column_keys_to_aggregate: list[CK] | None = None, keep_non_numeric_columns_set_to_nan: bool = False, ignore_rowfun_columns: bool = True) -> "UnitedDataframe[CK]":
        """
        Calculate the mean of numeric columns for each group.
        
        Args:
            keep_non_numeric_columns_set_to_nan (bool): If True, include non-numeric columns in result with NaN values
            ignore_rowfun_columns (bool): If True, skip columns created by row functions during grouping
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and mean results
        """
        return self._aggregate_numeric(self._NumericOperation.MEAN, keep_non_numeric_columns_set_to_nan, ignore_rowfun_columns, column_keys_to_aggregate)
    
    def std(self, keep_non_numeric_columns_set_to_nan: bool = False, ignore_rowfun_columns: bool = True, column_keys_to_aggregate: list[CK] | None = None) -> "UnitedDataframe[CK]":
        """
        Calculate the standard deviation of numeric columns for each group.

        Args:
            keep_non_numeric_columns_set_to_nan (bool): If True, include non-numeric columns in result with NaN values
            ignore_rowfun_columns (bool): If True, skip columns created by row functions during grouping
            column_keys_to_aggregate (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
        """
        return self._aggregate_numeric(self._NumericOperation.STD, keep_non_numeric_columns_set_to_nan, ignore_rowfun_columns, column_keys_to_aggregate)

    def count(
            self, 
            column_keys_to_consider: Sequence[CK] | None = None,
            result_column_type: ColumnType = ColumnType.INTEGER_64,
            ignore_rowfun_columns: bool = True, 
    ) -> "UnitedDataframe[CK]":
        """
        Count non-null values for each group.
        
        Args:
            ignore_rowfun_columns (bool): If True, skip columns created by row functions during grouping
            column_keys_to_consider (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            result_column_type (ColumnType): The type of the result column(s).
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and count results
        """

        with self._dataframe._rlock: # type: ignore
            
            # Step 1: Get the column keys to consider

            _column_keys_to_consider: list[CK] = []
            for col, col_info in self._available_column_information.items():
                if column_keys_to_consider is not None:
                    if col not in column_keys_to_consider:
                        continue
                _column_keys_to_consider.append(col)

                # Skip all categorical columns
                if col in self._categorical_column_information.keys():
                    continue

            # Step 2: Count the number of non-null values for each group for the columns to consider
            
            group_data: list[dict[CK, LOWLEVEL_TYPE]] = []
            
            for container in self._grouping_containers:
                row_data: dict[CK, LOWLEVEL_TYPE] = {}
                
                # Add group key values
                for col in self._categorical_column_information.keys():
                    row_data[col] = container.categorical_key_values[list(self._categorical_column_information.keys()).index(col)]
                
                # Add count results using GroupingContainer method
                for i, col in enumerate(_column_keys_to_consider):
                    result_key: CK = _column_keys_to_consider[i]
                    col_info: ColumnInformation = self._available_column_information[col]
                    row_data[result_key] = container.dataframe[col_info.internal_dataframe_column_name].count() # type: ignore
                
                group_data.append(row_data)
            
            # Step 3: Create the result dataframe
            
            result_df = pd.DataFrame(group_data)
            
            # Collect the column information for the result dataframe of the categorical columns
            result_column_keys: list[CK] = list(self._categorical_column_information.keys())
            result_column_types: dict[CK, ColumnType] = {col: col_info.column_type for col, col_info in self._categorical_column_information.items() if col in result_column_keys}
            result_column_units: dict[CK, Unit|None] = {col: col_info.column_unit for col, col_info in self._categorical_column_information.items() if col in result_column_keys}
            
            # Add the result column information for the columns to consider
            for i, col in enumerate(_column_keys_to_consider):
                result_key: CK = col
                result_column_types[result_key] = self._available_column_information[col].column_type
                result_column_units[result_key] = self._available_column_information[col].column_unit
            
            return UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=result_df,
                column_keys=result_column_keys,
                column_types=result_column_types,
                column_units=result_column_units,
                internal_dataframe_column_name_formatter=self._dataframe.internal_dataframe_column_name_formatter,
                read_only=False,
                copy_dataframe=False,
                rename_dataframe_columns=False
            )
    
    def apply(self, func_tuple: Tuple[CK, Callable[["UnitedDataframe[CK]"], SCALAR_TYPE]]) -> "UnitedDataframe[CK]":
        """
        Apply a function to each group.
        
        Args:
            func_tuple: A tuple of (result_column_key, function) where function takes a United_Dataframe and returns a scalar
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and function results
        """
        result_column_key, func = func_tuple
        
        with self._dataframe._rlock: # type: ignore
            group_data: list[dict[CK, LOWLEVEL_TYPE]] = []
            result_column_type: ColumnType | None = None
            result_column_unit: Unit | None = None
            first_result: bool = True
            
            for container in self._grouping_containers:
                row_data: dict[CK, LOWLEVEL_TYPE] = {}
                
                # Add group key values
                for col in self._categorical_column_information.keys():
                    row_data[col] = container.categorical_key_values[list(self._categorical_column_information.keys()).index(col)]
                
                # Add function result
                group_df = container.united_dataframe()
                result: SCALAR_TYPE = func(group_df)
                
                # Determine column type and unit from first result
                if first_result:
                    result_type = type(result)
                    result_column_type = ColumnType.infer_approbiate_column_type(result_type) # type: ignore[reportUnknownReturnType]
                    
                    # Get unit if result is a UnitedScalar
                    if isinstance(result, UnitedScalar):
                        result_column_unit = result.unit # type: ignore
                    else:
                        result_column_unit = None
                    
                    first_result = False

                # Get and add the lowlevel result to the row data
                row_data[result_column_key] = result_column_type.get_value_for_dataframe(result, result_column_unit) # type: ignore
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result using column information
            result_column_keys = list(self._categorical_column_information.keys()) + [result_column_key]
            result_column_types = {col: col_info.column_type for col, col_info in self._categorical_column_information.items()}
            result_column_units = {col: col_info.column_unit for col, col_info in self._categorical_column_information.items()}
            
            # Add the inferred column type and unit for the result
            if result_column_type is not None:
                result_column_types[result_column_key] = result_column_type
                result_column_units[result_column_key] = result_column_unit
            
            return UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=result_df,
                column_keys=result_column_keys,
                column_types=result_column_types,
                column_units=result_column_units,
                internal_dataframe_column_name_formatter=self._dataframe.internal_dataframe_column_name_formatter,
                read_only=False,
                copy_dataframe=False,
                rename_dataframe_columns=False
            )
    
    def head(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Return the first n rows from each group.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe[CK]: A dataframe with the first n rows from each group
        """
        with self._dataframe._rlock: # type: ignore
            all_rows: list[pd.DataFrame] = []
            
            for container in self._grouping_containers:
                if len(container.dataframe) >= n:
                    head_rows = container.dataframe.head(n) # type: ignore
                    all_rows.append(head_rows)
                else:
                    all_rows.append(container.dataframe) # type: ignore
            
            # Combine all rows
            if all_rows:
                result_df = pd.concat(all_rows, ignore_index=True)
            else:
                result_df = pd.DataFrame()
            
            # Create United_Dataframe for result
            return UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=result_df,
                column_keys=self._dataframe.colkeys,
                column_types=self._dataframe.coltypes,
                column_units=self._dataframe.units,
                internal_dataframe_column_name_formatter=self._dataframe.internal_dataframe_column_name_formatter,
                read_only=False,
                copy_dataframe=False,
                rename_dataframe_columns=False
            )
    
    def first(self) -> "UnitedDataframe[CK]":
        """
        Return the first row from each group.
        
        Returns:
            United_Dataframe[CK]: A dataframe with the first row from each group
        """
        return self.head(1)
    
    def tail(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Return the last n rows from each group.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe[CK]: A dataframe with the last n rows from each group
        """
        with self._dataframe._rlock: # type: ignore
            all_rows: list[pd.DataFrame] = []
            
            for container in self._grouping_containers:
                if len(container.dataframe) >= n:
                    tail_rows = container.dataframe.tail(n) # type: ignore
                    all_rows.append(tail_rows)
                else:
                    all_rows.append(container.dataframe) # type: ignore
            
            # Combine all rows
            if all_rows:
                result_df = pd.concat(all_rows, ignore_index=True)
            else:
                result_df = pd.DataFrame()
            
            # Create United_Dataframe for result
            return UnitedDataframe[CK]._construct(  # type: ignore
                dataframe=result_df,
                column_keys=self._dataframe.colkeys,
                column_types=self._dataframe.coltypes,
                column_units=self._dataframe.units,
                internal_dataframe_column_name_formatter=self._dataframe.internal_dataframe_column_name_formatter,
                read_only=False,
                copy_dataframe=False,
                rename_dataframe_columns=False
            )
    
    def last(self) -> "UnitedDataframe[CK]":
        """
        Return the last row from each group.
        
        Returns:
            United_Dataframe[CK]: A dataframe with the last row from each group
        """
        return self.tail(1)
    
    def get_filtered(self, filter_dict: dict[CK, SCALAR_TYPE]) -> "BaseGrouping[CK]":
        """
        Get a filtered version of the GroupBy object.
        
        Args:
            filter_dict (dict[CK, SCALAR_TYPE]): Dictionary of column keys and values to filter by
            
        Returns:
            BaseGrouping[CK]: A new BaseGrouping object with filtered data
        """
        with self._dataframe._rlock: # type: ignore
            # Apply filter to the original dataframe
            filtered_df = self._dataframe.copy()
            
            for column_key, filter_value in filter_dict.items():
                if callable(filter_value):
                    # Apply lambda function filter
                    # This would need to be implemented in the parent dataframe
                    raise NotImplementedError("Lambda function filtering not yet implemented")
                else:
                    # Apply exact value filter
                    mask = filtered_df.mask_get_equal_to(column_key, filter_value)
                    filtered_df = filtered_df._mask_apply_to_dataframe(mask) # type: ignore
            
            return BaseGrouping(filtered_df, self._available_column_keys)
    
    def isna(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are missing/null for each group.
        
        Args:
            subset (list[CK] | None): List of column keys to check for missing values.
                                     If None, checks all columns.
            
        Returns:
            np.ndarray: Boolean array where True indicates missing values
        """
        with self._dataframe._rlock: # type: ignore
            if subset is None:
                subset = list(self._available_column_information.keys())
            
            all_masks: list[bool] = []
            
            for container in self._grouping_containers:
                group_mask: np.ndarray = np.zeros(len(container.dataframe), dtype=bool)
                
                for col in subset:
                    col_info = self._available_column_information[col]
                    col_mask = container.dataframe[col_info.internal_dataframe_column_name].isna() # type: ignore
                    group_mask = group_mask | col_mask.values # type: ignore
                
                all_masks.extend(group_mask.tolist())
            
            return np.array(all_masks)
    
    def notna(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are not missing/null for each group.
        
        Args:
            subset (list[CK] | None): List of column keys to check for non-missing values.
                                     If None, checks all columns.
            
        Returns:
            np.ndarray: Boolean array where True indicates non-missing values
        """
        return ~self.isna(subset)