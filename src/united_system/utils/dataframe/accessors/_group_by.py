

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe
    from ..column_key import ColumnKey

from ...scalars.united_scalar import UnitedScalar
from ..column_type import SCALAR_TYPE, ColumnType
from ..internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter
import pandas as pd
import numpy as np

CK = TypeVar("CK", bound="ColumnKey|str")


class GroupBy(Generic[CK]):
    """
    A GroupBy object for performing grouped operations on United_Dataframe.
    
    This class provides pandas-like groupby functionality with unit awareness.
    """
    
    def __init__(self, dataframe: "UnitedDataframe[CK]", by: list[CK]|set[CK]):
        """
        Initialize a GroupBy object.
        
        Args:
            dataframe: The United_Dataframe to group
            by: List of columns to group by
        """
        self._dataframe: "UnitedDataframe[CK]" = dataframe
        self._by: list[CK] = list(by)
        self._groups: dict[tuple, "UnitedDataframe[CK]"] | None = None
        self._group_keys: list[tuple] = []
    
    def _get_groups(self) -> dict[tuple, "UnitedDataframe[CK]"]:
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
                    col_name = self._dataframe._get_internal_dataframe_column_name(col)
                    group_columns.append(self._dataframe._internal_dataframe[col_name])
                
                # Create group keys
                group_keys = list(zip(*group_columns))
                unique_keys = list(set(group_keys))
                
                # Create groups
                self._groups = {}
                self._group_keys = unique_keys
                
                for key in unique_keys:
                    # Create boolean mask for this group
                    mask = pd.Series([gk == key for gk in group_keys], index=self._dataframe._internal_dataframe.index)
                    
                    # Get the subset of data for this group
                    subset_df = self._dataframe._internal_dataframe[mask].copy()

                    # Create United_Dataframe for this group - import here to avoid circular import
                    from ...united_dataframe import UnitedDataframe
                    group_df = UnitedDataframe[CK](
                        subset_df,
                        self._dataframe.colkeys,
                        self._dataframe.column_types,
                        self._dataframe.column_units,
                        self._dataframe._internal_dataframe_column_name_formatter
                    )
                    
                    self._groups[key] = group_df
        
        return self._groups
    
    @property
    def groups(self) -> dict[tuple, "UnitedDataframe[CK]"]:
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
        if not self._group_keys:
            self._get_groups()
        return self._group_keys
    
    def size(self, result_column_key: CK) -> "UnitedDataframe[CK]":
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
                for _, col in enumerate(self._by):
                    col_name_string = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[col_name_string] = self._dataframe._internal_dataframe[col_name_string].iloc[0]
                
                # Add size result
                row_data[result_column_key] = len(group_df)
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = list(self._by) + [result_column_key]
            result_column_types = {key: self._dataframe.column_types[key] for key in self._by}
            result_column_types[result_column_key] = ColumnType.INTEGER_64
            result_column_units = {key: self._dataframe.column_units[key] for key in self._by}
            result_column_units[result_column_key] = None

            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                result_column_keys,
                result_column_types,
                result_column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def sum(self, numeric_only: bool = True, result_column_keys: list[CK] | None = None) -> "UnitedDataframe[CK]":
        """
        Calculate the sum of numeric columns for each group.
        
        Args:
            numeric_only (bool): If True, only sum numeric columns (default: True)
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and sum results
            
        Raises:
            ValueError: If numeric_only is True and no numeric columns exist
            
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
                    raise ValueError("No numeric columns found for summation.")
            else:
                numeric_columns = self._dataframe.colkeys
            
            if result_column_keys is None:
                result_column_keys = numeric_columns.copy()
            elif len(result_column_keys) != len(numeric_columns):
                raise ValueError(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to sum ({len(numeric_columns)}).")
            
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for _, col in enumerate(self._by):
                    col_name_string = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[col_name_string] = self._dataframe._internal_dataframe[col_name_string].iloc[0]
                
                # Add sum results
                for i, col in enumerate(numeric_columns):
                    result_key = result_column_keys[i]
                    col_name = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[result_key] = group_df._internal_dataframe[col_name].sum()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = list(self._by) + result_column_keys
            result_column_types = {key: self._dataframe.column_types[key] for key in self._by}
            result_column_units = {key: self._dataframe.column_units[key] for key in self._by}
            
            for i, col in enumerate(numeric_columns):
                result_key = result_column_keys[i]
                result_column_types[result_key] = self._dataframe.column_types[col]
                result_column_units[result_key] = self._dataframe.column_units[col]
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                result_column_keys,
                result_column_types,
                result_column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def mean(self, numeric_only: bool = True, result_column_keys: list[CK] | None = None) -> "UnitedDataframe[CK]":
        """
        Calculate the mean of numeric columns for each group.
        
        Args:
            numeric_only (bool): If True, only calculate mean for numeric columns (default: True)
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and mean results
            
        Raises:
            ValueError: If numeric_only is True and no numeric columns exist
            
        Examples:
            # Calculate mean for all numeric columns
            means = grouped.mean()
            
            # Calculate mean with custom result column keys
            means = grouped.mean(result_column_keys=['mean_temp', 'mean_pressure'])
        """
        with self._dataframe._rlock:
            if numeric_only:
                numeric_columns = self._dataframe.get_numeric_column_keys()
                if not numeric_columns:
                    raise ValueError("No numeric columns found for mean calculation.")
            else:
                numeric_columns = self._dataframe.colkeys
            
            if result_column_keys is None:
                result_column_keys = numeric_columns.copy()
            elif len(result_column_keys) != len(numeric_columns):
                raise ValueError(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to calculate mean ({len(numeric_columns)}).")
            
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for _, col in enumerate(self._by):
                    col_name_string = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[col_name_string] = self._dataframe._internal_dataframe[col_name_string].iloc[0]
                
                # Add mean results
                for i, col in enumerate(numeric_columns):
                    result_key = result_column_keys[i]
                    col_name = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[result_key] = group_df._internal_dataframe[col_name].mean()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = list(self._by) + result_column_keys
            result_column_types = {key: self._dataframe.column_types[key] for key in self._by}
            result_column_units = {key: self._dataframe.column_units[key] for key in self._by}
            
            for i, col in enumerate(numeric_columns):
                result_key = result_column_keys[i]
                result_column_types[result_key] = self._dataframe.column_types[col]
                result_column_units[result_key] = self._dataframe.column_units[col]
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                result_column_keys,
                result_column_types,
                result_column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def count(self, result_column_keys: list[CK] | None = None) -> "UnitedDataframe[CK]":
        """
        Count non-null values for each group.
        
        Args:
            result_column_keys (list[CK] | None): List of column keys for the result columns.
                                                 If None, uses the original column keys.
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and count results
            
        Examples:
            # Count non-null values for all columns
            counts = grouped.count()
            
            # Count with custom result column keys
            counts = grouped.count(result_column_keys=['count_temp', 'count_pressure'])
        """
        with self._dataframe._rlock:
            columns = self._dataframe.colkeys
            
            if result_column_keys is None:
                result_column_keys = columns.copy()
            elif len(result_column_keys) != len(columns):
                raise ValueError(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to count ({len(columns)}).")
            
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for _, col in enumerate(self._by):
                    col_name_string = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[col_name_string] = self._dataframe._internal_dataframe[col_name_string].iloc[0]
                
                # Add count results
                for i, col in enumerate(columns):
                    result_key = result_column_keys[i]
                    col_name = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[result_key] = group_df._internal_dataframe[col_name].count()
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = list(self._by) + result_column_keys
            result_column_types = {key: self._dataframe.column_types[key] for key in self._by}
            result_column_units = {key: self._dataframe.column_units[key] for key in self._by}
            
            for i, col in enumerate(columns):
                result_key = result_column_keys[i]
                result_column_types[result_key] = ColumnType.INTEGER_64
                result_column_units[result_key] = None
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                result_column_keys,
                result_column_types,
                result_column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def apply(self, func: Callable[["UnitedDataframe[CK]"], UnitedScalar], result_column_key: CK) -> "UnitedDataframe[CK]":
        """
        Apply a function to each group.
        
        Args:
            func: A function that takes a United_Dataframe and returns a scalar
            result_column_key (CK): The column key to use for the result
            
        Returns:
            United_Dataframe[CK]: A dataframe with group keys and function results
            
        Examples:
            # Apply custom function to each group
            def custom_func(group_df):
                # Your custom logic here
                return group_df.column_get_values('temperature')[0]
            
            results = grouped.apply(custom_func, 'first_temp')
        """
        with self._dataframe._rlock:
            group_data = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                row_data = {}
                
                # Add group key values
                for _, col in enumerate(self._by):
                    col_name_string = self._dataframe._get_internal_dataframe_column_name(col)
                    row_data[col_name_string] = self._dataframe._internal_dataframe[col_name_string].iloc[0]
                
                # Add function result
                result = func(group_df)
                row_data[result_column_key] = result
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = list(self._by) + [result_column_key]
            result_column_types = {key: self._dataframe.column_types[key] for key in self._by}
            result_column_units = {key: self._dataframe.column_units[key] for key in self._by}
            
            # Determine column type for result based on function return type
            result_column_types[result_column_key] = ColumnType.REAL_UNITED_SCALAR
            result_column_units[result_column_key] = None
            
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                result_column_keys,
                result_column_types,
                result_column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def head(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Return the first n rows from each group.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe[CK]: A dataframe with the first n rows from each group
            
        Examples:
            # Get first row from each group
            first_rows = grouped.head(1)
            
            # Get first 3 rows from each group
            first_3_rows = grouped.head(3)
        """
        with self._dataframe._rlock:
            all_rows = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                if len(group_df) >= n:
                    head_rows = group_df._internal_dataframe.head(n)
                    all_rows.append(head_rows)
                else:
                    all_rows.append(group_df._internal_dataframe)
            
            # Combine all rows
            if all_rows:
                result_df = pd.concat(all_rows, ignore_index=True)
            else:
                result_df = pd.DataFrame()
            
            # Create United_Dataframe for result
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                self._dataframe.colkeys,
                self._dataframe.column_types,
                self._dataframe.column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def first(self) -> "UnitedDataframe[CK]":
        """
        Return the first row from each group.
        
        Returns:
            United_Dataframe[CK]: A dataframe with the first row from each group
            
        Examples:
            # Get first row from each group
            first_rows = grouped.first()
        """
        return self.head(1)
    
    def tail(self, n: int = 1) -> "UnitedDataframe[CK]":
        """
        Return the last n rows from each group.
        
        Args:
            n (int): Number of rows to return from each group (default: 1)
            
        Returns:
            United_Dataframe[CK]: A dataframe with the last n rows from each group
            
        Examples:
            # Get last row from each group
            last_rows = grouped.tail(1)
            
            # Get last 3 rows from each group
            last_3_rows = grouped.tail(3)
        """
        with self._dataframe._rlock:
            all_rows = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                if len(group_df) >= n:
                    tail_rows = group_df._internal_dataframe.tail(n)
                    all_rows.append(tail_rows)
                else:
                    all_rows.append(group_df._internal_dataframe)
            
            # Combine all rows
            if all_rows:
                result_df = pd.concat(all_rows, ignore_index=True)
            else:
                result_df = pd.DataFrame()
            
            # Create United_Dataframe for result
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                result_df,
                self._dataframe.colkeys,
                self._dataframe.column_types,
                self._dataframe.column_units,
                self._dataframe._internal_dataframe_column_name_formatter
            )
    
    def last(self) -> "UnitedDataframe[CK]":
        """
        Return the last row from each group.
        
        Returns:
            United_Dataframe[CK]: A dataframe with the last row from each group
            
        Examples:
            # Get last row from each group
            last_rows = grouped.last()
        """
        return self.tail(1)
    
    def get_filtered(self, filter_dict: dict[CK, SCALAR_TYPE]) -> "GroupBy[CK]":
        """
        Get a filtered version of the GroupBy object.
        
        Args:
            filter_dict (dict[CK, SCALAR_TYPE]): Dictionary of column keys and values to filter by
            
        Returns:
            _GroupBy[CK]: A new GroupBy object with filtered data
            
        Examples:
            # Filter groups where temperature > 20
            filtered_grouped = grouped.get_filtered({'temperature': lambda x: x > 20})
        """
        with self._dataframe._rlock:
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
                    filtered_df = filtered_df._mask_apply_to_dataframe(mask)
            
            return GroupBy(filtered_df, self._by)
    
    def isna(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are missing/null for each group.
        
        Args:
            subset (list[CK] | None): List of column keys to check for missing values.
                                     If None, checks all columns.
            
        Returns:
            np.ndarray: Boolean array where True indicates missing values
            
        Examples:
            # Check for missing values in all columns
            missing_mask = grouped.isna()
            
            # Check for missing values in specific columns
            missing_mask = grouped.isna(['temperature', 'pressure'])
        """
        with self._dataframe._rlock:
            if subset is None:
                subset = self._dataframe.colkeys
            
            all_masks = []
            
            for key in self.group_keys:
                group_df = self.groups[key]
                group_mask = np.zeros(len(group_df), dtype=bool)
                
                for col in subset:
                    col_name = self._dataframe._get_internal_dataframe_column_name(col)
                    col_mask = group_df._internal_dataframe[col_name].isna()
                    group_mask = group_mask | col_mask.values
                
                all_masks.extend(group_mask)
            
            return np.array(all_masks)
    
    def notna(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are not missing/null for each group.
        
        Args:
            subset (list[CK] | None): List of column keys to check for non-missing values.
                                     If None, checks all columns.
            
        Returns:
            np.ndarray: Boolean array where True indicates non-missing values
            
        Examples:
            # Check for non-missing values in all columns
            valid_mask = grouped.notna()
            
            # Check for non-missing values in specific columns
            valid_mask = grouped.notna(['temperature', 'pressure'])
        """
        return ~self.isna(subset)