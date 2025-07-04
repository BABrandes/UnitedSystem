from typing import Generic, Callable, TypeVar
from ..united_dataframe.united_dataframe import United_Dataframe, Column_Key
from ..scalars.united_scalar import UnitedScalar
from ..units.unit_quantity import UnitQuantity
from ..units.unit import Unit
from ..united_dataframe.united_dataframe import Value_Type
from ..united_dataframe.united_dataframe import _INTERNAL_INIT_TOKEN
from ..united_dataframe._row_accessor import _RowAccessor
import pandas as pd




CK = TypeVar("CK", bound=Column_Key|str)


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
                        self._dataframe._unit_quantities,
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
            result_si_quantities: dict[CK, UnitQuantity] = {column_key: UnitQuantity.NO_NUMBER for column_key in result_column_keys}  # counts have no units
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
                numeric_columns = self._dataframe.column_keys
            
            if result_column_keys is None:
                result_column_keys = numeric_columns.copy()
            elif len(result_column_keys) != len(numeric_columns):
                raise ValueError(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to sum ({len(numeric_columns)}).")
            
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
            result_si_quantities: dict[CK, UnitQuantity] = self._dataframe.unit_quantities.copy()
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
            ValueError: If numeric_only is True and no numeric columns exist
            
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
                    raise ValueError("No numeric columns found for mean calculation.")
            else:
                numeric_columns = self._dataframe.column_keys
            
            if result_column_keys is None:
                result_column_keys = numeric_columns.copy()
            elif len(result_column_keys) != len(numeric_columns):
                raise ValueError(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to calculate mean ({len(numeric_columns)}).")
            
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
            result_si_quantities: dict[CK, UnitQuantity] = self._dataframe.unit_quantities.copy()
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
                raise ValueError(f"Number of result column keys ({len(result_column_keys)}) must match number of columns to count ({len(columns_to_count)}).")
            
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
            result_si_quantities: dict[CK, UnitQuantity] = {column_key: UnitQuantity.NO_NUMBER for column_key in result_column_keys_final}  # counts have no units
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
    
    def apply(self, func: Callable[[United_Dataframe[CK]], UnitedScalar], result_column_key: CK) -> United_Dataframe[CK]:
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
                raise ValueError(f"Column {result_column_key} does not exist in the dataframe.")
            
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
                if not isinstance(result, UnitedScalar):
                    raise ValueError(f"Function must return a UnitedScalar, got {type(result)}")
                
                # Add result to row data
                col_name = self._dataframe.internal_dataframe_column_string(result_column_key)
                row_data[col_name] = result.canonical_value
                
                group_data.append(row_data)
            
            # Create result dataframe
            result_df = pd.DataFrame(group_data)
            
            # Create United_Dataframe for result
            result_column_keys = self._dataframe.column_keys.copy()
            result_si_quantities = self._dataframe.unit_quantities.copy()
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
            ValueError: If n is negative
            
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
                raise ValueError(f"n must be non-negative, got {n}")
            
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
                return United_Dataframe[CK].create_empty(
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
            ValueError: If any group is empty
            
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
                    raise ValueError(f"Cannot get first row from empty group {key}")
                
                first_row = group_df.rowfun_first()
                all_first_rows.append(first_row)
            
            # Concatenate all the first rows from all groups
            if all_first_rows:
                return United_Dataframe[CK].concatenate_dataframes(all_first_rows[0], *all_first_rows[1:])
            else:
                # Return empty dataframe with same structure
                return United_Dataframe[CK].create_empty(
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
            ValueError: If n is negative
            
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
                raise ValueError(f"n must be non-negative, got {n}")
            
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
                return United_Dataframe[CK].create_empty(
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
            ValueError: If any group is empty
            
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
                    raise ValueError(f"Cannot get last row from empty group {key}")
                
                last_row = group_df.rowfun_last()
                all_last_rows.append(last_row)
            
            # Concatenate all the last rows from all groups
            if all_last_rows:
                return United_Dataframe[CK].concatenate_dataframes(all_last_rows[0], *all_last_rows[1:])
            else:
                # Return empty dataframe with same structure
                return United_Dataframe[CK].create_empty(
                    self._dataframe._column_keys,
                    self._dataframe._display_units,
                    self._dataframe._value_types
                )

    def get_filtered(self, filter_dict: dict[CK, UnitedScalar|str|bool|datetime]) -> "GroupBy[CK]":
        """
        Filter each group by a dictionary of column keys and values.
        
        This method applies the same filtering criteria to each group individually,
        then returns a new GroupBy object with the filtered groups.
        
        Args:
            filter_dict (dict[CK, UnitedScalar|str|bool|datetime]): A dictionary of column keys and values to filter by
            
        Returns:
            GroupBy[CK]: A new GroupBy object with filtered groups
            
        Raises:
            ValueError: If any column key does not exist in the dataframe
            
        Examples:
            # Filter groups to only include rows where 'status' is 'active'
            filtered_groups = grouped.get_filtered({'status': 'active'})
            
            # Filter groups with multiple criteria
            filtered_groups = grouped.get_filtered({
                'status': 'active',
                'priority': UnitedScalar.united_number(5, Unit.one)
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
                empty_dataframe = United_Dataframe[CK].create_empty(
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