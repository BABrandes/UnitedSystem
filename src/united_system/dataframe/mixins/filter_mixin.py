"""
Filter operations mixin for UnitedDataframe.

Contains all operations related to filtering data, including value-based filtering,
condition-based filtering, and other filtering operations.
"""

from typing import Generic, TypeVar, Callable, Any
import numpy as np

from ..column_type import SCALAR_TYPE

CK = TypeVar("CK", bound=str, default=str)

class FilterMixin(Generic[CK]):
    """
    Filter operations mixin for UnitedDataframe.
    
    Provides all functionality related to filtering operations,
    including value-based filtering, condition-based filtering, and other filtering operations.
    """

    # ----------- Filter operations ------------

    def filter_by_value(self, column_key: CK, value: SCALAR_TYPE, operator: str = "==") -> "UnitedDataframe[CK]":
        """
        Filter rows by a specific value in a column.
        
        Args:
            column_key (CK): The column key to filter by
            value (SCALAR_TYPE): The value to filter for
            operator (str): The comparison operator ("==", "!=", ">", "<", ">=", "<=")
            
        Returns:
            UnitedDataframe[CK]: New dataframe with filtered rows
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Get the column values
            column_values = self.column(column_key)
            
            # Apply the filter based on operator
            match operator:
                case "==":
                    mask = column_values == value
                case "!=":
                    mask = column_values != value
                case ">":
                    mask = column_values > value
                case "<":
                    mask = column_values < value
                case ">=":
                    mask = column_values >= value
                case "<=":
                    mask = column_values <= value
                case _:
                    raise ValueError(f"Unsupported operator: {operator}")
            
            return self.mask_apply_boolean_filter(mask)

    def filter_by_range(self, column_key: CK, min_value: SCALAR_TYPE, max_value: SCALAR_TYPE, inclusive: str = "both") -> "UnitedDataframe[CK]":
        """
        Filter rows by a range of values in a column.
        
        Args:
            column_key (CK): The column key to filter by
            min_value (SCALAR_TYPE): The minimum value (inclusive)
            max_value (SCALAR_TYPE): The maximum value (inclusive)
            inclusive (str): Which bounds to include ("both", "neither", "left", "right")
            
        Returns:
            UnitedDataframe[CK]: New dataframe with filtered rows
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Get the column values
            column_values = self.column(column_key)
            
            # Apply the filter based on inclusive parameter
            match inclusive:
                case "both":
                    mask = (column_values >= min_value) & (column_values <= max_value)
                case "neither":
                    mask = (column_values > min_value) & (column_values < max_value)
                case "left":
                    mask = (column_values >= min_value) & (column_values < max_value)
                case "right":
                    mask = (column_values > min_value) & (column_values <= max_value)
                case _:
                    raise ValueError(f"Invalid inclusive option: {inclusive}")
            
            return self.mask_apply_boolean_filter(mask)

    def filter_by_values(self, column_key: CK, values: list[SCALAR_TYPE], include: bool = True) -> "UnitedDataframe[CK]":
        """
        Filter rows by a list of values in a column.
        
        Args:
            column_key (CK): The column key to filter by
            values (list[SCALAR_TYPE]): The list of values to filter for/against
            include (bool): If True, include rows with values in the list. If False, exclude them.
            
        Returns:
            UnitedDataframe[CK]: New dataframe with filtered rows
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Get the column values
            column_values = self.column(column_key)
            
            # Create mask based on whether values are in the list
            mask = column_values.isin(values)
            
            if not include:
                mask = ~mask
            
            return self.mask_apply_boolean_filter(mask)

    def filter_by_condition(self, condition_func: Callable[[Any], bool]) -> "UnitedDataframe[CK]":
        """
        Filter rows by a custom condition function.
        
        Args:
            condition_func: Function that takes a row and returns boolean
            
        Returns:
            UnitedDataframe[CK]: New dataframe with filtered rows
        """
        with self._rlock:
            mask = []
            for idx, row in self._internal_canonical_dataframe.iterrows():
                try:
                    mask.append(condition_func(row))
                except Exception:
                    mask.append(False)
            
            return self.mask_apply_boolean_filter(np.array(mask))

    def filter_by_missing_values(self, column_keys: list[CK] | None = None, how: str = "any") -> "UnitedDataframe[CK]":
        """
        Filter rows based on missing values.
        
        Args:
            column_keys (list[CK] | None): List of column keys to check. If None, checks all columns.
            how (str): How to apply the filter:
                      - "any": Keep rows with any missing values in specified columns
                      - "all": Keep rows with all missing values in specified columns
                      - "none": Keep rows with no missing values in specified columns
            
        Returns:
            UnitedDataframe[CK]: New dataframe with filtered rows
        """
        with self._rlock:
            if column_keys is None:
                column_keys = self._column_keys
            
            # Validate column keys
            for column_key in column_keys:
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Get internal column names
            internal_columns = [self._internal_dataframe_column_strings[col] for col in column_keys]
            
            # Create mask based on how parameter
            match how:
                case "any":
                    # Keep rows with any missing values
                    mask = self._internal_canonical_dataframe[internal_columns].isna().any(axis=1)
                case "all":
                    # Keep rows with all missing values
                    mask = self._internal_canonical_dataframe[internal_columns].isna().all(axis=1)
                case "none":
                    # Keep rows with no missing values
                    mask = self._internal_canonical_dataframe[internal_columns].notna().all(axis=1)
                case _:
                    raise ValueError(f"Invalid how parameter: {how}")
            
            return self.mask_apply_boolean_filter(mask.values)

    def filter_by_duplicates(self, column_keys: list[CK] | None = None, keep: str = "first") -> "UnitedDataframe[CK]":
        """
        Filter out duplicate rows.
        
        Args:
            column_keys (list[CK] | None): List of column keys to check for duplicates.
                                          If None, checks all columns.
            keep (str): Which duplicates to keep:
                       - "first": Keep the first occurrence of each duplicate
                       - "last": Keep the last occurrence of each duplicate
                       - False: Remove all duplicates
            
        Returns:
            UnitedDataframe[CK]: New dataframe with duplicates removed
        """
        with self._rlock:
            return self.mask_remove_duplicates(column_keys, keep)

    def filter_by_top_n(self, column_key: CK, n: int, largest: bool = True) -> "UnitedDataframe[CK]":
        """
        Filter to get the top n rows by a column's values.
        
        Args:
            column_key (CK): The column key to sort by
            n (int): Number of top rows to return
            largest (bool): If True, return largest values. If False, return smallest values.
            
        Returns:
            UnitedDataframe[CK]: New dataframe with top n rows
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if n <= 0:
                raise ValueError("Number of rows must be positive")
            
            internal_column_string = self._internal_dataframe_column_strings[column_key]
            
            if largest:
                top_n_df = self._internal_canonical_dataframe.nlargest(n, internal_column_string)
            else:
                top_n_df = self._internal_canonical_dataframe.nsmallest(n, internal_column_string)
            
            return UnitedDataframe(
                top_n_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter
            )

    def filter_by_percentile(self, column_key: CK, percentile: float, above: bool = True) -> "UnitedDataframe[CK]":
        """
        Filter rows by percentile threshold in a column.
        
        Args:
            column_key (CK): The column key to filter by
            percentile (float): Percentile threshold (0.0 to 1.0)
            above (bool): If True, keep rows above percentile. If False, keep rows below.
            
        Returns:
            UnitedDataframe[CK]: New dataframe with filtered rows
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            if not 0 <= percentile <= 1:
                raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")
            
            # Get the column values
            column_values = self.column(column_key)
            
            # Calculate percentile threshold
            threshold = column_values.quantile(percentile)
            
            # Create mask
            if above:
                mask = column_values > threshold
            else:
                mask = column_values < threshold
            
            return self.mask_apply_boolean_filter(mask) 