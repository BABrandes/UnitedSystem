"""
Mask operations mixin for UnitedDataframe.

Contains all operations related to boolean masking operations,
including mask creation, application, and filtering.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Union, Any, List
import numpy as np
from .dataframe_protocol import UnitedDataframeMixin, CK
from ...bool_array import BoolArray

class MaskOperationsMixin(UnitedDataframeMixin[CK]):
    """
    Mask operations mixin for UnitedDataframe.
    
    Provides all functionality related to boolean masking operations,
    including mask creation, application, and filtering.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Mask Operations: Creation ------------

    def mask_get_valid_values(self, column_key: CK) -> BoolArray:
        """
        Get a boolean mask for valid (non-null) values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BoolArray: Boolean mask where True indicates valid values
        """
        with self._rlock:  # Full IDE support!
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_valid_mask = self._internal_canonical_dataframe[internal_column_name].notna()
            return BoolArray(is_valid_mask.tolist())

    def mask_get_missing_values(self, column_key: CK) -> BoolArray:
        """
        Get a boolean mask for missing (null) values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BoolArray: Boolean mask where True indicates missing values
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_missing_mask = self._internal_canonical_dataframe[internal_column_name].isna()
            return BoolArray(is_missing_mask.tolist())

    def mask_get_equal_to(self, column_key: CK, value: Any) -> BoolArray:
        """
        Get a boolean mask for values equal to a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates equal values
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_equal_mask = self._internal_canonical_dataframe[internal_column_name] == value
            return BoolArray(is_equal_mask.tolist())

    def mask_get_not_equal_to(self, column_key: CK, value: Any) -> BoolArray:
        """
        Get a boolean mask for values not equal to a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates not equal values
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_not_equal_mask = self._internal_canonical_dataframe[internal_column_name] != value
            return BoolArray(is_not_equal_mask.tolist())

    def mask_get_greater_than(self, column_key: CK, value: Any) -> BoolArray:
        """
        Get a boolean mask for values greater than a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates greater values
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_greater_mask = self._internal_canonical_dataframe[internal_column_name] > value
            return BoolArray(is_greater_mask.tolist())

    def mask_get_less_than(self, column_key: CK, value: Any) -> BoolArray:
        """
        Get a boolean mask for values less than a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates lesser values
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_less_mask = self._internal_canonical_dataframe[internal_column_name] < value
            return BoolArray(is_less_mask.tolist())

    def mask_get_in_range(self, column_key: CK, min_value: Any, max_value: Any) -> BoolArray:
        """
        Get a boolean mask for values within a specific range (inclusive).
        
        Args:
            column_key (CK): The column key
            min_value (Any): The minimum value (inclusive)
            max_value (Any): The maximum value (inclusive)
            
        Returns:
            BoolArray: Boolean mask where True indicates values in range
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            column_data = self._internal_canonical_dataframe[internal_column_name]
            is_in_range_mask = (column_data >= min_value) & (column_data <= max_value)
            return BoolArray(is_in_range_mask.tolist())

    # ----------- Mask Operations: Application ------------

    def mask_apply_to_dataframe(self, mask: BoolArray) -> "UnitedDataframe":
        """
        Apply a boolean mask to the dataframe, returning a new filtered dataframe.
        
        Args:
            mask (BoolArray): Boolean mask to apply
            
        Returns:
            UnitedDataframe: New filtered dataframe
        """
        with self._rlock:
            if len(mask) != len(self._internal_canonical_dataframe):
                raise ValueError(f"Mask length ({len(mask)}) does not match dataframe length ({len(self._internal_canonical_dataframe)}).")
            
            # Filter internal dataframe
            filtered_dataframe = self._internal_canonical_dataframe[mask.to_pandas()]
            
            # Create new UnitedDataframe with filtered data
            from ...united_dataframe import UnitedDataframe
            new_df = UnitedDataframe[CK](
                internal_canonical_dataframe=filtered_dataframe.copy(),
                column_keys=self._column_keys.copy(),
                column_types=self._column_types.copy(),
                dimensions=self._dimensions.copy(),
                display_units=self._display_units.copy(),
                internal_dataframe_column_strings=self._internal_dataframe_column_strings.copy(),
                internal_dataframe_name_formatter=self._internal_dataframe_name_formatter,
                read_only=True  # Filtered dataframes are read-only
            )
            return new_df

    def mask_get_row_indices(self, mask: BoolArray) -> List[int]:
        """
        Get the row indices where a boolean mask is True.
        
        Args:
            mask (BoolArray): Boolean mask
            
        Returns:
            List[int]: List of row indices where mask is True
        """
        with self._rlock:
            if len(mask) != len(self._internal_canonical_dataframe):
                raise ValueError(f"Mask length ({len(mask)}) does not match dataframe length ({len(self._internal_canonical_dataframe)}).")
            
            return [i for i, value in enumerate(mask) if value]

    # ----------- Mask operations ------------

    def mask_get_duplicates(self, subset: list[CK] | None = None, keep: str = "first") -> np.ndarray:
        """
        Return a boolean mask indicating which rows are duplicates.
        
        Args:
            subset (list[CK] | None): List of column keys to check for duplicates.
                                    If None, checks all columns.
            keep (str): Which duplicates to mark as True:
                       - "first": Mark duplicates as True, except for the first occurrence
                       - "last": Mark duplicates as True, except for the last occurrence  
                       - "all": Mark all duplicates as True
        
        Returns:
            np.ndarray: Boolean array where True indicates duplicate rows
        """
        with self._rlock:
            if subset is None:
                subset = self._column_keys
            
            # Validate subset
            for column_key in subset:
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Get internal column names
            internal_columns = [self._internal_dataframe_column_strings[col] for col in subset]
            
            # Create mask for duplicates
            if keep == "all":
                # Mark all duplicates as True (including first occurrences)
                mask = self._internal_canonical_dataframe[internal_columns].duplicated(keep=False)
            else:
                # Mark duplicates as True, except for first/last occurrence
                mask = self._internal_canonical_dataframe[internal_columns].duplicated(keep=keep)
            
            return mask.values

    def mask_remove_duplicates(self, subset: list[CK] | None = None, keep: str = "first") -> "UnitedDataframe[CK]":
        """
        Remove duplicate rows and return a new dataframe.
        
        Args:
            subset (list[CK] | None): List of column keys to check for duplicates.
                                    If None, checks all columns.
            keep (str): Which duplicates to keep:
                       - "first": Keep the first occurrence of each duplicate
                       - "last": Keep the last occurrence of each duplicate
                       - False: Remove all duplicates
        
        Returns:
            UnitedDataframe[CK]: New dataframe with duplicates removed
        """
        with self._rlock:
            if subset is None:
                subset = self._column_keys
            
            # Validate subset
            for column_key in subset:
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Get internal column names
            internal_columns = [self._internal_dataframe_column_strings[col] for col in subset]
            
            # Remove duplicates
            deduplicated_df = self._internal_canonical_dataframe.drop_duplicates(subset=internal_columns, keep=keep)
            
            # Create new dataframe with same column information
            return UnitedDataframe(
                deduplicated_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter
            )

    def mask_get_complete_cases(self) -> np.ndarray:
        """
        Return a boolean mask indicating which rows have no missing values.
        
        Returns:
            np.ndarray: Boolean array where True indicates rows with no missing values
        """
        with self._rlock:
            # A row is complete if it has no missing values in any column
            return self._internal_canonical_dataframe.notna().all(axis=1).values

    def mask_get_incomplete_cases(self) -> np.ndarray:
        """
        Return a boolean mask indicating which rows have at least one missing value.
        
        Returns:
            np.ndarray: Boolean array where True indicates rows with at least one missing value
        """
        with self._rlock:
            return ~self.mask_get_complete_cases()

    def mask_count_missing_per_row(self) -> np.ndarray:
        """
        Count the number of missing values in each row.
        
        Returns:
            np.ndarray: Array of integers indicating the number of missing values per row
        """
        with self._rlock:
            return self._internal_canonical_dataframe.isna().sum(axis=1).values

    def mask_count_missing_per_column(self) -> dict[CK, int]:
        """
        Count the number of missing values in each column.
        
        Returns:
            dict[CK, int]: Dictionary mapping column keys to number of missing values
        """
        with self._rlock:
            missing_counts = {}
            for column_key in self._column_keys:
                internal_column_string = self._internal_dataframe_column_strings[column_key]
                missing_counts[column_key] = self._internal_canonical_dataframe[internal_column_string].isna().sum()
            return missing_counts 