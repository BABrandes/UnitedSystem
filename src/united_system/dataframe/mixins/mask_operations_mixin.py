"""
Mask operations mixin for UnitedDataframe.

Contains all operations related to boolean masking, including creating masks
for missing values, valid values, and applying masks for filtering.
"""

from typing import Generic, TypeVar
import numpy as np
import pandas as pd

CK = TypeVar("CK", bound=str, default=str)

class MaskOperationsMixin(Generic[CK]):
    """
    Mask operations mixin for UnitedDataframe.
    
    Provides all functionality related to boolean masking operations,
    including creating masks for missing values, valid values, and applying masks.
    """

    # ----------- Mask operations ------------

    def mask_get_missing_values(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are NA/NaN.
        
        Args:
            subset (list[CK] | None): List of column keys to check for missing values.
                                    If None, checks all columns.
        
        Returns:
            np.ndarray: Boolean array with same shape as original, where True indicates missing values
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
            
            # Create mask - True where any of the specified columns is NA
            mask = self._internal_canonical_dataframe[internal_columns].isna()
            
            if len(internal_columns) == 1:
                return mask.values.flatten()
            else:
                return mask.any(axis=1).values

    def mask_get_valid_values(self, subset: list[CK] | None = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are not NA/NaN.
        
        This is the inverse of mask_get_missing_values() - returns True for non-NA values.
        
        Args:
            subset (list[CK] | None): List of column keys to check for non-NA values.
                                    If None, checks all columns.
        
        Returns:
            np.ndarray: Boolean array with same shape as original, where True indicates non-NA values
        """
        with self._rlock:
            return ~self.mask_get_missing_values(subset)

    def mask_apply_boolean_filter(self, mask: np.ndarray) -> "UnitedDataframe[CK]":
        """
        Apply a boolean mask to filter the dataframe.
        
        Args:
            mask (np.ndarray): Boolean array with same length as number of rows
            
        Returns:
            UnitedDataframe[CK]: New dataframe with only rows where mask is True
        """
        with self._rlock:
            if len(mask) != len(self._internal_canonical_dataframe):
                raise ValueError(f"Mask length {len(mask)} does not match dataframe length {len(self._internal_canonical_dataframe)}")
            
            # Apply mask to internal dataframe
            filtered_df = self._internal_canonical_dataframe[mask]
            
            # Create new dataframe with same column information
            return UnitedDataframe(
                filtered_df,
                self._column_information,
                self._internal_dataframe_column_name_formatter
            )

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