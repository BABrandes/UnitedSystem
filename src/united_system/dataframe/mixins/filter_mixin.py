"""
Filter operations mixin for UnitedDataframe.

Contains all operations related to advanced filtering operations,
including complex filters, multi-column filters, and filter combinations.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Any, List, Callable, Union
from .dataframe_protocol import UnitedDataframeMixin, CK
from ...bool_array import BoolArray

class FilterMixin(UnitedDataframeMixin[CK]):
    """
    Filter operations mixin for UnitedDataframe.
    
    Provides all functionality related to advanced filtering operations,
    including complex filters, multi-column filters, and filter combinations.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Filter Operations: Single Column ------------

    def filter_column_equals(self, column_key: CK, value: Any) -> "UnitedDataframe":
        """
        Filter dataframe where column equals a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:  # Full IDE support!
            mask = self.mask_get_equal_to(column_key, value)
            return self.mask_apply_to_dataframe(mask)

    def filter_column_not_equals(self, column_key: CK, value: Any) -> "UnitedDataframe":
        """
        Filter dataframe where column does not equal a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_not_equal_to(column_key, value)
            return self.mask_apply_to_dataframe(mask)

    def filter_column_greater_than(self, column_key: CK, value: Any) -> "UnitedDataframe":
        """
        Filter dataframe where column is greater than a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_greater_than(column_key, value)
            return self.mask_apply_to_dataframe(mask)

    def filter_column_less_than(self, column_key: CK, value: Any) -> "UnitedDataframe":
        """
        Filter dataframe where column is less than a specific value.
        
        Args:
            column_key (CK): The column key
            value (Any): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_less_than(column_key, value)
            return self.mask_apply_to_dataframe(mask)

    def filter_column_in_range(self, column_key: CK, min_value: Any, max_value: Any) -> "UnitedDataframe":
        """
        Filter dataframe where column is within a specific range.
        
        Args:
            column_key (CK): The column key
            min_value (Any): The minimum value (inclusive)
            max_value (Any): The maximum value (inclusive)
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_in_range(column_key, min_value, max_value)
            return self.mask_apply_to_dataframe(mask)

    def filter_column_in_values(self, column_key: CK, values: List[Any]) -> "UnitedDataframe":
        """
        Filter dataframe where column value is in a list of values.
        
        Args:
            column_key (CK): The column key
            values (List[Any]): List of values to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_in_values_mask = self._internal_canonical_dataframe[internal_column_name].isin(values)
            mask = BoolArray(is_in_values_mask.tolist())
            return self.mask_apply_to_dataframe(mask)

    def filter_column_not_in_values(self, column_key: CK, values: List[Any]) -> "UnitedDataframe":
        """
        Filter dataframe where column value is not in a list of values.
        
        Args:
            column_key (CK): The column key
            values (List[Any]): List of values to exclude
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            is_not_in_values_mask = ~self._internal_canonical_dataframe[internal_column_name].isin(values)
            mask = BoolArray(is_not_in_values_mask.tolist())
            return self.mask_apply_to_dataframe(mask)

    # ----------- Filter Operations: Multiple Conditions ------------

    def filter_and(self, *masks: BoolArray) -> "UnitedDataframe":
        """
        Filter dataframe using AND logic on multiple boolean masks.
        
        Args:
            *masks (BoolArray): Boolean masks to combine with AND
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            if not masks:
                raise ValueError("At least one mask must be provided.")
            
            # Combine all masks with AND logic
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
            
            return self.mask_apply_to_dataframe(combined_mask)

    def filter_or(self, *masks: BoolArray) -> "UnitedDataframe":
        """
        Filter dataframe using OR logic on multiple boolean masks.
        
        Args:
            *masks (BoolArray): Boolean masks to combine with OR
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            if not masks:
                raise ValueError("At least one mask must be provided.")
            
            # Combine all masks with OR logic
            combined_mask = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
            
            return self.mask_apply_to_dataframe(combined_mask)

    def filter_not(self, mask: BoolArray) -> "UnitedDataframe":
        """
        Filter dataframe using NOT logic on a boolean mask.
        
        Args:
            mask (BoolArray): Boolean mask to negate
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            inverted_mask = ~mask
            return self.mask_apply_to_dataframe(inverted_mask)

    # ----------- Filter Operations: Custom Functions ------------

    def filter_by_function(self, column_key: CK, filter_func: Callable[[Any], bool]) -> "UnitedDataframe":
        """
        Filter dataframe using a custom function on a column.
        
        Args:
            column_key (CK): The column key
            filter_func (Callable[[Any], bool]): Function that takes a value and returns True/False
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            internal_column_name = self._internal_dataframe_column_strings[column_key]
            column_data = self._internal_canonical_dataframe[internal_column_name]
            
            # Apply function to each value
            mask_values = [filter_func(value) for value in column_data]
            mask = BoolArray(mask_values)
            return self.mask_apply_to_dataframe(mask)

    def filter_by_row_function(self, filter_func: Callable[[dict], bool]) -> "UnitedDataframe":
        """
        Filter dataframe using a custom function on entire rows.
        
        Args:
            filter_func (Callable[[dict], bool]): Function that takes a row dict and returns True/False
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask_values = []
            
            for row_index in range(len(self._internal_canonical_dataframe)):
                row_dict = {}
                for column_key in self._column_keys:
                    internal_column_name = self._internal_dataframe_column_strings[column_key]
                    row_dict[column_key] = self._internal_canonical_dataframe.loc[row_index, internal_column_name]
                
                mask_values.append(filter_func(row_dict))
            
            mask = BoolArray(mask_values)
            return self.mask_apply_to_dataframe(mask)

    # ----------- Filter Operations: Null/Valid Values ------------

    def filter_valid_values(self, column_key: CK) -> "UnitedDataframe":
        """
        Filter dataframe to only include rows with valid (non-null) values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_valid_values(column_key)
            return self.mask_apply_to_dataframe(mask)

    def filter_missing_values(self, column_key: CK) -> "UnitedDataframe":
        """
        Filter dataframe to only include rows with missing (null) values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_missing_values(column_key)
            return self.mask_apply_to_dataframe(mask) 