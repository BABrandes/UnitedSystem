"""
Filter operations mixin for UnitedDataframe.

Contains all operations related to advanced filtering operations,
including complex filters, multi-column filters, and filter combinations.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Callable, TYPE_CHECKING, overload, Mapping
from .dataframe_protocol import UnitedDataframeProtocol, CK, SCALAR_TYPE, VALUE_TYPE
from ..._arrays.bool_array import BoolArray

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class FilterMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Filter operations mixin for UnitedDataframe.
    
    Provides all functionality related to advanced filtering operations,
    including complex filters, multi-column filters, and filter combinations.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Filter Operations: Single Column ------------

    def filter_column_equals(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column equals a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_equal_to(column_key, item)
            return self._mask_apply_to_dataframe(mask)

    def filter_column_not_equals(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column does not equal a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_not_equal_to(column_key, item)
            return self._mask_apply_to_dataframe(mask)

    def filter_column_greater_than(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is greater than a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_greater_than(column_key, item)
            return self._mask_apply_to_dataframe(mask)
        
    def filter_column_greater_equal(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is greater than or equal to a specific value.

        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_greater_equal(column_key, item)
            return self._mask_apply_to_dataframe(mask)

    def filter_column_less_than(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is less than a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_less_than(column_key, item)
            return self._mask_apply_to_dataframe(mask)
        
    def filter_column_less_equal(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is less than or equal to a specific value.

        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to filter by
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask = self.mask_get_less_equal(column_key, item)
            return self._mask_apply_to_dataframe(mask)
        
    def filter_column_in_range(self, column_key: CK, min_value: SCALAR_TYPE|VALUE_TYPE, max_value: SCALAR_TYPE|VALUE_TYPE) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is within a specific range.
        
        Args:
            column_key (CK): The column key
            min_value (SCALAR_TYPE|VALUE_TYPE): The minimum value (inclusive)
            max_value (SCALAR_TYPE|VALUE_TYPE): The maximum value (inclusive)
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_in_range(column_key, min_value, max_value)
            return self._mask_apply_to_dataframe(mask)
    
    @overload
    def filter_columns_in_range(self, range_dict: Mapping[CK, tuple[VALUE_TYPE, VALUE_TYPE]]) -> "UnitedDataframe[CK]":
        ...
    @overload
    def filter_columns_in_range(self, range_dict: Mapping[CK, tuple[SCALAR_TYPE, SCALAR_TYPE]]) -> "UnitedDataframe[CK]":
        ...
    def filter_columns_in_range(self, range_dict:
                                Mapping[CK, tuple[SCALAR_TYPE|VALUE_TYPE, SCALAR_TYPE|VALUE_TYPE]]|
                                Mapping[CK, tuple[VALUE_TYPE, VALUE_TYPE]]|
                                Mapping[CK, tuple[SCALAR_TYPE, SCALAR_TYPE]]) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where multiple columns are within a specific range.

        Args:
            range_dict (Mapping[CK, tuple[SCALAR_TYPE|VALUE_TYPE, SCALAR_TYPE|VALUE_TYPE]]): A dictionary of column keys and their range.
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            masks: list[BoolArray] = []
            for column_key, (min_value, max_value) in range_dict.items():
                masks.append(self.mask_get_in_range(column_key, min_value, max_value))
            return self.filter_and(*masks)
        
    def filter_column_get_complete_rows(self, *column_keys: CK) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is missing.
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_complete_rows(*column_keys)
            return self._mask_apply_to_dataframe(mask)
        
    def filter_column_get_incomplete_rows(self, *column_keys: CK) -> "UnitedDataframe[CK]":
        """
        Filter dataframe where column is not missing.
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_incomplete_rows(*column_keys)
            return self._mask_apply_to_dataframe(mask)


    # ----------- Filter Operations: Multiple Conditions ------------

    def filter_and(self, *masks: BoolArray) -> "UnitedDataframe[CK]":
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
            combined_mask: BoolArray = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask & mask
            
            return self._mask_apply_to_dataframe(combined_mask)

    def filter_or(self, *masks: BoolArray) -> "UnitedDataframe[CK]":
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
            combined_mask: BoolArray = masks[0]
            for mask in masks[1:]:
                combined_mask = combined_mask | mask
            
            return self._mask_apply_to_dataframe(combined_mask)

    def filter_not(self, mask: BoolArray) -> "UnitedDataframe[CK]":
        """
        Filter dataframe using NOT logic on a boolean mask.
        
        Args:
            mask (BoolArray): Boolean mask to negate
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            inverted_mask = ~mask
            return self._mask_apply_to_dataframe(inverted_mask)

    # ----------- Filter Operations: Custom Functions ------------

    def filter_by_function(self, filter_func: Callable[[Mapping[CK, SCALAR_TYPE]], bool], column_keys: list[CK]|None = None) -> "UnitedDataframe[CK]":
        """
        Filter dataframe using a custom function on a column.
        
        Args:
            column_key (CK): The column key
            filter_func (Callable[[SCALAR_TYPE], bool]): Function that takes a value and returns True/False
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_by_function(filter_func, column_keys)
            return self._mask_apply_to_dataframe(mask)

    # ----------- Filter Operations: Missing Values ------------

    def filter_valid_values(self, column_key: CK) -> "UnitedDataframe[CK]":
        """
        Filter dataframe to keep only rows with valid (non-missing) values in the specified column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_complete_rows()
            return self._mask_apply_to_dataframe(mask)

    def filter_missing_values(self, column_key: CK) -> "UnitedDataframe[CK]":
        """
        Filter dataframe to keep only rows with missing values in the specified column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            UnitedDataframe: Filtered dataframe
        """
        with self._rlock:
            mask: BoolArray = self.mask_get_incomplete_rows()
            return self._mask_apply_to_dataframe(mask) 