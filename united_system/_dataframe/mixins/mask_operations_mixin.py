"""
Mask operations mixin for UnitedDataframe.

Contains all operations related to boolean masking operations,
including mask creation, application, and filtering.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Callable, TYPE_CHECKING, Literal, Mapping, overload, Optional
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..._scalars.united_scalar import UnitedScalar
from ..._units_and_dimension.unit import Unit
from ..._utils.scalar_type import SCALAR_TYPE
from ..._utils.value_type import VALUE_TYPE
from ..._arrays.bool_array import BoolArray
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    from ..._dataframe.united_dataframe import UnitedDataframe

class MaskOperationsMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Mask operations mixin for UnitedDataframe.
    
    Provides all functionality related to boolean masking operations,
    including mask creation, application, and filtering.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Mask Operations: Creation ------------

    def _mask_get_valid_values(self, column_key: CK) -> BoolArray:
        """
        Internal: Get a boolean mask for valid (non-null) values in a column. (No locks)
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BoolArray: Boolean mask where True indicates valid values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        is_valid_mask: pd.Series = self._internal_dataframe[internal_column_name].notna() # type: ignore
        return BoolArray(is_valid_mask) # type: ignore

    def mask_get_valid_values(self, column_key: CK) -> BoolArray:
        """
        Get a boolean mask for valid (non-null) values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BoolArray: Boolean mask where True indicates valid values
        """
        with self._rlock:
            return self._mask_get_valid_values(column_key)

    def _mask_get_missing_values(self, column_key: CK) -> BoolArray:
        """
        Internal: Get a boolean mask for missing (null) values in a column. (No locks)
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BoolArray: Boolean mask where True indicates missing values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        is_missing_mask: pd.Series = self._internal_dataframe[internal_column_name].isna() # type: ignore
        return BoolArray(is_missing_mask) # type: ignore

    def mask_get_missing_values(self, column_key: CK) -> BoolArray:
        """
        Get a boolean mask for missing (null) values in a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BoolArray: Boolean mask where True indicates missing values
        """
        with self._rlock:
            return self._mask_get_missing_values(column_key)

    def _mask_get_equal_to(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Internal: Get a boolean mask for values equal to a specific value. (No locks)
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates equal values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            is_equal_mask: pd.Series = self._internal_dataframe[internal_column_name] == item.value_in_unit(unit) # type: ignore
        else:
            is_equal_mask: pd.Series = self._internal_dataframe[internal_column_name] == item # type: ignore
        return BoolArray(is_equal_mask) # type: ignore

    def mask_get_equal_to(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Get a boolean mask for values equal to a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates equal values
        """
        with self._rlock:
            return self._mask_get_equal_to(column_key, item)

    def _mask_get_not_equal_to(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Internal: Get a boolean mask for values not equal to a specific value. (No locks)
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates not equal values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            is_not_equal_mask: pd.Series = self._internal_dataframe[internal_column_name] != item.value_in_unit(unit) # type: ignore
        else:
            is_not_equal_mask: pd.Series = self._internal_dataframe[internal_column_name] != item # type: ignore
        return BoolArray(is_not_equal_mask) # type: ignore

    def mask_get_not_equal_to(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Get a boolean mask for values not equal to a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates not equal values
        """
        with self._rlock:
            return self._mask_get_not_equal_to(column_key, item)

    def _mask_get_greater_than(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Internal: Get a boolean mask for values greater than a specific value. (No locks)
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates greater values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            is_greater_mask: pd.Series = self._internal_dataframe[internal_column_name] > item.value_in_unit(unit) # type: ignore
        else:
            is_greater_mask: pd.Series = self._internal_dataframe[internal_column_name] > item # type: ignore
        return BoolArray(is_greater_mask) # type: ignore

    def mask_get_greater_than(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Get a boolean mask for values greater than a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates greater values
        """
        with self._rlock:
            return self._mask_get_greater_than(column_key, item)
        
    def _mask_get_greater_equal(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Internal: Get a boolean mask for values greater than or equal to a specific value. (No locks)
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates greater values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            is_greater_mask: pd.Series = self._internal_dataframe[internal_column_name] >= item.value_in_unit(unit) # type: ignore
        else:
            is_greater_mask: pd.Series = self._internal_dataframe[internal_column_name] >= item # type: ignore
        return BoolArray(is_greater_mask) # type: ignore

    def mask_get_greater_equal(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Get a boolean mask for values greater than or equal to a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates greater values
        """
        with self._rlock:
            return self._mask_get_greater_equal(column_key, item)

    def _mask_get_less_than(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Internal: Get a boolean mask for values less than a specific value. (No locks)
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates lesser values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            is_less_mask: pd.Series = self._internal_dataframe[internal_column_name] < item.value_in_unit(unit) # type: ignore
        else:
            is_less_mask: pd.Series = self._internal_dataframe[internal_column_name] < item # type: ignore
        return BoolArray(is_less_mask) # type: ignore

    def mask_get_less_than(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Get a boolean mask for values less than a specific value.
        
        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates lesser values
        """
        with self._rlock:
            return self._mask_get_less_than(column_key, item)
        
    def _mask_get_less_equal(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Internal: Get a boolean mask for values less than or equal to a specific value. (No locks)

        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates lesser values
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            is_less_mask: pd.Series = self._internal_dataframe[internal_column_name] <= item.value_in_unit(unit) # type: ignore
        else:
            is_less_mask: pd.Series = self._internal_dataframe[internal_column_name] <= item # type: ignore
        return BoolArray(is_less_mask) # type: ignore

    def mask_get_less_equal(self, column_key: CK, item: SCALAR_TYPE|VALUE_TYPE) -> BoolArray:
        """
        Get a boolean mask for values less than or equal to a specific value.

        Args:
            column_key (CK): The column key
            item (SCALAR_TYPE|VALUE_TYPE): The value to compare against
            
        Returns:
            BoolArray: Boolean mask where True indicates lesser values
        """
        with self._rlock:
            return self._mask_get_less_equal(column_key, item)

    def _mask_get_in_range(self, column_key: CK, min_item: SCALAR_TYPE|VALUE_TYPE, max_item: SCALAR_TYPE|VALUE_TYPE, filter_mode: Literal["inclusive", "exclusive"] = "inclusive") -> BoolArray:
        """
        Internal: Get a boolean mask for values within a specific range (inclusive). (No locks)
        
        Args:
            column_key (CK): The column key
            min_item (SCALAR_TYPE|VALUE_TYPE): The minimum value (inclusive)
            max_item (SCALAR_TYPE|VALUE_TYPE): The maximum value (inclusive)
            filter_mode (Literal["inclusive", "exclusive"]): The mode of the filter. "inclusive" means that the min_item and max_item are included and will give TRUE.
            
        Returns:
            BoolArray: Boolean mask where True indicates values in range
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(min_item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            match filter_mode:
                case "inclusive":
                    is_in_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] >= min_item.value_in_unit(unit)) & (self._internal_dataframe[internal_column_name] <= max_item.value_in_unit(unit)) # type: ignore
                case "exclusive":
                    is_in_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] > min_item.value_in_unit(unit)) & (self._internal_dataframe[internal_column_name] < max_item.value_in_unit(unit)) # type: ignore
                case _: # type: ignore
                    raise ValueError(f"Invalid filter mode: {filter_mode}")
        else:
            match filter_mode:
                case "inclusive":
                    is_in_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] >= min_item) & (self._internal_dataframe[internal_column_name] <= max_item) # type: ignore
                case "exclusive":
                    is_in_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] > min_item) & (self._internal_dataframe[internal_column_name] < max_item) # type: ignore
                case _: # type: ignore
                    raise ValueError(f"Invalid filter mode: {filter_mode}")
        return BoolArray(is_in_range_mask) # type: ignore

    def mask_get_in_range(self, column_key: CK, min_item: SCALAR_TYPE|VALUE_TYPE, max_item: SCALAR_TYPE|VALUE_TYPE, filter_mode: Literal["inclusive", "exclusive"] = "inclusive") -> BoolArray:
        """
        Get a boolean mask for values within a specific range (inclusive).
        
        Args:
            column_key (CK): The column key
            min_item (SCALAR_TYPE|VALUE_TYPE): The minimum value (inclusive)
            max_item (SCALAR_TYPE|VALUE_TYPE): The maximum value (inclusive)
            filter_mode (Literal["inclusive", "exclusive"]): The mode of the filter. "inclusive" means that the min_item and max_item are included and will give TRUE.
            
        Returns:
            BoolArray: Boolean mask where True indicates values in range
        """
        with self._rlock:
            return self._mask_get_in_range(column_key, min_item, max_item, filter_mode)
        
    def _mask_get_outside_range(self, column_key: CK, min_item: SCALAR_TYPE|VALUE_TYPE, max_item: SCALAR_TYPE|VALUE_TYPE, filter_mode: Literal["inclusive", "exclusive"] = "exclusive") -> BoolArray:
        """
        Internal: Get a boolean mask for values outside a specific range (exclusive). (No locks)

        Args:
            column_key (CK): The column key
            min_item (SCALAR_TYPE|VALUE_TYPE): The minimum value (inclusive)
            max_item (SCALAR_TYPE|VALUE_TYPE): The maximum value (inclusive)
            filter_mode (Literal["inclusive", "exclusive"]): The mode of the filter. "inclusive" means that the min_item and max_item are included and will give TRUE.

        Returns:
            BoolArray: Boolean mask where True indicates values outside range
        """
        if column_key not in self._column_keys:
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        internal_column_name = self._get_internal_dataframe_column_name(column_key)
        if isinstance(min_item, UnitedScalar):
            if not self._unit_has(column_key):
                raise ValueError(f"Column {column_key} has no unit, so it cannot be compared to a UnitedScalar.")
            unit: Unit = self._unit_get(column_key)
            match filter_mode:
                case "inclusive":
                    is_outside_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] <= min_item.value_in_unit(unit)) | (self._internal_dataframe[internal_column_name] >= max_item.value_in_unit(unit)) # type: ignore
                case "exclusive":
                    is_outside_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] < min_item.value_in_unit(unit)) | (self._internal_dataframe[internal_column_name] > max_item.value_in_unit(unit)) # type: ignore
                case _: # type: ignore
                    raise ValueError(f"Invalid filter mode: {filter_mode}")
        else:
            match filter_mode:
                case "inclusive":
                    is_outside_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] <= min_item) | (self._internal_dataframe[internal_column_name] >= max_item) # type: ignore
                case "exclusive":   
                    is_outside_range_mask: pd.Series = (self._internal_dataframe[internal_column_name] < min_item) | (self._internal_dataframe[internal_column_name] > max_item) # type: ignore
                case _: # type: ignore
                    raise ValueError(f"Invalid filter mode: {filter_mode}")
        return BoolArray(is_outside_range_mask) # type: ignore

    def mask_get_outside_range(self, column_key: CK, min_item: SCALAR_TYPE|VALUE_TYPE, max_item: SCALAR_TYPE|VALUE_TYPE, filter_mode: Literal["inclusive", "exclusive"] = "exclusive") -> BoolArray:
        """
        Get a boolean mask for values outside a specific range (exclusive).

        Args:
            column_key (CK): The column key
            min_item (SCALAR_TYPE|VALUE_TYPE): The minimum value (inclusive)
            max_item (SCALAR_TYPE|VALUE_TYPE): The maximum value (inclusive)
            filter_mode (Literal["inclusive", "exclusive"]): The mode of the filter. "inclusive" means that the min_item and max_item are included and will give TRUE.

        Returns:
            BoolArray: Boolean mask where True indicates values outside range
        """
        with self._rlock:
            return self._mask_get_outside_range(column_key, min_item, max_item, filter_mode)

    def _mask_get_complete_rows(self, *column_keys: CK) -> BoolArray:
        """
        Internal: Return a boolean mask indicating which rows have no missing values. (No locks)
        
        Returns:
            BoolArray: Boolean array where True indicates rows with no missing values
        """
        if len(column_keys) == 0:
            return BoolArray(self._internal_dataframe.notna().all(axis=1)) # type: ignore
        else:
            column_names_to_check: list[str] = self._get_internal_dataframe_column_names(column_keys)
            return BoolArray(self._internal_dataframe[column_names_to_check].notna().all(axis=1)) # type: ignore

    def mask_get_complete_rows(self, *column_keys: CK) -> BoolArray:
        """
        Return a boolean mask indicating which rows have no missing values.
        
        Returns:
            BoolArray: Boolean array where True indicates rows with no missing values
        """
        with self._rlock:
            return self._mask_get_complete_rows(*column_keys)

    def _mask_get_incomplete_rows(self, *column_keys: CK) -> BoolArray:
        """
        Internal: Method to apply a boolean mask to filter rows in the dataframe. (No locks)

        Args:
            column_keys (CK): The column keys to check

        Returns:
            BoolArray: Boolean array where True indicates rows with at least one missing value
        """

        if len(column_keys) == 0:
            return BoolArray(~self._internal_dataframe.notna().all(axis=1)) # type: ignore
        else:
            column_names_to_check: list[str] = self._get_internal_dataframe_column_names(column_keys)
            return BoolArray(~self._internal_dataframe[column_names_to_check].notna().all(axis=1)) # type: ignore

    def mask_get_incomplete_rows(self, *column_keys: CK) -> BoolArray:
        """
        Return a boolean mask indicating which rows have at least one missing value.

        Args:
            column_keys (CK): The column keys to check
        
        Returns:
            BoolArray: Boolean array where True indicates rows with at least one missing value
        """
        with self._rlock:
            return self._mask_get_incomplete_rows(*column_keys)
        
    def _mask_get_by_function(self, filter_func: Callable[[Mapping[CK, SCALAR_TYPE]], bool], column_keys: list[CK]|None = None) -> BoolArray:
        """
        Internal: Return a boolean mask indicating which rows satisfy a custom function. (No locks)

        Args:
            filter_func (Callable[[Mapping[CK, SCALAR_TYPE]], bool]): Function that takes a row and returns True/False

        Returns:
            BoolArray: Boolean array where True indicates rows that satisfy the function
        """
        return BoolArray([filter_func(self._row_get_as_dict(i, column_keys=column_keys)) for i in range(self._number_of_rows())]) # type: ignore

    def mask_get_by_function(self, filter_func: Callable[[Mapping[CK, SCALAR_TYPE]], bool], column_keys: list[CK]|None = None) -> BoolArray:
        """
        Return a boolean mask indicating which rows satisfy a custom function.

        Args:
            filter_func (Callable[[Mapping[CK, SCALAR_TYPE]], bool]): Function that takes a row and returns True/False

        Returns:
            BoolArray: Boolean array where True indicates rows that satisfy the function
        """
        with self._rlock:
            return self._mask_get_by_function(filter_func, column_keys)

    # ----------- Mask Operations: Application ------------

    def _mask_apply_to_dataframe(self, mask: BoolArray) -> "UnitedDataframe[CK]":
        """
        Internal: Method to apply a boolean mask to filter rows in the dataframe. (No locks, no read-only check)

        Args:
            mask (np.ndarray): A 1D boolean array of length equal to the number of rows.

        Returns:
            UnitedDataframe[CK]: A new dataframe with only the rows where mask is True.
        """
        
        if len(mask) != self._number_of_rows():
            raise ValueError(f"Mask must have length {self._number_of_rows()}, got {len(mask)}")

        row_indices: list[int] = np.nonzero(mask.canonical_np_array)[0].tolist()
        if len(row_indices) == 0:
            return self.create_empty(column_keys=self._column_keys, column_types=self._column_types, column_units_or_dimensions=self._column_units, internal_dataframe_column_name_formatter=self._internal_dataframe_column_name_formatter)
        else:
            cropped_df: "UnitedDataframe[CK]" = self._crop_dataframe(row_indices=row_indices)
            return cropped_df

    def mask_apply_to_dataframe(self, mask: BoolArray) -> "UnitedDataframe[CK]":
        """
        Apply a boolean mask to the dataframe, returning a new filtered dataframe.
        
        Args:
            mask (BoolArray): Boolean mask to apply
            
        Returns:
            UnitedDataframe: New filtered dataframe
        """
        with self._rlock:
            return self._mask_apply_to_dataframe(mask)

    def _mask_get_row_indices(self, mask: BoolArray) -> list[int]:
        """
        Internal: Get the row indices where a boolean mask is True. (No locks)
        
        Args:
            mask (BoolArray): Boolean mask
            
        Returns:
            List[int]: List of row indices where mask is True
        """
        if len(mask) != self._number_of_rows():
            raise ValueError(f"Mask length ({len(mask)}) does not match dataframe length ({self._number_of_rows()}).")
        
        return [i for i, value in enumerate(mask) if value]

    def mask_get_row_indices(self, mask: BoolArray) -> list[int]:
        """
        Get the row indices where a boolean mask is True.
        
        Args:
            mask (BoolArray): Boolean mask
            
        Returns:
            List[int]: List of row indices where mask is True
        """
        with self._rlock:
            return self._mask_get_row_indices(mask)

    # ----------- Mask operations ------------

    def _mask_get_duplicates(self, subset: list[CK] | None = None, keep: str = "first") -> BoolArray:
        """
        Internal: Return a boolean mask indicating which rows are duplicates. (No locks)
        
        Args:
            subset (list[CK] | None): List of column keys to check for duplicates.
                                    If None, checks all columns.
            keep (str): Which duplicates to mark as True:
                       - "first": Mark duplicates as True, except for the first occurrence
                       - "last": Mark duplicates as True, except for the last occurrence  
                       - "all": Mark all duplicates as True
        
        Returns:
            BoolArray: Boolean array where True indicates duplicate rows
        """
        if subset is None:
            subset = self._column_keys
        
        # Validate subset
        for column_key in subset:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Get internal column names
        internal_columns = [self._get_internal_dataframe_column_name(col) for col in subset]
        
        # Create mask for duplicates
        if keep == "all":
            # Mark all duplicates as True (including first occurrences)
            mask: pd.Series = self._internal_dataframe[internal_columns].duplicated(keep=False) # type: ignore
        else:
            # Mark duplicates as True, except for first/last occurrence
            mask: pd.Series = self._internal_dataframe[internal_columns].duplicated(keep=keep) # type: ignore
        
        return BoolArray(mask) # type: ignore

    def mask_get_duplicates(self, subset: list[CK] | None = None, keep: str = "first") -> BoolArray:
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
            BoolArray: Boolean array where True indicates duplicate rows
        """
        with self._rlock:
            return self._mask_get_duplicates(subset, keep)
        
    def _mask_remove_duplicates(self, subset: list[CK] | None = None, keep: Literal["first", "last"] = "first", inplace: Literal[False, True] = False) -> Optional["UnitedDataframe[CK]"]: # type: ignore
        """
        Internal: Remove duplicate rows and return a new dataframe. (No locks)
        
        Args:
            subset (list[CK] | None): List of column keys to check for duplicates.
                                    If None, checks all columns.
            keep (str): Which duplicates to keep:
                       - "first": Keep the first occurrence of each duplicate
                       - "last": Keep the last occurrence of each duplicate
                       - False: Remove all duplicates
            inplace (bool | None): Whether to modify the dataframe in place.
                                   If None, returns a new dataframe.
        
        Returns:
            UnitedDataframe[CK]: New dataframe with duplicates removed
        """
        if subset is None:
            subset = self._column_keys
        
        # Validate subset
        for column_key in subset:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Get internal column names
        internal_columns = [self._get_internal_dataframe_column_name(col) for col in subset]
        
        # Remove duplicates
        if inplace:
            self._internal_dataframe.drop_duplicates(subset=internal_columns, keep=keep, inplace=inplace)
            return None

        else:
            deduplicated_df = self._internal_dataframe.drop_duplicates(subset=internal_columns, keep=keep, inplace=False) # type: ignore
            return self._create_with_replaced_internal_dataframe(deduplicated_df, copy_dataframe=False)

    @overload
    def mask_remove_duplicates(self, subset: list[CK] | None = None, keep: Literal["first", "last"] = "first", inplace: Literal[False] = False) -> "UnitedDataframe[CK]": # type: ignore
        ...
    @overload
    def mask_remove_duplicates(self, subset: list[CK] | None = None, keep: Literal["first", "last"] = "first", inplace: Literal[True] = True) -> None:
        ...
    def mask_remove_duplicates(self, subset: list[CK] | None = None, keep: Literal["first", "last"] = "first", inplace: Literal[False, True] = False) -> Optional["UnitedDataframe[CK]"]: # type: ignore
        """
        Remove duplicate rows and return a new dataframe.
        
        Args:
            subset (list[CK] | None): List of column keys to check for duplicates.
                                    If None, checks all columns.
            keep (str): Which duplicates to keep:
                       - "first": Keep the first occurrence of each duplicate
                       - "last": Keep the last occurrence of each duplicate
                       - False: Remove all duplicates
            inplace (bool | None): Whether to modify the dataframe in place.
                                   If None, returns a new dataframe.
        
        Returns:
            UnitedDataframe[CK]: New dataframe with duplicates removed
        """
        with self._rlock:
            return self._mask_remove_duplicates(subset, keep, inplace)