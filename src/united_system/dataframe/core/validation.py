"""
Validation mixin for UnitedDataframe.

This module provides validation and compatibility checking functionality.
"""

from typing import Generic, TypeVar, List, Any, Optional, Union, Type, Set, Dict
import numpy as np
import pandas as pd
from pandas._typing import Dtype
from typing import Literal

from ...units.base_classes.base_unit import BaseUnit, UnitQuantity
from ...scalars.united_scalar import UnitedScalar
from ...utils import JSONable, HDF5able
from ...units.utils import United
from ...arrays.utils import ArrayLike
from ..utils import ColumnKey, ColumnInformation
from ..column_type import ColumnType, SCALAR_TYPE, ARRAY_TYPE

CK = TypeVar("CK", bound=ColumnKey|str, default=str)
CK_CF = TypeVar("CK_CF", bound=ColumnKey|str, default=str)


class ValidationMixin(Generic[CK]):
    """
    Mixin providing validation and compatibility checking functionality.
    
    This mixin handles:
    - Column validation and type checking
    - Compatibility validation between values and columns
    - Numeric column identification
    - Type-based column filtering
    """
    
    def compatible_with_column(self, column_key: CK, value: Union[SCALAR_TYPE, ARRAY_TYPE, np.ndarray, pd.Series]) -> bool:
        """Check if a value is compatible with a column's type and unit."""
        with self._rlock:  # type: ignore
            column_type: ColumnType = self.column_type(column_key)  # type: ignore
            
            # Check unit compatibility
            match column_type.value.has_unit, isinstance(value, United):
                case True, True:
                    # Both have units - check if they match
                    if value.unit_quantity != self.unit_quantity(column_key):  # type: ignore
                        return False
                case True, False:
                    # Column has unit, value doesn't
                    return False
                case False, False:
                    # Neither has units - OK
                    pass
                case False, True:
                    # Column has no unit, value has unit
                    return False
                case _:
                    raise ValueError(f"Invalid value type: {type(value)}")
            
            # Check value type compatibility
            return column_type.check_compatibility(value)
    
    def is_numeric(self, column_key: CK) -> bool:
        """Check if a column contains numeric data."""
        with self._rlock:  # type: ignore
            column_type = self.column_type(column_key)  # type: ignore
            return column_type.is_numeric
    
    def get_numeric_column_keys(self) -> List[CK]:
        """Get a list of column keys for numeric columns only."""
        with self._rlock:  # type: ignore
            return [column_key for column_key in self._column_keys if self.is_numeric(column_key)]  # type: ignore
    
    def column_keys_of_type(self, *column_key_types: Type[CK_CF]) -> List[CK_CF]:
        """Get column keys that match the specified types."""
        with self._rlock:  # type: ignore
            column_keys_to_keep: List[CK_CF] = []
            for column_key in self._column_keys:  # type: ignore
                if isinstance(column_key, tuple(column_key_types)):
                    column_keys_to_keep.append(column_key)  # type: ignore
            return column_keys_to_keep
    
    def column_information_of_type(self, *column_key_types: Type[CK_CF]) -> List[tuple[CK_CF, ColumnInformation[CK_CF]]]:
        """Get column information for columns that match the specified types."""
        with self._rlock:  # type: ignore
            column_information_list: List[tuple[CK_CF, ColumnInformation[CK_CF]]] = []
            for column_key in self._column_keys:  # type: ignore
                if isinstance(column_key, tuple(column_key_types)):
                    column_info = ColumnInformation[CK_CF](
                        column_key,  # type: ignore
                        self._unit_quantities[column_key],  # type: ignore
                        self._column_types[column_key],  # type: ignore
                        self._display_units[column_key]  # type: ignore
                    )
                    column_information_list.append((column_key, column_info))  # type: ignore
            return column_information_list
    
    def _check_scalar_compatibility(self, column_key: CK, value: SCALAR_TYPE) -> bool:
        """Internal method to check if a scalar value is compatible with a column."""
        return self.compatible_with_column(column_key, value)
    
    def _get_numpy_dtype_from_precision(self, 
                                       column_key_or_type: Union[CK, ColumnType], 
                                       precision: Optional[Literal[8, 16, 32, 64, 128, 256]]) -> Dtype:
        """Get the numpy dtype based on precision requirements."""
        if isinstance(column_key_or_type, ColumnType):
            column_type = column_key_or_type
        else:
            column_type = self.column_type(column_key_or_type)  # type: ignore
        
        if precision is None:
            return column_type.value.numpy_storage_options[0]
        else:
            for numpy_dtype in column_type.value.numpy_storage_options:
                if numpy_dtype.itemsize * 8 == precision:  # Convert bytes to bits
                    return numpy_dtype
            raise ValueError(f"Precision {precision} not available for column type {column_type}")
    
    def _validate_column_exists(self, column_key: CK) -> None:
        """Validate that a column exists, raising an error if not."""
        if not self.has_column(column_key):  # type: ignore
            raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
    
    def _validate_not_read_only(self) -> None:
        """Validate that the dataframe is not read-only, raising an error if it is."""
        if self.is_read_only():  # type: ignore
            raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
    
    def _validate_row_count_match(self, values: Union[List, np.ndarray, pd.Series]) -> None:
        """Validate that the number of values matches the number of rows."""
        if len(values) != len(self._internal_canonical_dataframe):  # type: ignore
            raise ValueError(
                f"The number of values ({len(values)}) does not match the number of rows "
                f"({len(self._internal_canonical_dataframe)})"  # type: ignore
            )
    
    def _validate_row_index(self, row_index: int) -> None:
        """Validate that a row index is within bounds."""
        if not (0 <= row_index < len(self._internal_canonical_dataframe)):  # type: ignore
            raise ValueError(
                f"Row index {row_index} is out of bounds. "
                f"The dataframe has {len(self._internal_canonical_dataframe)} rows."  # type: ignore
            )
