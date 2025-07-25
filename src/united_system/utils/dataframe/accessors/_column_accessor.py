from typing import Generic, TypeVar, Iterator, overload, Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe
    from ....column_key import ColumnKey

from ....column_type import ARRAY_TYPE, ColumnType, NUMERIC_SCALAR_TYPE, SCALAR_TYPE
from ....dimension import Dimension
from ....unit import Unit
from ....bool_array import BoolArray
import pandas as pd
import numpy as np

CK = TypeVar("CK", bound="ColumnKey|str")
AT = TypeVar("AT", bound=ARRAY_TYPE)

class ColumnAccessor(Generic[CK]):
    """
    Internal class for column-based access to cell values.
    
    Provides a pandas-like interface for accessing individual cells.
    """
    def __init__(self, parent: "UnitedDataframe[CK]", column_key: CK, slice_: slice|None = None):
        self._parent: "UnitedDataframe[CK]" = parent
        self._column_key: CK = column_key

        if slice_ is not None:
            self._slice: slice = slice_
        else:
            self._slice: slice = slice(0, len(self._parent), 1)
        
    def __getitem__(self, row: int) -> SCALAR_TYPE:
        return self._parent.cell_get_value(row, self._column_key)
    
    def __setitem__(self, row: int, value: SCALAR_TYPE) -> None:
        self._parent.cell_set_value(row, self._column_key, value)

    def __len__(self) -> int:
        return len(self._parent)
    
    def __iter__(self) -> Iterator[SCALAR_TYPE]:
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column iteration not yet implemented")
    
    def __contains__(self, value: SCALAR_TYPE) -> bool:
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column contains not yet implemented")
    
    def as_numpy_array(self, unit: Unit|None=None) -> np.ndarray:
        return self._parent.column_get_as_numpy_array(self._column_key)
    
    def as_pandas_series(self, unit: Unit|None=None) -> pd.Series: # type: ignore[reportUnknownReturnType]
        return self._parent.column_get_as_pd_series(self._column_key) # type: ignore[reportUnknownReturnType]
    
    @overload
    def as_array(self) -> ARRAY_TYPE: ...
    @overload
    def as_array(self, expected_column_type: type[AT]) -> AT: ...
    def as_array(self, expected_column_type: type[AT]|None = None) -> AT|ARRAY_TYPE:
        """
        Get the column data for a column.
        
        Args:
            column_key (CK): The column key
            
        Returns:
            BaseArray: The column data
        """

        if expected_column_type is None:
            return self._parent.column_get_as_array(self._column_key, slice=self._slice)
        else:
            exp_type: type[AT] = expected_column_type # type: ignore        
            return self._parent.column_get_as_array(self._column_key, exp_type, self._slice)
    
    def sum(self) -> NUMERIC_SCALAR_TYPE:
        return self._parent.column_get_sum(self._column_key)
    
    def mean(self) -> NUMERIC_SCALAR_TYPE:
        return self._parent.column_get_mean(self._column_key)
    
    def std(self) -> NUMERIC_SCALAR_TYPE:
        return self._parent.column_get_std(self._column_key)
    
    def min(self) -> NUMERIC_SCALAR_TYPE:
        return self._parent.column_get_min(self._column_key)
    
    def max(self) -> NUMERIC_SCALAR_TYPE:
        return self._parent.column_get_max(self._column_key)
    
    def unique(self) -> list[SCALAR_TYPE]:
        return self._parent.column_get_unique(self._column_key)
    
    def smallest_positive_nonzero_value(self) -> NUMERIC_SCALAR_TYPE:
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column smallest_positive_nonzero_value not yet implemented")
    
    def largest_positive_nonzero_value(self) -> NUMERIC_SCALAR_TYPE:
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column largest_positive_nonzero_value not yet implemented")
    
    def largest_negative_nonzero_value(self) -> NUMERIC_SCALAR_TYPE:
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column largest_negative_nonzero_value not yet implemented")
    
    def smallest_negative_nonzero_value(self) -> NUMERIC_SCALAR_TYPE:
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column smallest_negative_nonzero_value not yet implemented")
    
    @overload
    def count(self) -> dict[SCALAR_TYPE, int]: 
        """
        Count the number of occurrences of each unique value in the column.
        """
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column count not yet implemented")

    @overload
    def count(self, value: SCALAR_TYPE) -> int:
        """
        Count the number of occurrences of the specified value in the column.
        """
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column count not yet implemented")

    def count(self, value: SCALAR_TYPE|None=None) -> int|dict[SCALAR_TYPE, int]:
        if value is None:
            return self._parent.column_count_non_missing_values(self._column_key)
        else:
            # This would need to be implemented in the parent dataframe
            raise NotImplementedError("Column count with value not yet implemented")

    def isna(self) -> BoolArray:
        """
        Return a boolean array indicating which values in this column are NA/NaN.
        
        This method works similarly to pandas Series.isna(), returning a numpy array
        with the same length as the column, where True indicates NA/NaN values.
        
        Returns:
            np.ndarray: Boolean array where True indicates NA values
            
        Examples:
            # Check which values in the column are NA
            na_mask = df['column_name'].isna()
            
            # Count NA values
            na_count = df['column_name'].isna().sum()
            
            # Filter to non-NA values
            non_na_values = df['column_name'][~df['column_name'].isna()]
        """

        return self._parent.mask_get_incomplete_rows(self._column_key)[self._slice]
    
    def notna(self) -> BoolArray:
        """
        Return a boolean array indicating which values in this column are not NA/NaN.
        
        This is the inverse of isna() - returns True for non-NA values.
        
        Returns:
            np.ndarray: Boolean array where True indicates non-NA values
            
        Examples:
            # Check which values in the column are not NA
            non_na_mask = df['column_name'].notna()
            
            # Count non-NA values
            non_na_count = df['column_name'].notna().sum()
            
            # Filter to non-NA values
            non_na_values = df['column_name'][df['column_name'].notna()]
        """
        return ~self.isna()[self._slice]
    
    def __ge__(self, other: SCALAR_TYPE) -> BoolArray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.mask_get_greater_equal(self._column_key, other)[self._slice]
    
    def __le__(self, other: SCALAR_TYPE) -> BoolArray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by

        Returns:    
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.mask_get_less_equal(self._column_key, other)[self._slice]
    
    def __gt__(self, other: SCALAR_TYPE) -> BoolArray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.mask_get_greater_than(self._column_key, other)[self._slice]
    
    def __lt__(self, other: SCALAR_TYPE) -> BoolArray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by    

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.mask_get_less_than(self._column_key, other)[self._slice]
    
    @property
    def unit(self) -> Unit:
        return self._parent.unit_get_unit(self._column_key)
    
    @property
    def unit_has(self) -> bool:
        return self._parent.unit_has_unit(self._column_key)

    @property
    def dimension(self) -> Dimension:
        if not self.dimension_has:
            raise ValueError(f"Column {self._column_key} has no dimension.")
        return self._parent.dim_get_dimension(self._column_key)

    @property
    def dimension_has(self) -> bool:
        return self._parent.dim_has_dimension(self._column_key)

    @property
    def column_type(self) -> ColumnType:
        return self._parent.coltype_get(self._column_key)
    
    @property
    def column_key(self) -> CK:
        return self._column_key
    
    def row_index(self, value: SCALAR_TYPE, case: Literal["first", "last"] = "first") -> int:
        """
        Get the row index of the first occurrence of a value in a column.

        Args:
            value (UnitedScalar): The value to get the row index of
            case (Literal["first", "last"]): The case to get the row index of

        Returns:
            int: The row index of the first occurrence of the value in the column. Returns -1 if the value is not found.
        """
        # This would need to be implemented in the parent dataframe
        raise NotImplementedError("Column row_index not yet implemented")


    

