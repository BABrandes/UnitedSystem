from typing import Generic, TypeVar, Iterator, overload, Literal
from ..united_dataframe.united_dataframe import United_Dataframe, Column_Key
from ..scalars.united_scalar import UnitedScalar
from ..arrays.united_array import UnitedArray
from ..units.unit_quantity import UnitQuantity
from ..units.unit import Unit
from ..united_dataframe.united_dataframe import Value_Type
import pandas as pd
import numpy as np

CK = TypeVar("CK", bound=Column_Key|str)

class _ColumnAccessor(Generic[CK]):
    """
    Internal class for column-based access to cell values.
    
    Provides a pandas-like interface for accessing individual cells.
    """
    def __init__(self, parent: United_Dataframe[CK], column_key: CK):
        self._parent: United_Dataframe[CK] = parent
        self._column_key: CK = column_key
        
    def __getitem__(self, row: int) -> UnitedScalar:
        return self._parent.cell_value_get(row, self._column_key)
    
    def __setitem__(self, row: int, value: UnitedScalar):
        self._parent.cell_value_set(row, self._column_key, value)

    def __len__(self) -> int:
        return len(self._parent)
    
    def __iter__(self) -> Iterator[UnitedScalar]:
        return self._parent.get_iterator_for_column(self._column_key)
    
    def __contains__(self, value: UnitedScalar) -> bool:
        return value in self._parent.get_iterator_for_column(self._column_key)
    
    def as_numpy_array(self, unit: Unit) -> np.ndarray:
        return self._parent.column_values_as_numpy_array(self._column_key, unit)
    
    def as_pandas_series(self, unit: Unit) -> pd.Series:
        return self._parent.column_values_as_pandas_series(self._column_key, unit)
    
    def as_united_array(self, display_unit: Unit|None=None) -> UnitedArray:
        if display_unit is None:
            display_unit = self._parent.display_unit(self._column_key)
        return self._parent.column_values_as_array(self._column_key, display_unit)

    def sum(self) -> UnitedScalar:
        return self._parent.colfun_sum(self._column_key)
    
    def mean(self) -> UnitedScalar:
        return self._parent.colfun_mean(self._column_key)
    
    def std(self) -> UnitedScalar:
        return self._parent.colfun_std(self._column_key)
    
    def min(self) -> UnitedScalar:
        return self._parent.colfun_min(self._column_key)
    
    def max(self) -> UnitedScalar:
        return self._parent.colfun_max(self._column_key)
    
    def unique(self) -> list[UnitedScalar]:
        return self._parent.colfun_unique(self._column_key)
    
    def smallest_positive_nonzero_value(self) -> UnitedScalar:
        return self._parent.colfun_smallest_positive_nonzero_value(self._column_key)
    
    def largest_positive_nonzero_value(self) -> UnitedScalar:
        return self._parent.colfun_largest_positive_nonzero_value(self._column_key)
    
    def largest_negative_nonzero_value(self) -> UnitedScalar:
        return self._parent.colfun_largest_negative_nonzero_value(self._column_key)
    
    def smallest_negative_nonzero_value(self) -> UnitedScalar:
        return self._parent.colfun_smallest_negative_nonzero_value(self._column_key)
    
    @overload
    def count(self) -> dict[UnitedScalar, int]: 
        """
        Count the number of occurrences of each unique value in the column.
        """
        return self._parent.colfun_count_value_occurances(self._column_key)

    @overload
    def count(self, value: UnitedScalar) -> int:
        """
        Count the number of occurrences of the specified value in the column.
        """
        return self._parent.colfun_count_value_occurances(self._column_key, value)

    def count(self, value: UnitedScalar|None=None) -> int|dict[UnitedScalar, int]:
        if value is None:
            return self._parent.colfun_count_value_occurances(self._column_key)
        else:
            return self._parent.colfun_count_value_occurances(self._column_key, value)

    def isna(self) -> np.ndarray:
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
        dataframe_column_name: str = self._parent.internal_dataframe_column_string(self._column_key)
        
        # Get the pandas Series for this column
        column_series: pd.Series = self._parent._internal_canonical_dataframe[dataframe_column_name]
        
        # Check for NA values
        result_series: pd.Series = pd.isna(column_series)

        # Convert the pandas Series to a numpy array
        return result_series.to_numpy()
    
    def notna(self) -> np.ndarray:
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
        return ~self.isna()
    
    def __ge__(self, other: UnitedScalar) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x >= other})
    
    def __le__(self, other: UnitedScalar) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by

        Returns:    
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x <= other})
    
    def __gt__(self, other: UnitedScalar) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x > other})  
    
    def __lt__(self, other: UnitedScalar) -> np.ndarray:
        """
        Get a numpy mask for the column based on the filter function.

        Args:
            other (UnitedScalar): The value to filter by    

        Returns:
            np.ndarray: A numpy mask for the column based on the filter function
        """
        return self._parent.maskfun_get_from_filter({self._column_key: lambda x: x < other})
    
    @property
    def display_unit(self) -> Unit:
        return self._parent.display_unit(self._column_key)
    
    @property
    def UnitQuantity(self) -> UnitQuantity:
        return self._parent.UnitQuantity(self._column_key)
    
    @property
    def value_type(self) -> Value_Type:
        return self._parent.value_type(self._column_key)
    
    @property
    def column_key(self) -> CK:
        return self._column_key
    
    def row_index(self, value: UnitedScalar, case: Literal["first", "last"] = "first") -> int:
        """
        Get the row index of the first occurrence of a value in a column.

        Args:
            value (UnitedScalar): The value to get the row index of
            case (Literal["first", "last"]): The case to get the row index of

        Returns:
            int: The row index of the first occurrence of the value in the column. Returns -1 if the value is not found.
        """
        return self._parent.colfun_row_index(self._column_key, value, case)


    

