"""
Column statistics mixin for UnitedDataframe.

Contains all statistical operations on columns, including min, max, mean, 
standard deviation, percentiles, and other statistical measures.
"""

from typing import Generic, TypeVar, Literal
import numpy as np

from ..column_type import ColumnType, SCALAR_TYPE, ARRAY_TYPE, NUMERIC_SCALAR_TYPE

CK = TypeVar("CK", bound=str, default=str)

class ColumnStatisticsMixin(Generic[CK]):
    """
    Column statistics mixin for UnitedDataframe.
    
    Provides all statistical operations on columns, including min, max, mean,
    standard deviation, percentiles, and other statistical measures.
    """

    # ----------- Column statistics and analysis ------------

    def column_get_min(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> NUMERIC_SCALAR_TYPE:
        """
        Get the minimum value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The minimum value with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.min()

    def column_get_max(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> NUMERIC_SCALAR_TYPE:
        """
        Get the maximum value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The maximum value with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.max()

    def column_get_mean(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> NUMERIC_SCALAR_TYPE:
        """
        Get the mean (average) value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The mean value with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.mean()

    def column_get_std(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all", ddof: int = 1) -> NUMERIC_SCALAR_TYPE:
        """
        Get the standard deviation of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            ddof (int): Delta degrees of freedom (default 1 for sample std)
            
        Returns:
            UnitedScalar: The standard deviation with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.std(ddof=ddof)

    def column_get_sum(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> NUMERIC_SCALAR_TYPE:
        """
        Get the sum of values in a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The sum with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.sum()

    def column_get_median(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> NUMERIC_SCALAR_TYPE:
        """
        Get the median value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The median value with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.median()

    def column_get_percentile(self, column_key: CK, percentile: float, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> NUMERIC_SCALAR_TYPE:
        """
        Get a specific percentile value of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            percentile (float): The percentile to calculate (0.0 to 1.0)
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            
        Returns:
            UnitedScalar: The percentile value with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            if not 0 <= percentile <= 1:
                raise ValueError(f"Percentile must be between 0 and 1, got {percentile}")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.quantile(percentile)

    def column_get_var(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all", ddof: int = 1) -> NUMERIC_SCALAR_TYPE:
        """
        Get the variance of a column with optional filtering.
        
        Args:
            column_key (CK): The column key of the column
            case (str): Filtering criteria for the values:
                - "only_positive": only positive values (value > 0)
                - "only_negative": only negative values (value < 0)
                - "only_non_negative": only non-negative values (value >= 0)
                - "only_non_positive": only non-positive values (value <= 0)
                - "all": all values
            ddof (int): Delta degrees of freedom (default 1 for sample variance)
            
        Returns:
            UnitedScalar: The variance with appropriate unit information
        """
        with self._rlock:
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_values = self.column(column_key)
            
            match case:
                case "only_positive":
                    filtered_values = column_values[column_values > 0]
                case "only_negative":
                    filtered_values = column_values[column_values < 0]
                case "only_non_negative":
                    filtered_values = column_values[column_values >= 0]
                case "only_non_positive":
                    filtered_values = column_values[column_values <= 0]
                case "all":
                    filtered_values = column_values
                case _:
                    raise ValueError(f"Invalid case: {case}")
            
            if len(filtered_values) == 0:
                raise ValueError(f"No values found matching criteria '{case}' in column '{column_key}'")
            
            return filtered_values.var(ddof=ddof)

    def column_count_valid(self, column_key: CK) -> int:
        """
        Count the number of valid (non-NaN) values in a column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            int: Number of valid values
        """
        with self._rlock:
            column_values = self.column(column_key)
            return column_values.count()

    def column_count_missing(self, column_key: CK) -> int:
        """
        Count the number of missing (NaN) values in a column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            int: Number of missing values
        """
        with self._rlock:
            column_values = self.column(column_key)
            return column_values.isna().sum()

    def column_count_unique(self, column_key: CK) -> int:
        """
        Count the number of unique values in a column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            int: Number of unique values
        """
        with self._rlock:
            column_values = self.column(column_key)
            return column_values.nunique()

    def column_get_unique(self, column_key: CK) -> ARRAY_TYPE:
        """
        Get the unique values in a column.
        
        Args:
            column_key (CK): The column key of the column
            
        Returns:
            ARRAY_TYPE: Array of unique values
        """
        with self._rlock:
            column_values = self.column(column_key)
            return column_values.unique() 