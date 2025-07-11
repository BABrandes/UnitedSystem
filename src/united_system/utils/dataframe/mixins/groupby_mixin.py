"""
Groupby operations mixin for UnitedDataframe.

Contains all operations related to groupby functionality,
including grouping, aggregation, and group-based operations.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Dict, List, Callable, Union, Optional, TYPE_CHECKING
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..accessors._group_by import GroupBy

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class GroupbyMixin(UnitedDataframeProtocol[CK]):
    """
    Groupby operations mixin for UnitedDataframe.
    
    Provides all functionality related to groupby operations,
    including grouping, aggregation, and group-based operations.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Groupby Operations: Basic ------------

    def _groupby(self, by: Union[CK, List[CK]], sort: bool = True, dropna: bool = True) -> GroupBy[CK]:
        """
        Internal: Create a GroupBy object for the specified columns (no lock).
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            sort (bool): Whether to sort group keys
            dropna (bool): Whether to exclude NA values from group keys
            
        Returns:
            _GroupBy[CK]: GroupBy object for further operations
        """
        # Ensure by is a list
        if not isinstance(by, list):
            by = [by]
        
        # Validate that all column keys exist
        for column_key in by:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Create GroupBy object
        return GroupBy(self, by, sort=sort, dropna=dropna)

    def groupby(self, by: Union[CK, List[CK]], sort: bool = True, dropna: bool = True) -> GroupBy[CK]:
        """
        Create a GroupBy object for the specified columns.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            sort (bool): Whether to sort group keys
            dropna (bool): Whether to exclude NA values from group keys
            
        Returns:
            _GroupBy[CK]: GroupBy object for further operations
        """
        with self._rlock:  # Full IDE support!
            return self._groupby(by, sort=sort, dropna=dropna)

    def group_apply(self, by: Union[CK, List[CK]], func: Callable) -> UnitedDataframe[CK]:
        """
        Apply a function to each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            func (Callable): Function to apply to each group
            
        Returns:
            UnitedDataframe: Result of applying function to each group
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.apply(func)

    def group_transform(self, by: Union[CK, List[CK]], func: Callable) -> UnitedDataframe[CK]:
        """
        Transform each group using a function.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            func (Callable): Function to transform each group
            
        Returns:
            UnitedDataframe: Transformed dataframe
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.transform(func)

    # ----------- Groupby Operations: Aggregation ------------

    def group_aggregate(self, by: Union[CK, List[CK]], agg_dict: Dict[CK, Union[str, Callable]]) -> UnitedDataframe[CK]:
        """
        Aggregate groups using different functions for different columns.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            agg_dict (Dict[CK, Union[str, Callable]]): Dictionary mapping column keys to aggregation functions
            
        Returns:
            UnitedDataframe: Aggregated dataframe
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.agg(agg_dict)

    def group_sum(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Sum values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to sum. If None, sums all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group sums
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.sum()
            else:
                return grouped.sum(columns)

    def group_mean(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate mean values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate mean. If None, calculates mean of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group means
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.mean()
            else:
                return grouped.mean(columns)

    def group_median(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate median values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate median. If None, calculates median of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group medians
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.median()
            else:
                return grouped.median(columns)

    def group_std(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate standard deviation in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate std. If None, calculates std of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group standard deviations
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.std()
            else:
                return grouped.std(columns)

    def group_var(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate variance in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate variance. If None, calculates variance of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group variances
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.var()
            else:
                return grouped.var(columns)

    def group_min(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Find minimum values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to find minimum. If None, finds minimum of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group minimums
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.min()
            else:
                return grouped.min(columns)

    def group_max(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Find maximum values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to find maximum. If None, finds maximum of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group maximums
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.max()
            else:
                return grouped.max(columns)

    def group_count(self, by: Union[CK, List[CK]]) -> UnitedDataframe[CK]:
        """
        Count non-null values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            
        Returns:
            UnitedDataframe: Dataframe with group counts
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.count()

    def group_size(self, by: Union[CK, List[CK]]) -> UnitedDataframe[CK]:
        """
        Get the size of each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            
        Returns:
            UnitedDataframe: Dataframe with group sizes
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.size()

    # ----------- Groupby Operations: Advanced ------------

    def group_quantile(self, by: Union[CK, List[CK]], q: float, columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate quantiles in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            q (float): Quantile to calculate (0.0 to 1.0)
            columns (Optional[List[CK]]): Columns to calculate quantile. If None, calculates quantile of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group quantiles
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.quantile(q)
            else:
                return grouped.quantile(q, columns)

    def group_first(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Get the first value in each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to get first value. If None, gets first value of all columns.
            
        Returns:
            UnitedDataframe: Dataframe with first values in each group
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.first()
            else:
                return grouped.first(columns)

    def group_last(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Get the last value in each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to get last value. If None, gets last value of all columns.
            
        Returns:
            UnitedDataframe: Dataframe with last values in each group
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.last()
            else:
                return grouped.last(columns)

    def group_nth(self, by: Union[CK, List[CK]], n: int, columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Get the nth value in each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            n (int): Position to get (0-based)
            columns (Optional[List[CK]]): Columns to get nth value. If None, gets nth value of all columns.
            
        Returns:
            UnitedDataframe: Dataframe with nth values in each group
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.nth(n)
            else:
                return grouped.nth(n, columns)

    # ----------- Groupby Operations: Cumulative ------------

    def group_cumsum(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate cumulative sum within groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate cumulative sum. If None, calculates cumulative sum of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with cumulative sums
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.cumsum()
            else:
                return grouped.cumsum(columns)

    def group_cumprod(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Calculate cumulative product within groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate cumulative product. If None, calculates cumulative product of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with cumulative products
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.cumprod()
            else:
                return grouped.cumprod(columns)

    def group_shift(self, by: Union[CK, List[CK]], periods: int = 1, columns: Optional[List[CK]] = None) -> UnitedDataframe[CK]:
        """
        Shift values within groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            periods (int): Number of periods to shift
            columns (Optional[List[CK]]): Columns to shift. If None, shifts all columns.
            
        Returns:
            UnitedDataframe: Dataframe with shifted values
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.shift(periods)
            else:
                return grouped.shift(periods, columns)

    def group_rank(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None, method: str = "average") -> UnitedDataframe[CK]:
        """
        Calculate ranks within groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to rank. If None, ranks all numeric columns.
            method (str): Ranking method ('average', 'min', 'max', 'first', 'dense')
            
        Returns:
            UnitedDataframe: Dataframe with ranks
        """
        with self._rlock:
            grouped = self._groupby(by)
            if columns is None:
                return grouped.rank(method=method)
            else:
                return grouped.rank(columns, method=method)