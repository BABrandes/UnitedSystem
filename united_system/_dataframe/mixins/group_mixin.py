"""
Groupby operations mixin for UnitedDataframe.

Contains all operations related to groupby functionality,
including grouping, aggregation, and group-based operations.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import List, Callable, Union, Optional, TYPE_CHECKING
from collections.abc import Sequence
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..column_type import SCALAR_TYPE
from ..grouping._groups import Groups

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class GroupbyMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
    """
    Groupby operations mixin for UnitedDataframe.
    
    Provides all functionality related to groupby operations,
    including grouping, aggregation, and group-based operations.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Groupby Operations: Basic ------------

    def _groupby(self, by: Union[CK, Sequence[CK]], sort: bool = True, dropna: bool = True) -> Groups[CK]:
        """
        Internal: Create a GroupBy object for the specified columns (no lock).
        
        Args:
            by (Any): Column key(s) to group by
            sort (bool): Whether to sort group keys
            dropna (bool): Whether to exclude NA values from group keys
            
        Returns:
            _GroupBy[CK]: GroupBy object for further operations
        """
        # Ensure by is a list
        if not isinstance(by, list):
            by_: list[CK] = [by] # type: ignore[reportUnknownArgumentType]
        else:
            by_: list[CK]  = by
        
        # Validate that all column keys exist
        for column_key in by_:
            if column_key not in self._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Create GroupBy object
        return Groups[CK](self, by) # type: ignore[reportUnknownArgumentType]

    def groupby(self, by: Union[CK, Sequence[CK]], sort: bool = True, dropna: bool = True) -> Groups[CK]:
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

    def group_apply(self, by: Union[CK, Sequence[CK]], result_column_key: CK, func: Callable[["UnitedDataframe[CK]"], SCALAR_TYPE]) -> "UnitedDataframe[CK]":
        """
        Apply a function to each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            result_column_key (CK): Column key for the result
            func (Callable): Function to apply to each group
            
        Returns:
            UnitedDataframe: Result of applying function to each group
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.apply((result_column_key, func))

    # ----------- Groupby Operations: Aggregation ------------

    def group_sum(self, by: Union[CK, Sequence[CK]], columns: Optional[Sequence[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Sum values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to sum. If None, sums all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group sums
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            if columns is None:
                return grouped.sum()
            else:
                return grouped.sum(columns)

    def group_mean(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Calculate mean values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            columns (Optional[List[CK]]): Columns to calculate mean. If None, calculates mean of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with group means
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            if columns is None:
                return grouped.mean()
            else:
                return grouped.mean(columns)

    def group_count(self, by: Union[CK, List[CK]], result_column_keys: Optional[List[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Count non-null values in groups.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            result_column_keys (Optional[List[CK]]): Column keys for the result
            
        Returns:
            UnitedDataframe: Dataframe with group counts
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            return grouped.count(result_column_keys)

    def group_size(self, by: Union[CK, List[CK]], result_column_key: CK) -> "UnitedDataframe[CK]":
        """
        Get the size of each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            result_column_key (CK): Column key for the result
            
        Returns:
            UnitedDataframe: Dataframe with group sizes
        """
        with self._rlock:
            grouped = self._groupby(by)
            return grouped.size(result_column_key)

    # ----------- Groupby Operations: Basic Access ------------

    def group_head(self, by: Union[CK, Sequence[CK]], n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the first n rows from each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            n (int): Number of rows to return from each group
            
        Returns:
            UnitedDataframe: Dataframe with first n rows from each group
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            return grouped.head(n)

    def group_first(self, by: Union[CK, Sequence[CK]]) -> "UnitedDataframe[CK]":
        """
        Get the first row from each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            
        Returns:
            UnitedDataframe: Dataframe with first row from each group
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            return grouped.first()

    def group_tail(self, by: Union[CK, Sequence[CK]], n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the last n rows from each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            n (int): Number of rows to return from each group
            
        Returns:
            UnitedDataframe: Dataframe with last n rows from each group
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            return grouped.tail(n)

    def group_last(self, by: Union[CK, Sequence[CK]]) -> "UnitedDataframe[CK]":
        """
        Get the last row from each group.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to group by
            
        Returns:
            UnitedDataframe: Dataframe with last row from each group
        """
        with self._rlock:
            grouped: Groups[CK] = self._groupby(by)
            return grouped.last()