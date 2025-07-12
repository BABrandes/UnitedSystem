"""
Segment operations mixin for UnitedDataframe.

Contains all operations related to segment functionality,
including segmentation, aggregation, and segment-based operations.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import List, Callable, Union, Optional, TYPE_CHECKING
from collections.abc import Sequence
import numpy as np
from .dataframe_protocol import UnitedDataframeProtocol, CK
from ..column_type import SCALAR_TYPE
from ..grouping._segments import Segments

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class SegmentMixin(UnitedDataframeProtocol[CK]):
    """
    Segment operations mixin for UnitedDataframe.
    
    Provides all functionality related to segment operations,
    including segmentation, aggregation, and segment-based operations.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Segment Operations: Basic ------------

    def _segment(self, by: Union[CK, Sequence[CK]], sort: bool = True, dropna: bool = True) -> Segments[CK]:
        """
        Internal: Create a Segments object for the specified columns (no lock).
        
        Args:
            by (Any): Column key(s) to segment by
            sort (bool): Whether to sort segment keys
            dropna (bool): Whether to exclude NA values from segment keys
            
        Returns:
            Segments[CK]: Segments object for further operations
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
        
        # Create Segments object
        return Segments[CK](self, by) # type: ignore[reportUnknownArgumentType]

    def segment(self, by: Union[CK, List[CK]], sort: bool = True, dropna: bool = True) -> Segments[CK]:
        """
        Create a Segments object for the specified columns.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            sort (bool): Whether to sort segment keys
            dropna (bool): Whether to exclude NA values from segment keys
            
        Returns:
            Segments[CK]: Segments object for further operations
        """
        with self._rlock:  # Full IDE support!
            return self._segment(by, sort=sort, dropna=dropna)

    def segment_apply(self, by: Union[CK, Sequence[CK]], result_column_key: CK, func: Callable[["UnitedDataframe[CK]"], SCALAR_TYPE]) -> "UnitedDataframe[CK]":
        """
        Apply a function to each segment.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            result_column_key (CK): Column key for the result
            func (Callable): Function to apply to each segment
            
        Returns:
            UnitedDataframe: Result of applying function to each segment
        """
        with self._rlock:
            segmented = self._segment(by)
            return segmented.apply((result_column_key, func))

    # ----------- Segment Operations: Aggregation ------------

    def segment_sum(self, by: Union[CK, Sequence[CK]], columns: Optional[Sequence[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Sum values in segments.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            columns (Optional[List[CK]]): Columns to sum. If None, sums all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with segment sums
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            if columns is None:
                return segmented.sum()
            else:
                return segmented.sum(column_keys_to_aggregate=columns)

    def segment_mean(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Calculate mean values in segments.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            columns (Optional[List[CK]]): Columns to calculate mean. If None, calculates mean of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with segment means
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            if columns is None:
                return segmented.mean()
            else:
                return segmented.mean(column_keys_to_aggregate=columns)

    def segment_std(self, by: Union[CK, List[CK]], columns: Optional[List[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Calculate standard deviation values in segments.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            columns (Optional[List[CK]]): Columns to calculate std. If None, calculates std of all numeric columns.
            
        Returns:
            UnitedDataframe: Dataframe with segment standard deviations
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            if columns is None:
                return segmented.std()
            else:
                return segmented.std(column_keys_to_aggregate=columns)

    def segment_count(self, by: Union[CK, List[CK]], columns: Optional[Sequence[CK]] = None) -> "UnitedDataframe[CK]":
        """
        Count non-null values in segments.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            columns (Optional[Sequence[CK]]): Columns to count. If None, counts all columns.
            
        Returns:
            UnitedDataframe: Dataframe with segment counts
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            if columns is None:
                return segmented.count()
            else:
                return segmented.count(column_keys_to_consider=columns)

    def segment_size(self, by: Union[CK, List[CK]], result_column_key: CK) -> "UnitedDataframe[CK]":
        """
        Get the size of each segment.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            result_column_key (CK): Column key for the result
            
        Returns:
            UnitedDataframe: Dataframe with segment sizes
        """
        with self._rlock:
            segmented = self._segment(by)
            return segmented.size(result_column_key)

    # ----------- Segment Operations: Basic Access ------------

    def segment_head(self, by: Union[CK, Sequence[CK]], n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the first n rows from each segment.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            n (int): Number of rows to return from each segment
            
        Returns:
            UnitedDataframe: Dataframe with first n rows from each segment
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            return segmented.head(n)

    def segment_first(self, by: Union[CK, Sequence[CK]]) -> "UnitedDataframe[CK]":
        """
        Get the first row from each segment.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            
        Returns:
            UnitedDataframe: Dataframe with first row from each segment
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            return segmented.first()

    def segment_tail(self, by: Union[CK, Sequence[CK]], n: int = 1) -> "UnitedDataframe[CK]":
        """
        Get the last n rows from each segment.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            n (int): Number of rows to return from each segment
            
        Returns:
            UnitedDataframe: Dataframe with last n rows from each segment
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            return segmented.tail(n)

    def segment_last(self, by: Union[CK, Sequence[CK]]) -> "UnitedDataframe[CK]":
        """
        Get the last row from each segment.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            
        Returns:
            UnitedDataframe: Dataframe with last row from each segment
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            return segmented.last()

    # ----------- Segment Operations: Utility ------------

    def segment_isna(self, by: Union[CK, Sequence[CK]], subset: Optional[List[CK]] = None) -> "np.ndarray":
        """
        Check for missing values in segments.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            subset (Optional[List[CK]]): Columns to check. If None, checks all columns.
            
        Returns:
            np.ndarray: Boolean array indicating missing values
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            return segmented.isna(subset)

    def segment_notna(self, by: Union[CK, Sequence[CK]], subset: Optional[List[CK]] = None) -> "np.ndarray":
        """
        Check for non-missing values in segments.
        
        Args:
            by (Union[CK, List[CK]]): Column key(s) to segment by
            subset (Optional[List[CK]]): Columns to check. If None, checks all columns.
            
        Returns:
            np.ndarray: Boolean array indicating non-missing values
        """
        with self._rlock:
            segmented: Segments[CK] = self._segment(by)
            return segmented.notna(subset) 