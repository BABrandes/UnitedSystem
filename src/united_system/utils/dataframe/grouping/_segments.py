from typing import TypeVar, Tuple, Callable, TYPE_CHECKING
from collections.abc import Sequence
from bidict import bidict

from ._base_grouping import BaseGrouping, GroupingContainer
from ..column_key import ColumnKey
from ..column_type import SCALAR_TYPE, LOWLEVEL_TYPE
from ..accessors._row_accessor import RowAccessor
if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

CK = TypeVar("CK", bound=ColumnKey|str)

class Segments(BaseGrouping[CK]):
    """
    A Segments object for performing segmented operations on United_Dataframe.
    
    This class provides pandas-like groupby functionality with unit awareness.
    Creates segments where a new segment starts whenever any of the grouping values changes.
    """

    ######################### Initialization #########################
    
    def __init__(
            self,
            dataframe: "UnitedDataframe[CK]",
            by_unique_values_of_columns: Sequence[CK] = [],
            by_unique_results_of_row_functions: Sequence[Tuple[CK, Callable[[RowAccessor[CK]], SCALAR_TYPE]]] = []):
        """
        Initialize a Segments object.
        
        Args:
            dataframe: The United_Dataframe to group
            by_unique_values_of_columns: List of columns for the groupings
            by_unique_results_of_row_functions: List of tuples of column keys for the results of the row functions and functions to apply to each row to get a unique result for each grouping
        """
        super().__init__(dataframe, by_unique_values_of_columns, by_unique_results_of_row_functions)
        
        # Create segments
        self._create_segments()

    def _create_segments(self) -> None:
        """Create segment dataframes based on changes in key values."""
        # Get categorical column names
        categorical_column_names = [col_info.internal_dataframe_column_name for col_info in self._categorical_column_information.values()]
        categorical_column_keys = list(self._categorical_column_information.keys())
        
        if not categorical_column_names:
            # No grouping columns, create single segment with all data
            self._create_single_segment(categorical_column_keys)
            return
        
        # Create group keys (combinations of values) directly from the dataframe
        group_data = self._working_df[categorical_column_names].values # type: ignore
        all_group_keys = [tuple(row) for row in group_data]
        
        # Find segment boundaries where key values change
        segment_boundaries = self._find_segment_boundaries(all_group_keys)
        
        # Create segments based on boundaries
        self._create_segments_from_boundaries(segment_boundaries, categorical_column_keys)

    def _find_segment_boundaries(self, all_group_keys: list[tuple[LOWLEVEL_TYPE, ...]]) -> list[int]:
        """Find indices where segment boundaries occur (when key values change)."""
        if not all_group_keys:
            return []
        
        boundaries = [0]  # Start with first row
        
        for i in range(1, len(all_group_keys)):
            if all_group_keys[i] != all_group_keys[i-1]:
                boundaries.append(i)
        
        boundaries.append(len(all_group_keys))  # End with last row
        return boundaries

    def _create_segments_from_boundaries(self, boundaries: list[int], categorical_column_keys: list[CK]) -> None:
        """Create segments based on the boundary indices."""
        # Prepare common container data
        internal_dataframe_column_names = bidict({col: col_info.internal_dataframe_column_name for col, col_info in self._available_column_information.items()})
        available_column_keys = list(self._available_column_information.keys())
        available_column_types = {col: col_info.column_type for col, col_info in self._available_column_information.items()}
        available_column_units = {col: col_info.column_unit for col, col_info in self._available_column_information.items()}
        
        # Create segments
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Get segment dataframe
            segment_df = self._working_df.iloc[start_idx:end_idx].copy() # type: ignore
            
            # Get the key values for this segment (from first row)
            segment_key: tuple[LOWLEVEL_TYPE, ...] = tuple(self._working_df.iloc[start_idx][col_info.internal_dataframe_column_name] for col_info in self._categorical_column_information.values()) # type: ignore
            
            # Create GroupingContainer
            self._grouping_containers.append(GroupingContainer(
                parent_united_dataframe=self._dataframe,
                dataframe=segment_df,
                internal_dataframe_column_names=internal_dataframe_column_names,
                available_column_keys=available_column_keys,
                available_column_types=available_column_types,
                available_column_units=available_column_units,
                categorical_column_keys=categorical_column_keys,
                categorical_key_values=segment_key
            ))

    def _create_single_segment(self, categorical_column_keys: list[CK]) -> None:
        """Create a single segment when no grouping columns are specified."""
        # Prepare common container data
        internal_dataframe_column_names = bidict({col: col_info.internal_dataframe_column_name for col, col_info in self._available_column_information.items()})
        available_column_keys = list(self._available_column_information.keys())
        available_column_types = {col: col_info.column_type for col, col_info in self._available_column_information.items()}
        available_column_units = {col: col_info.column_unit for col, col_info in self._available_column_information.items()}
        
        # Create single segment with all data
        self._grouping_containers.append(GroupingContainer(
            parent_united_dataframe=self._dataframe,
            dataframe=self._working_df.copy(),
            internal_dataframe_column_names=internal_dataframe_column_names,
            available_column_keys=available_column_keys,
            available_column_types=available_column_types,
            available_column_units=available_column_units,
            categorical_column_keys=categorical_column_keys,
            categorical_key_values=()
        ))
    
    ######################### Properties #########################
    
    @property
    def segments(self) -> "list[UnitedDataframe[CK]]":
        """
        Get the segmented dataframes.
        
        Returns:
            list: List of United_Dataframe instances for each segment
        """
        return self.groupings()
    
    @property
    def segment_keys(self) -> list[tuple[LOWLEVEL_TYPE, ...]]:
        """
        Get the segment keys.
        
        Returns:
            list: List of segment key tuples
        """
        return self.categorical_key_values
    
    def get_segment_by_key(self, segment_key: tuple[LOWLEVEL_TYPE, ...]) -> "UnitedDataframe[CK] | None":
        """
        Get a specific segment by its key.
        
        Args:
            segment_key: The key of the segment to retrieve
            
        Returns:
            UnitedDataframe[CK] | None: The segment dataframe if found, None otherwise
        """
        try:
            index = self.categorical_key_values.index(segment_key)
            return self.groupings()[index]
        except ValueError:
            return None
    
    def get_segment_by_index(self, index: int) -> "tuple[tuple[LOWLEVEL_TYPE, ...], UnitedDataframe[CK]] | None":
        """
        Get a specific segment by its index.
        
        Args:
            index: The index of the segment to retrieve
            
        Returns:
            tuple | None: Tuple of (segment_key, segment_dataframe) if index is valid, None otherwise
        """
        if 0 <= index < len(self.categorical_key_values):
            return (self.categorical_key_values[index], self.groupings()[index])
        return None