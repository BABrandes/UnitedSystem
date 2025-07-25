

from typing import Callable, TypeVar, Tuple, TYPE_CHECKING, Union
from collections.abc import Sequence
from bidict import bidict

from ....column_key import ColumnKey
from ..accessors._row_accessor import RowAccessor
from ._base_grouping import BaseGrouping, GroupingContainer
from ....column_type import SCALAR_TYPE, LOWLEVEL_TYPE
import pandas as pd

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

CK = TypeVar("CK", bound="ColumnKey|str")

class Groups(BaseGrouping[CK]):
    """
    A Groups object for performing grouped operations on United_Dataframe.
    
    This class provides pandas-like groupby functionality with unit awareness.
    Groups data by unique combinations of column values.
    """

######################### Initialization #########################
    
    def __init__(
            self,
            dataframe: "UnitedDataframe[CK]",
            by_unique_values_of_columns: Sequence[CK] = [],
            by_unique_results_of_row_functions: Sequence[Tuple[CK, Callable[["RowAccessor[CK]"], SCALAR_TYPE]]] = []):
        """
        Initialize a Groups object.
        
        Args:
            dataframe: The United_Dataframe to group
            by_unique_values_of_columns: List of columns to group by
            by_unique_results_of_row_functions: List of tuples of column keys for the results of the row functions and functions to apply to each row to get a unique result for each group
        """
        super().__init__(dataframe, by_unique_values_of_columns, by_unique_results_of_row_functions)
        
        # Create groups
        self._create_groups()

    def _create_groups(self) -> None:
        """Create group dataframes based on unique combinations."""
        # Get categorical column names and create group keys in one pass
        categorical_column_names = [col_info.internal_dataframe_column_name for col_info in self._categorical_column_information.values()]
        categorical_column_keys = list(self._categorical_column_information.keys())
        
        # Create group keys (combinations of values) directly from the dataframe
        group_data = self._working_df[categorical_column_names].values # type: ignore
        all_group_keys = [tuple(row) for row in group_data]
        unique_group_keys = list(set(all_group_keys))

        internal_dataframe_column_names = bidict({col: col_info.internal_dataframe_column_name for col, col_info in self._available_column_information.items()})
        available_column_keys = list(self._available_column_information.keys())
        available_column_types = {col: col_info.column_type for col, col_info in self._available_column_information.items()}
        available_column_units = {col: col_info.column_unit for col, col_info in self._available_column_information.items()}

        # Create group dataframes
        for group_key in unique_group_keys:
            # Create mask for this group
            mask = pd.Series([gk == group_key for gk in all_group_keys], index=self._working_df.index) # type: ignore
            group_df: pd.DataFrame = self._working_df[mask].copy() # type: ignore

            # Create GroupingContainer with all required information
            self._grouping_containers.append(GroupingContainer(
                parent_united_dataframe=self._dataframe,
                dataframe=group_df,
                internal_dataframe_column_names=internal_dataframe_column_names,
                available_column_keys=available_column_keys,
                available_column_types=available_column_types,
                available_column_units=available_column_units,
                categorical_column_keys=categorical_column_keys,
                categorical_key_values=group_key
            ))
    
######################### Properties #########################
    
    @property
    def groups(self) -> list["UnitedDataframe[CK]"]:
        """
        Get the grouped dataframes.
        
        Returns:
            list: List of United_Dataframe instances for each group
        """
        return self.groupings()
    
    @property
    def group_keys(self) -> list[tuple[LOWLEVEL_TYPE, ...]]:
        """
        Get the group keys.
        
        Returns:
            list: List of group key tuples
        """
        return self.categorical_key_values
    
    def get_group_by_key(self, group_key: tuple[LOWLEVEL_TYPE, ...]) -> Union["UnitedDataframe[CK]", None]:
        """
        Get a specific group by its key.
        
        Args:
            group_key: The key of the group to retrieve
            
        Returns:
            UnitedDataframe[CK] | None: The group dataframe if found, None otherwise
        """
        try:
            index = self.categorical_key_values.index(group_key)
            return self.groupings()[index]
        except ValueError:
            return None
    
    def get_group_by_index(self, index: int) -> tuple[tuple[LOWLEVEL_TYPE, ...], "UnitedDataframe[CK]"] | None:
        """
        Get a specific group by its index.
        
        Args:
            index: The index of the group to retrieve
            
        Returns:
            tuple | None: Tuple of (group_key, group_dataframe) if index is valid, None otherwise
        """
        if 0 <= index < len(self.categorical_key_values):
            return (self.categorical_key_values[index], self.groupings()[index])
        return None