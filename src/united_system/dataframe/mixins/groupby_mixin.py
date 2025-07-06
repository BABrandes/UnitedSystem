"""
Groupby operations mixin for UnitedDataframe.

Contains all operations related to groupby functionality, including
grouping, aggregation, and group-wise operations.
"""

from typing import Generic, TypeVar, Dict, List, Callable, Any
import pandas as pd

from ..column_type import SCALAR_TYPE

CK = TypeVar("CK", bound=str, default=str)

class GroupbyMixin(Generic[CK]):
    """
    Groupby operations mixin for UnitedDataframe.
    
    Provides all functionality related to groupby operations,
    including grouping, aggregation, and group-wise operations.
    """

    # ----------- Groupby operations ------------

    def group_by(self, column_keys: List[CK]) -> "UnitedDataframeGroupBy[CK]":
        """
        Group the dataframe by specified column keys.
        
        Args:
            column_keys (List[CK]): List of column keys to group by
            
        Returns:
            UnitedDataframeGroupBy[CK]: Groupby object for further operations
        """
        with self._rlock:
            # Validate column keys
            for column_key in column_keys:
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            return UnitedDataframeGroupBy(self, column_keys)

    def aggregate(self, agg_dict: Dict[CK, str | List[str] | Callable]) -> "UnitedDataframe[CK]":
        """
        Aggregate the dataframe using specified aggregation functions.
        
        Args:
            agg_dict (Dict[CK, str | List[str] | Callable]): Dictionary mapping column keys to aggregation functions
            
        Returns:
            UnitedDataframe[CK]: New dataframe with aggregated results
        """
        with self._rlock:
            # Validate column keys
            for column_key in agg_dict.keys():
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Convert to internal column names
            internal_agg_dict = {}
            for column_key, agg_func in agg_dict.items():
                internal_column_name = self._internal_dataframe_column_strings[column_key]
                internal_agg_dict[internal_column_name] = agg_func
            
            # Perform aggregation
            agg_result = self._internal_canonical_dataframe.agg(internal_agg_dict)
            
            # Convert back to UnitedDataframe
            if isinstance(agg_result, pd.Series):
                # Single row result
                agg_df = pd.DataFrame([agg_result])
            else:
                agg_df = agg_result
            
            # Create new column information (simplified for aggregated data)
            new_column_information = {}
            for column_key in agg_dict.keys():
                new_column_information[column_key] = self._column_information[column_key]
            
            return UnitedDataframe(agg_df, new_column_information, self._internal_dataframe_column_name_formatter)

    def pivot_table(self, values: CK, index: CK | List[CK], columns: CK | List[CK],
                   aggfunc: str | Callable = "mean", fill_value: Any = None) -> "UnitedDataframe[CK]":
        """
        Create a pivot table from the dataframe.
        
        Args:
            values (CK): Column key to aggregate
            index (CK | List[CK]): Column key(s) to use as row index
            columns (CK | List[CK]): Column key(s) to use as columns
            aggfunc (str | Callable): Aggregation function to use
            fill_value (Any): Value to fill missing combinations
            
        Returns:
            UnitedDataframe[CK]: New dataframe with pivot table
        """
        with self._rlock:
            # Validate column keys
            all_columns = [values]
            if isinstance(index, list):
                all_columns.extend(index)
            else:
                all_columns.append(index)
            if isinstance(columns, list):
                all_columns.extend(columns)
            else:
                all_columns.append(columns)
            
            for column_key in all_columns:
                if column_key not in self._column_keys:
                    raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
            
            # Convert to internal column names
            values_internal = self._internal_dataframe_column_strings[values]
            index_internal = ([self._internal_dataframe_column_strings[idx] for idx in index] 
                            if isinstance(index, list) 
                            else self._internal_dataframe_column_strings[index])
            columns_internal = ([self._internal_dataframe_column_strings[col] for col in columns] 
                              if isinstance(columns, list) 
                              else self._internal_dataframe_column_strings[columns])
            
            # Create pivot table
            pivot_result = self._internal_canonical_dataframe.pivot_table(
                values=values_internal,
                index=index_internal,
                columns=columns_internal,
                aggfunc=aggfunc,
                fill_value=fill_value
            )
            
            # Convert back to UnitedDataframe (simplified column information)
            new_column_information = {}
            for col in pivot_result.columns:
                # Use the values column information for all pivot columns
                new_column_information[str(col)] = self._column_information[values]
            
            return UnitedDataframe(pivot_result, new_column_information, self._internal_dataframe_column_name_formatter)


class UnitedDataframeGroupBy(Generic[CK]):
    """
    GroupBy object for UnitedDataframe.
    
    Provides groupby functionality similar to pandas GroupBy objects.
    """
    
    def __init__(self, dataframe: "UnitedDataframe[CK]", group_keys: List[CK]):
        self._dataframe = dataframe
        self._group_keys = group_keys
        
        # Create internal groupby object
        internal_group_keys = [dataframe._internal_dataframe_column_strings[key] for key in group_keys]
        self._internal_groupby = dataframe._internal_canonical_dataframe.groupby(internal_group_keys)
    
    def aggregate(self, agg_dict: Dict[CK, str | List[str] | Callable]) -> "UnitedDataframe[CK]":
        """
        Aggregate the grouped data.
        
        Args:
            agg_dict (Dict[CK, str | List[str] | Callable]): Dictionary mapping column keys to aggregation functions
            
        Returns:
            UnitedDataframe[CK]: New dataframe with aggregated results
        """
        # Validate column keys
        for column_key in agg_dict.keys():
            if column_key not in self._dataframe._column_keys:
                raise ValueError(f"Column key {column_key} does not exist in the dataframe.")
        
        # Convert to internal column names
        internal_agg_dict = {}
        for column_key, agg_func in agg_dict.items():
            internal_column_name = self._dataframe._internal_dataframe_column_strings[column_key]
            internal_agg_dict[internal_column_name] = agg_func
        
        # Perform aggregation
        agg_result = self._internal_groupby.agg(internal_agg_dict)
        
        # Create new column information
        new_column_information = {}
        
        # Add group keys to column information
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        # Add aggregated columns to column information
        for column_key in agg_dict.keys():
            new_column_information[column_key] = self._dataframe._column_information[column_key]
        
        return UnitedDataframe(agg_result.reset_index(), new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter)
    
    def sum(self, numeric_only: bool = True) -> "UnitedDataframe[CK]":
        """
        Calculate sum for each group.
        
        Args:
            numeric_only (bool): If True, only aggregate numeric columns
            
        Returns:
            UnitedDataframe[CK]: New dataframe with group sums
        """
        sum_result = self._internal_groupby.sum(numeric_only=numeric_only)
        
        # Create new column information
        new_column_information = {}
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        for col in sum_result.columns:
            # Find the corresponding column key
            for column_key in self._dataframe._column_keys:
                if self._dataframe._internal_dataframe_column_strings[column_key] == col:
                    new_column_information[column_key] = self._dataframe._column_information[column_key]
                    break
        
        return UnitedDataframe(sum_result.reset_index(), new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter)
    
    def mean(self, numeric_only: bool = True) -> "UnitedDataframe[CK]":
        """
        Calculate mean for each group.
        
        Args:
            numeric_only (bool): If True, only aggregate numeric columns
            
        Returns:
            UnitedDataframe[CK]: New dataframe with group means
        """
        mean_result = self._internal_groupby.mean(numeric_only=numeric_only)
        
        # Create new column information (similar to sum)
        new_column_information = {}
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        for col in mean_result.columns:
            for column_key in self._dataframe._column_keys:
                if self._dataframe._internal_dataframe_column_strings[column_key] == col:
                    new_column_information[column_key] = self._dataframe._column_information[column_key]
                    break
        
        return UnitedDataframe(mean_result.reset_index(), new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter)
    
    def count(self) -> "UnitedDataframe[CK]":
        """
        Count non-null values for each group.
        
        Returns:
            UnitedDataframe[CK]: New dataframe with group counts
        """
        count_result = self._internal_groupby.count()
        
        # Create new column information
        new_column_information = {}
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        for col in count_result.columns:
            for column_key in self._dataframe._column_keys:
                if self._dataframe._internal_dataframe_column_strings[column_key] == col:
                    # Count results are always integers
                    new_column_information[column_key] = self._dataframe._column_information[column_key]
                    break
        
        return UnitedDataframe(count_result.reset_index(), new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter)
    
    def size(self) -> "UnitedDataframe[CK]":
        """
        Get the size of each group.
        
        Returns:
            UnitedDataframe[CK]: New dataframe with group sizes
        """
        size_result = self._internal_groupby.size().reset_index(name='size')
        
        # Create new column information
        new_column_information = {}
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        # Add size column
        new_column_information['size'] = ColumnInformation('size', None, ColumnType.INT, None)
        
        return UnitedDataframe(size_result, new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter)
    
    def min(self, numeric_only: bool = True) -> "UnitedDataframe[CK]":
        """
        Calculate minimum for each group.
        
        Args:
            numeric_only (bool): If True, only aggregate numeric columns
            
        Returns:
            UnitedDataframe[CK]: New dataframe with group minimums
        """
        min_result = self._internal_groupby.min(numeric_only=numeric_only)
        
        # Create new column information (similar to sum)
        new_column_information = {}
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        for col in min_result.columns:
            for column_key in self._dataframe._column_keys:
                if self._dataframe._internal_dataframe_column_strings[column_key] == col:
                    new_column_information[column_key] = self._dataframe._column_information[column_key]
                    break
        
        return UnitedDataframe(min_result.reset_index(), new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter)
    
    def max(self, numeric_only: bool = True) -> "UnitedDataframe[CK]":
        """
        Calculate maximum for each group.
        
        Args:
            numeric_only (bool): If True, only aggregate numeric columns
            
        Returns:
            UnitedDataframe[CK]: New dataframe with group maximums
        """
        max_result = self._internal_groupby.max(numeric_only=numeric_only)
        
        # Create new column information (similar to sum)
        new_column_information = {}
        for group_key in self._group_keys:
            new_column_information[group_key] = self._dataframe._column_information[group_key]
        
        for col in max_result.columns:
            for column_key in self._dataframe._column_keys:
                if self._dataframe._internal_dataframe_column_strings[column_key] == col:
                    new_column_information[column_key] = self._dataframe._column_information[column_key]
                    break
        
        return UnitedDataframe(max_result.reset_index(), new_column_information, 
                             self._dataframe._internal_dataframe_column_name_formatter) 