"""
Constructor operations mixin for UnitedDataframe.

Contains all class factory methods and constructor operations,
including creating empty dataframes, from arrays, and other construction patterns.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Any, Dict, List, Optional, Union, Callable, TYPE_CHECKING
import pandas as pd
from .dataframe_protocol import UnitedDataframeMixin, CK
from ..column_type import ColumnType
from ...units.base_classes.base_dimension import BaseDimension
from ...units.base_classes.base_unit import BaseUnit
from ...units.united import United
from ...arrays.base_classes.base_array import BaseArray

if TYPE_CHECKING:
    from ...united_dataframe import UnitedDataframe

class ConstructorMixin(UnitedDataframeMixin[CK]):
    """
    Constructor operations mixin for UnitedDataframe.
    
    Provides all class factory methods and constructor operations,
    including creating empty dataframes, from arrays, and other construction patterns.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- Class Factory Methods ------------

    @classmethod
    def create_empty(
        cls,
        column_keys: List[CK],
        column_types: Dict[CK, ColumnType],
        dimensions: Dict[CK, BaseDimension],
        display_units: Dict[CK, United],
        internal_dataframe_name_formatter: Callable[[CK], str]
    ) -> "UnitedDataframe":
        """
        Create an empty UnitedDataframe with specified column structure.
        
        Args:
            column_keys (List[CK]): List of column keys
            column_types (Dict[CK, ColumnType]): Column type mapping
            dimensions (Dict[CK, BaseDimension]): Dimension mapping
            display_units (Dict[CK, United]): Display unit mapping
            internal_dataframe_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: Empty dataframe with specified structure
        """
        import pandas as pd
        
        # Create empty pandas dataframe with proper column names
        internal_column_strings = {}
        for column_key in column_keys:
            internal_column_strings[column_key] = internal_dataframe_name_formatter(column_key)
        
        empty_df = pd.DataFrame(columns=list(internal_column_strings.values()))
        
        # Create UnitedDataframe instance
        from ...united_dataframe import UnitedDataframe
        return UnitedDataframe[CK](
            internal_canonical_dataframe=empty_df,
            column_keys=column_keys,
            column_types=column_types,
            dimensions=dimensions,
            display_units=display_units,
            internal_dataframe_column_strings=internal_column_strings,
            internal_dataframe_name_formatter=internal_dataframe_name_formatter,
            read_only=False
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: Dict[CK, BaseArray],
        column_types: Dict[CK, ColumnType],
        dimensions: Dict[CK, BaseDimension],
        display_units: Dict[CK, United],
        internal_dataframe_name_formatter: Callable[[CK], str]
    ) -> "UnitedDataframe":
        """
        Create a UnitedDataframe from a dictionary of arrays.
        
        Args:
            arrays (Dict[CK, BaseArray]): Dictionary mapping column keys to arrays
            column_types (Dict[CK, ColumnType]): Column type mapping
            dimensions (Dict[CK, BaseDimension]): Dimension mapping
            display_units (Dict[CK, United]): Display unit mapping
            internal_dataframe_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: New dataframe with data from arrays
        """
        import pandas as pd
        
        # Validate that all arrays have the same length
        lengths = [len(array) for array in arrays.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All arrays must have the same length.")
        
        # Create internal column strings
        internal_column_strings = {}
        for column_key in arrays.keys():
            internal_column_strings[column_key] = internal_dataframe_name_formatter(column_key)
        
        # Convert arrays to pandas dataframe
        pandas_data = {}
        for column_key, array in arrays.items():
            internal_column_name = internal_column_strings[column_key]
            pandas_data[internal_column_name] = array.to_pandas()
        
        df = pd.DataFrame(pandas_data)
        
        # Create UnitedDataframe instance
        from ...united_dataframe import UnitedDataframe
        return UnitedDataframe[CK](
            internal_canonical_dataframe=df,
            column_keys=list(arrays.keys()),
            column_types=column_types,
            dimensions=dimensions,
            display_units=display_units,
            internal_dataframe_column_strings=internal_column_strings,
            internal_dataframe_name_formatter=internal_dataframe_name_formatter,
            read_only=False
        )

    @classmethod
    def from_dict(
        cls,
        data: Dict[CK, List[Any]],
        column_types: Dict[CK, ColumnType],
        dimensions: Dict[CK, BaseDimension],
        display_units: Dict[CK, United],
        internal_dataframe_name_formatter: Callable[[CK], str]
    ) -> "UnitedDataframe":
        """
        Create a UnitedDataframe from a dictionary of lists.
        
        Args:
            data (Dict[CK, List[Any]]): Dictionary mapping column keys to lists of values
            column_types (Dict[CK, ColumnType]): Column type mapping
            dimensions (Dict[CK, BaseDimension]): Dimension mapping
            display_units (Dict[CK, United]): Display unit mapping
            internal_dataframe_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: New dataframe with data from dictionary
        """
        import pandas as pd
        
        # Validate that all lists have the same length
        lengths = [len(values) for values in data.values()]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError("All lists must have the same length.")
        
        # Create internal column strings
        internal_column_strings = {}
        for column_key in data.keys():
            internal_column_strings[column_key] = internal_dataframe_name_formatter(column_key)
        
        # Convert to pandas dataframe
        pandas_data = {}
        for column_key, values in data.items():
            internal_column_name = internal_column_strings[column_key]
            pandas_data[internal_column_name] = values
        
        df = pd.DataFrame(pandas_data)
        
        # Create UnitedDataframe instance
        from ...united_dataframe import UnitedDataframe
        return UnitedDataframe[CK](
            internal_canonical_dataframe=df,
            column_keys=list(data.keys()),
            column_types=column_types,
            dimensions=dimensions,
            display_units=display_units,
            internal_dataframe_column_strings=internal_column_strings,
            internal_dataframe_name_formatter=internal_dataframe_name_formatter,
            read_only=False
        )

    @classmethod
    def from_pandas(
        cls,
        df: "pd.DataFrame",
        column_key_mapping: Dict[str, CK],
        column_types: Dict[CK, ColumnType],
        dimensions: Dict[CK, BaseDimension],
        display_units: Dict[CK, United],
        internal_dataframe_name_formatter: Callable[[CK], str]
    ) -> "UnitedDataframe":
        """
        Create a UnitedDataframe from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): Source pandas DataFrame
            column_key_mapping (Dict[str, CK]): Mapping from pandas column names to UnitedDataframe column keys
            column_types (Dict[CK, ColumnType]): Column type mapping
            dimensions (Dict[CK, BaseDimension]): Dimension mapping
            display_units (Dict[CK, United]): Display unit mapping
            internal_dataframe_name_formatter (Callable[[CK], str]): Name formatter function
            
        Returns:
            UnitedDataframe: New dataframe with data from pandas DataFrame
        """
        
        # Create internal column strings
        internal_column_strings = {}
        for column_key in column_key_mapping.values():
            internal_column_strings[column_key] = internal_dataframe_name_formatter(column_key)
        
        # Rename columns in the dataframe
        reverse_mapping = {v: k for k, v in column_key_mapping.items()}
        renamed_df = df.copy()
        
        # Rename columns to internal names
        rename_dict = {}
        for pandas_col, united_col_key in column_key_mapping.items():
            internal_name = internal_column_strings[united_col_key]
            rename_dict[pandas_col] = internal_name
        
        renamed_df = renamed_df.rename(columns=rename_dict)
        
        # Create UnitedDataframe instance
        from ...united_dataframe import UnitedDataframe
        return UnitedDataframe[CK](
            internal_canonical_dataframe=renamed_df,
            column_keys=list(column_key_mapping.values()),
            column_types=column_types,
            dimensions=dimensions,
            display_units=display_units,
            internal_dataframe_column_strings=internal_column_strings,
            internal_dataframe_name_formatter=internal_dataframe_name_formatter,
            read_only=False
        )

    # ----------- Copy Operations ------------

    def copy(self) -> "UnitedDataframe":
        """
        Create a deep copy of the dataframe.
        
        Returns:
            UnitedDataframe: Deep copy of the dataframe
        """
        with self._rlock:  # Full IDE support!
            # Create UnitedDataframe instance
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                internal_canonical_dataframe=self._internal_canonical_dataframe.copy(),
                column_keys=self._column_keys.copy(),
                column_types=self._column_types.copy(),
                dimensions=self._dimensions.copy(),
                display_units=self._display_units.copy(),
                internal_dataframe_column_strings=self._internal_dataframe_column_strings.copy(),
                internal_dataframe_name_formatter=self._internal_dataframe_name_formatter,
                read_only=self._read_only
            )

    def copy_structure(self) -> "UnitedDataframe[CK]":
        """
        Create a copy of the dataframe structure (metadata) with empty data.
        
        Returns:
            UnitedDataframe[CK]: New empty dataframe with same structure
        """
        with self._rlock:
            return self.create_empty(
                column_keys=self._column_keys.copy(),
                column_types=self._column_types.copy(),
                dimensions=self._dimensions.copy(),
                display_units=self._display_units.copy(),
                internal_dataframe_name_formatter=self._internal_dataframe_name_formatter
            )

    # ----------- Conversion Operations ------------

    def to_read_only(self) -> "UnitedDataframe[CK]":
        """
        Create a read-only copy of the dataframe.
        
        Returns:
            UnitedDataframe[CK]: Read-only copy of the dataframe
        """
        with self._rlock:
            # Create UnitedDataframe instance
            from ...united_dataframe import UnitedDataframe
            return UnitedDataframe[CK](
                internal_canonical_dataframe=self._internal_canonical_dataframe.copy(),
                column_keys=self._column_keys.copy(),
                column_types=self._column_types.copy(),
                dimensions=self._dimensions.copy(),
                display_units=self._display_units.copy(),
                internal_dataframe_column_strings=self._internal_dataframe_column_strings.copy(),
                internal_dataframe_name_formatter=self._internal_dataframe_name_formatter,
                read_only=True  # Make it read-only
            )

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to pandas DataFrame with original column names.
        
        Returns:
            pd.DataFrame: Pandas DataFrame representation
        """
        with self._rlock:
            # Create a copy and rename columns back to original names
            df = self._internal_canonical_dataframe.copy()
            
            # Create reverse mapping from internal names to column keys
            reverse_mapping = {internal_name: str(column_key) 
                              for column_key, internal_name in self._internal_dataframe_column_strings.items()}
            
            df = df.rename(columns=reverse_mapping)
            return df 