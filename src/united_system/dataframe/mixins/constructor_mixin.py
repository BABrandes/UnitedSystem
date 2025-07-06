"""
Constructor mixin for UnitedDataframe.

Contains all factory methods and alternative constructors for creating
UnitedDataframe instances from various data sources.
"""

from typing import Generic, TypeVar, Dict, Any, List
import pandas as pd
import numpy as np

from ..column_information import ColumnInformation
from ..column_type import ColumnType, ARRAY_TYPE
from ...unit import Unit
from ...dimension import Dimension

CK = TypeVar("CK", bound=str, default=str)

class ConstructorMixin(Generic[CK]):
    """
    Constructor mixin for UnitedDataframe.
    
    Provides all factory methods and alternative constructors for creating
    UnitedDataframe instances from various data sources.
    """

    # ----------- Factory methods ------------

    @classmethod
    def create_empty(cls, column_keys: List[CK], column_types: List[ColumnType], 
                    display_units: List[Unit] = None, dimensions: List[Dimension] = None) -> "UnitedDataframe[CK]":
        """
        Create an empty UnitedDataframe with specified column structure.
        
        Args:
            column_keys (List[CK]): List of column keys
            column_types (List[ColumnType]): List of column types
            display_units (List[Unit], optional): List of display units
            dimensions (List[Dimension], optional): List of dimensions
            
        Returns:
            UnitedDataframe[CK]: New empty dataframe instance
        """
        if len(column_keys) != len(column_types):
            raise ValueError("Number of column keys must match number of column types")
        
        if display_units is None:
            display_units = [None] * len(column_keys)
        if dimensions is None:
            dimensions = [None] * len(column_keys)
        
        if len(column_keys) != len(display_units) or len(column_keys) != len(dimensions):
            raise ValueError("All lists must have the same length")
        
        # Create empty pandas DataFrame
        data = {}
        for i, (column_key, column_type) in enumerate(zip(column_keys, column_types)):
            data[str(column_key)] = pd.Series([], dtype=column_type.value.corresponding_pandas_type)
        
        df = pd.DataFrame(data)
        
        # Create column information
        column_information = {}
        for i, column_key in enumerate(column_keys):
            column_information[column_key] = ColumnInformation(
                column_key, dimensions[i], column_types[i], display_units[i]
            )
        
        return cls(df, column_information)

    @classmethod
    def create_from_dict(cls, data: Dict[CK, ARRAY_TYPE]) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a dictionary of column data.
        
        Args:
            data (Dict[CK, ARRAY_TYPE]): Dictionary mapping column keys to array data
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        if not data:
            raise ValueError("Data dictionary cannot be empty")
        
        # Convert arrays to pandas Series
        pandas_data = {}
        column_information = {}
        
        for column_key, array_data in data.items():
            # Infer column type from array data
            column_type = ColumnType.infer_from_value(array_data)
            display_unit = getattr(array_data, 'unit', None)
            dimension = getattr(array_data, 'dimension', None)
            
            # Convert to pandas Series
            pandas_data[str(column_key)] = column_type.united_array_to_pandas_series(array_data)
            
            # Create column information
            column_information[column_key] = ColumnInformation(
                column_key, dimension, column_type, display_unit
            )
        
        df = pd.DataFrame(pandas_data)
        return cls(df, column_information)

    @classmethod
    def create_from_records(cls, records: List[Dict[CK, Any]], 
                           column_types: Dict[CK, ColumnType] = None,
                           display_units: Dict[CK, Unit] = None,
                           dimensions: Dict[CK, Dimension] = None) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a list of record dictionaries.
        
        Args:
            records (List[Dict[CK, Any]]): List of dictionaries representing rows
            column_types (Dict[CK, ColumnType], optional): Mapping of column keys to types
            display_units (Dict[CK, Unit], optional): Mapping of column keys to display units
            dimensions (Dict[CK, Dimension], optional): Mapping of column keys to dimensions
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        if not records:
            raise ValueError("Records list cannot be empty")
        
        # Create pandas DataFrame from records
        df = pd.DataFrame(records)
        
        # Create column information
        column_information = {}
        for column_key in df.columns:
            # Use provided types or infer from data
            if column_types and column_key in column_types:
                column_type = column_types[column_key]
            else:
                column_type = ColumnType.infer_from_pandas_dtype(df[column_key].dtype)
            
            display_unit = display_units.get(column_key, None) if display_units else None
            dimension = dimensions.get(column_key, None) if dimensions else None
            
            column_information[column_key] = ColumnInformation(
                column_key, dimension, column_type, display_unit
            )
        
        return cls(df, column_information)

    @classmethod
    def create_from_pandas(cls, df: pd.DataFrame, 
                          column_types: Dict[str, ColumnType] = None,
                          display_units: Dict[str, Unit] = None,
                          dimensions: Dict[str, Dimension] = None) -> "UnitedDataframe[str]":
        """
        Create a UnitedDataframe from a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): The pandas DataFrame to convert
            column_types (Dict[str, ColumnType], optional): Mapping of column names to types
            display_units (Dict[str, Unit], optional): Mapping of column names to display units
            dimensions (Dict[str, Dimension], optional): Mapping of column names to dimensions
            
        Returns:
            UnitedDataframe[str]: New dataframe instance
        """
        # Create column information
        column_information = {}
        for column_name in df.columns:
            # Use provided types or infer from data
            if column_types and column_name in column_types:
                column_type = column_types[column_name]
            else:
                column_type = ColumnType.infer_from_pandas_dtype(df[column_name].dtype)
            
            display_unit = display_units.get(column_name, None) if display_units else None
            dimension = dimensions.get(column_name, None) if dimensions else None
            
            column_information[column_name] = ColumnInformation(
                column_name, dimension, column_type, display_unit
            )
        
        return cls(df.copy(), column_information)

    @classmethod
    def create_from_arrays(cls, arrays: List[ARRAY_TYPE], column_keys: List[CK] = None) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from a list of arrays.
        
        Args:
            arrays (List[ARRAY_TYPE]): List of arrays to use as columns
            column_keys (List[CK], optional): List of column keys. If None, uses numeric indices.
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        if not arrays:
            raise ValueError("Arrays list cannot be empty")
        
        if column_keys is None:
            column_keys = [f"column_{i}" for i in range(len(arrays))]
        
        if len(arrays) != len(column_keys):
            raise ValueError("Number of arrays must match number of column keys")
        
        # Create dictionary mapping column keys to arrays
        data = {column_key: array for column_key, array in zip(column_keys, arrays)}
        
        return cls.create_from_dict(data)

    @classmethod
    def create_like(cls, other: "UnitedDataframe[CK]", data: pd.DataFrame = None) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe with the same structure as another dataframe.
        
        Args:
            other (UnitedDataframe[CK]): The dataframe to copy structure from
            data (pd.DataFrame, optional): The data to use. If None, creates empty dataframe.
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance with same structure
        """
        if data is None:
            # Create empty dataframe with same structure
            return cls.create_empty(
                other.column_keys,
                [other.column_type(ck) for ck in other.column_keys],
                [other.display_unit(ck) for ck in other.column_keys],
                [other.dimension(ck) for ck in other.column_keys]
            )
        else:
            # Use provided data with same column information
            return cls(data, other._column_information, other._internal_dataframe_column_name_formatter)

    @classmethod
    def create_zeros(cls, rows: int, column_keys: List[CK], column_types: List[ColumnType],
                    display_units: List[Unit] = None, dimensions: List[Dimension] = None) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe filled with zeros.
        
        Args:
            rows (int): Number of rows
            column_keys (List[CK]): List of column keys
            column_types (List[ColumnType]): List of column types
            display_units (List[Unit], optional): List of display units
            dimensions (List[Dimension], optional): List of dimensions
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance filled with zeros
        """
        if rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        # Create the basic structure
        df = cls.create_empty(column_keys, column_types, display_units, dimensions)
        
        # Fill with zeros
        for i in range(rows):
            row_data = {}
            for column_key in column_keys:
                # Create zero value for this column type
                column_type = df.column_type(column_key)
                if column_type.is_numeric():
                    row_data[column_key] = 0
                elif column_type == ColumnType.STRING:
                    row_data[column_key] = ""
                elif column_type == ColumnType.BOOL:
                    row_data[column_key] = False
                else:
                    row_data[column_key] = None
            
            df.row_add_values(row_data)
        
        return df

    @classmethod
    def create_ones(cls, rows: int, column_keys: List[CK], column_types: List[ColumnType],
                   display_units: List[Unit] = None, dimensions: List[Dimension] = None) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe filled with ones.
        
        Args:
            rows (int): Number of rows
            column_keys (List[CK]): List of column keys
            column_types (List[ColumnType]): List of column types (must be numeric)
            display_units (List[Unit], optional): List of display units
            dimensions (List[Dimension], optional): List of dimensions
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance filled with ones
        """
        if rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        # Validate that all column types are numeric
        for column_type in column_types:
            if not column_type.is_numeric():
                raise ValueError(f"Column type {column_type} is not numeric")
        
        # Create the basic structure
        df = cls.create_empty(column_keys, column_types, display_units, dimensions)
        
        # Fill with ones
        for i in range(rows):
            row_data = {column_key: 1 for column_key in column_keys}
            df.row_add_values(row_data)
        
        return df

    @classmethod
    def create_random(cls, rows: int, column_keys: List[CK], column_types: List[ColumnType],
                     display_units: List[Unit] = None, dimensions: List[Dimension] = None,
                     random_state: int = None) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe filled with random values.
        
        Args:
            rows (int): Number of rows
            column_keys (List[CK]): List of column keys
            column_types (List[ColumnType]): List of column types
            display_units (List[Unit], optional): List of display units
            dimensions (List[Dimension], optional): List of dimensions
            random_state (int, optional): Random seed for reproducibility
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance filled with random values
        """
        if rows <= 0:
            raise ValueError("Number of rows must be positive")
        
        if random_state is not None:
            np.random.seed(random_state)
        
        # Create the basic structure
        df = cls.create_empty(column_keys, column_types, display_units, dimensions)
        
        # Fill with random values
        for i in range(rows):
            row_data = {}
            for column_key in column_keys:
                column_type = df.column_type(column_key)
                
                if column_type == ColumnType.FLOAT:
                    row_data[column_key] = np.random.random()
                elif column_type == ColumnType.INT:
                    row_data[column_key] = np.random.randint(0, 100)
                elif column_type == ColumnType.BOOL:
                    row_data[column_key] = np.random.choice([True, False])
                elif column_type == ColumnType.STRING:
                    row_data[column_key] = f"random_string_{i}_{np.random.randint(0, 1000)}"
                else:
                    row_data[column_key] = None
            
            df.row_add_values(row_data)
        
        return df 