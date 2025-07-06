"""
Serialization mixin for UnitedDataframe.

Contains all operations related to serialization and deserialization,
including JSON and HDF5 format support.
"""

from typing import Generic, TypeVar, Any, Dict
import json
import pandas as pd

CK = TypeVar("CK", bound=str, default=str)

class SerializationMixin(Generic[CK]):
    """
    Serialization mixin for UnitedDataframe.
    
    Provides all functionality related to serialization and deserialization,
    including JSON and HDF5 format support.
    """

    # ----------- JSON serialization ------------

    def to_json(self, orient: str = "records", date_format: str = "iso", **kwargs) -> str:
        """
        Convert the dataframe to JSON format.
        
        Args:
            orient (str): Format of the JSON string:
                         - "records": list of dictionaries
                         - "index": dictionary with index as keys
                         - "values": list of lists
                         - "columns": dictionary with column names as keys
            date_format (str): Date format ("iso", "epoch")
            **kwargs: Additional arguments passed to pandas.to_json()
            
        Returns:
            str: JSON string representation of the dataframe
        """
        with self._rlock:
            # Convert to pandas DataFrame for JSON serialization
            df_for_json = self._internal_canonical_dataframe.copy()
            
            # Add metadata about units and dimensions
            metadata = {
                "column_keys": [self.column_key_as_str(ck) for ck in self._column_keys],
                "column_types": {self.column_key_as_str(ck): str(self._column_types[ck]) for ck in self._column_keys},
                "display_units": {self.column_key_as_str(ck): str(self._display_units[ck]) for ck in self._column_keys},
                "dimensions": {self.column_key_as_str(ck): str(self._dimensions[ck]) for ck in self._column_keys}
            }
            
            # Create the final JSON structure
            json_data = {
                "data": json.loads(df_for_json.to_json(orient=orient, date_format=date_format, **kwargs)),
                "metadata": metadata
            }
            
            return json.dumps(json_data, indent=2)

    @classmethod
    def from_json(cls, json_str: str, **kwargs) -> "UnitedDataframe[CK]":
        """
        Create a UnitedDataframe from JSON string.
        
        Args:
            json_str (str): JSON string representation of the dataframe
            **kwargs: Additional arguments passed to pandas.read_json()
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        json_data = json.loads(json_str)
        
        # Extract data and metadata
        data = json_data["data"]
        metadata = json_data["metadata"]
        
        # Create pandas DataFrame from data
        df = pd.DataFrame(data)
        
        # Reconstruct column information
        column_information = {}
        for col_key_str in metadata["column_keys"]:
            column_key = col_key_str  # Assume string column keys for now
            column_type = ColumnType.from_string(metadata["column_types"][col_key_str])
            display_unit = Unit.from_string(metadata["display_units"][col_key_str])
            dimension = Dimension.from_string(metadata["dimensions"][col_key_str])
            
            column_information[column_key] = ColumnInformation(
                column_key, dimension, column_type, display_unit
            )
        
        # Create the dataframe
        return cls(df, column_information)

    # ----------- HDF5 serialization ------------

    def to_hdf5(self, filepath: str, key: str = "dataframe", **kwargs) -> None:
        """
        Save the dataframe to HDF5 format.
        
        Args:
            filepath (str): Path to the HDF5 file
            key (str): Key under which to store the dataframe in the HDF5 file
            **kwargs: Additional arguments passed to pandas.to_hdf()
        """
        with self._rlock:
            # Save the main dataframe
            self._internal_canonical_dataframe.to_hdf(filepath, key=key, **kwargs)
            
            # Save metadata
            metadata = {
                "column_keys": [self.column_key_as_str(ck) for ck in self._column_keys],
                "column_types": {self.column_key_as_str(ck): str(self._column_types[ck]) for ck in self._column_keys},
                "display_units": {self.column_key_as_str(ck): str(self._display_units[ck]) for ck in self._column_keys},
                "dimensions": {self.column_key_as_str(ck): str(self._dimensions[ck]) for ck in self._column_keys}
            }
            
            # Save metadata as a separate key
            metadata_df = pd.DataFrame([metadata])
            metadata_df.to_hdf(filepath, key=f"{key}_metadata", mode="a")

    @classmethod
    def from_hdf5(cls, filepath: str, key: str = "dataframe", **kwargs) -> "UnitedDataframe[CK]":
        """
        Load a UnitedDataframe from HDF5 format.
        
        Args:
            filepath (str): Path to the HDF5 file
            key (str): Key under which the dataframe is stored in the HDF5 file
            **kwargs: Additional arguments passed to pandas.read_hdf()
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        # Load the main dataframe
        df = pd.read_hdf(filepath, key=key, **kwargs)
        
        # Load metadata
        metadata_df = pd.read_hdf(filepath, key=f"{key}_metadata")
        metadata = metadata_df.iloc[0].to_dict()
        
        # Reconstruct column information
        column_information = {}
        for col_key_str in metadata["column_keys"]:
            column_key = col_key_str  # Assume string column keys for now
            column_type = ColumnType.from_string(metadata["column_types"][col_key_str])
            display_unit = Unit.from_string(metadata["display_units"][col_key_str])
            dimension = Dimension.from_string(metadata["dimensions"][col_key_str])
            
            column_information[column_key] = ColumnInformation(
                column_key, dimension, column_type, display_unit
            )
        
        # Create the dataframe
        return cls(df, column_information)

    # ----------- CSV serialization ------------

    def to_csv(self, filepath: str, include_metadata: bool = True, **kwargs) -> None:
        """
        Save the dataframe to CSV format.
        
        Args:
            filepath (str): Path to the CSV file
            include_metadata (bool): Whether to include metadata in a separate file
            **kwargs: Additional arguments passed to pandas.to_csv()
        """
        with self._rlock:
            # Save the main dataframe
            self._internal_canonical_dataframe.to_csv(filepath, **kwargs)
            
            if include_metadata:
                # Save metadata to a separate JSON file
                metadata_filepath = filepath.replace(".csv", "_metadata.json")
                metadata = {
                    "column_keys": [self.column_key_as_str(ck) for ck in self._column_keys],
                    "column_types": {self.column_key_as_str(ck): str(self._column_types[ck]) for ck in self._column_keys},
                    "display_units": {self.column_key_as_str(ck): str(self._display_units[ck]) for ck in self._column_keys},
                    "dimensions": {self.column_key_as_str(ck): str(self._dimensions[ck]) for ck in self._column_keys}
                }
                
                with open(metadata_filepath, 'w') as f:
                    json.dump(metadata, f, indent=2)

    @classmethod
    def from_csv(cls, filepath: str, load_metadata: bool = True, **kwargs) -> "UnitedDataframe[CK]":
        """
        Load a UnitedDataframe from CSV format.
        
        Args:
            filepath (str): Path to the CSV file
            load_metadata (bool): Whether to load metadata from a separate file
            **kwargs: Additional arguments passed to pandas.read_csv()
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        # Load the main dataframe
        df = pd.read_csv(filepath, **kwargs)
        
        if load_metadata:
            # Load metadata from JSON file
            metadata_filepath = filepath.replace(".csv", "_metadata.json")
            try:
                with open(metadata_filepath, 'r') as f:
                    metadata = json.load(f)
                
                # Reconstruct column information
                column_information = {}
                for col_key_str in metadata["column_keys"]:
                    column_key = col_key_str  # Assume string column keys for now
                    column_type = ColumnType.from_string(metadata["column_types"][col_key_str])
                    display_unit = Unit.from_string(metadata["display_units"][col_key_str])
                    dimension = Dimension.from_string(metadata["dimensions"][col_key_str])
                    
                    column_information[column_key] = ColumnInformation(
                        column_key, dimension, column_type, display_unit
                    )
                
                # Create the dataframe
                return cls(df, column_information)
            except FileNotFoundError:
                # If metadata file doesn't exist, create basic column information
                column_information = {}
                for col in df.columns:
                    column_information[col] = ColumnInformation(
                        col, None, ColumnType.infer_from_pandas_dtype(df[col].dtype), None
                    )
                return cls(df, column_information)
        else:
            # Create basic column information without metadata
            column_information = {}
            for col in df.columns:
                column_information[col] = ColumnInformation(
                    col, None, ColumnType.infer_from_pandas_dtype(df[col].dtype), None
                )
            return cls(df, column_information)

    # ----------- Pickle serialization ------------

    def to_pickle(self, filepath: str) -> None:
        """
        Save the dataframe to pickle format.
        
        Args:
            filepath (str): Path to the pickle file
        """
        with self._rlock:
            import pickle
            
            # Create a dictionary with all necessary data
            data_to_pickle = {
                "internal_canonical_dataframe": self._internal_canonical_dataframe,
                "column_information": self._column_information,
                "internal_dataframe_column_name_formatter": self._internal_dataframe_column_name_formatter
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_pickle, f)

    @classmethod
    def from_pickle(cls, filepath: str) -> "UnitedDataframe[CK]":
        """
        Load a UnitedDataframe from pickle format.
        
        Args:
            filepath (str): Path to the pickle file
            
        Returns:
            UnitedDataframe[CK]: New dataframe instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create the dataframe
        return cls(
            data["internal_canonical_dataframe"],
            data["column_information"],
            data["internal_dataframe_column_name_formatter"]
        ) 