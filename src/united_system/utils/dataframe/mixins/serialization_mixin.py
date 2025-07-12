"""
Serialization operations mixin for UnitedDataframe.

Contains all operations related to serialization and deserialization,
including JSON, CSV, HDF5, and pickle formats.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import Any, Dict, Union, TYPE_CHECKING, Type
import json
import pickle
from pathlib import Path
import h5py

from ..internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter, SimpleInternalDataFrameNameFormatter
from .dataframe_protocol import UnitedDataframeProtocol, CK

if TYPE_CHECKING:
    from ....united_dataframe import UnitedDataframe

class SerializationMixin(UnitedDataframeProtocol[CK]):
    """
    Serialization operations mixin for UnitedDataframe.
    
    Provides all functionality related to serialization and deserialization,
    including JSON, CSV, HDF5, and pickle formats.
    
    Now inherits from UnitedDataframeProtocol so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- JSON Serialization ------------

    def to_json(self) -> dict[str, Any]:
        """
        Serialize dataframe to JSON format.
        
        Args:
            **kwargs: Additional arguments passed to pandas.DataFrame.to_json()
            
        Returns:
            Optional[str]: JSON string if path is None, otherwise None
        """
        with self._rlock:  # Full IDE support!
            # Convert internal dataframe to JSON
            json_data: dict[str, Any] = {}
            json_data["dataframe"] = self._internal_dataframe.to_json(orient="records") # type: ignore
            return json_data
        
    @classmethod
    def from_json(cls, data: dict[str, Any], internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SimpleInternalDataFrameNameFormatter(), **_: Any) -> "UnitedDataframe[CK]":

        """
        Load dataframe from JSON format.
        
        Args:
            json_data (Union[str, Path]): JSON string or file path
            orient (str): JSON orientation ('records', 'index', 'values', 'columns')
            internal_dataframe_column_name_formatter (InternalDataFrameNameFormatter): Internal dataframe column name formatter
        """
        import pandas as pd

        df: pd.DataFrame = pd.read_json(data["dataframe"], orient="records") # type: ignore
        united_dataframe: "UnitedDataframe[CK]" = cls.create_dataframe_from_pandas_with_correct_column_names(
            pandas_dataframe=df,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            deep_copy=False
        ) # type: ignore
        return united_dataframe

    # ----------- CSV Serialization ------------

    def to_csv(self, path: str|None = None) -> str|None:
        """
        Serialize dataframe to CSV format.
        
        Args:
            path (str|None): File path to save CSV, or None to return as string
            
        Returns:
            str|None: CSV string if path is None, otherwise None
        """
        with self._rlock:
            if path is None:
                return self._internal_dataframe.to_csv(index=False)
            else:
                self._internal_dataframe.to_csv(path, index=False)
                return None

    def from_csv(self, path: Union[str, Path], **_: Type[Any]) -> None:
        """
        Load dataframe from CSV format.
        
        Args:
            path (Union[str, Path]): File path to load CSV from
            **kwargs: Additional arguments passed to pandas.read_csv()
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Load pandas dataframe from CSV
            import pandas as pd
            df = pd.read_csv(path) # type: ignore
            
            # Replace internal dataframe
            self._internal_canonical_dataframe = df

    # ----------- HDF5 Serialization ------------

    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """
        Serialize dataframe to HDF5 format.
        
        Args:
            path (Union[str, Path]): File path to save HDF5
            key (str): HDF5 key/group name
            **kwargs: Additional arguments passed to pandas.DataFrame.to_hdf()
        """
        with self._rlock:
            self._internal_canonical_dataframe.to_hdf(hdf5_group, "dataframe") # type: ignore

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SimpleInternalDataFrameNameFormatter(), **_: Any) -> "UnitedDataframe[CK]":
        """
        Load dataframe from HDF5 format.
        
        Args:
            path (Union[str, Path]): File path to load HDF5 from
            key (str): HDF5 key/group name
            **kwargs: Additional arguments passed to pandas.read_hdf()
        """
        import pandas as pd
        df: pd.DataFrame = pd.read_hdf(hdf5_group, "dataframe") # type: ignore
        united_dataframe: "UnitedDataframe[CK]" = cls.create_dataframe_from_pandas_with_correct_column_names(
            pandas_dataframe=df,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter,
            deep_copy=False
        ) # type: ignore
        return united_dataframe
    # ----------- Pickle Serialization ------------

    def to_pickle(self, path: str|None = None) -> None:
        """
        Serialize entire UnitedDataframe to pickle format.
        
        Args:
            path (str|None): File path to save pickle, or None for default path
        """
        with self._rlock:
            if path is None:
                path = "united_dataframe.pkl"
            # Serialize the entire dataframe object
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: str|None = None, **kwargs: Any) -> "UnitedDataframe[CK]":
        """
        Load UnitedDataframe from pickle format.
        
        Args:
            path (str|None): File path to load pickle from, or None for default path
            **kwargs: Additional arguments (ignored)
            
        Returns:
            UnitedDataframe: Loaded dataframe
        """
        if path is None:
            path = "united_dataframe.pkl"
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ----------- Custom Serialization ------------

    def to_dict(self, orient: str = "records") -> Union[Dict[str, Any], list[dict[str, Any]]]:
        """
        Convert dataframe to dictionary format.
        
        Args:
            orient (str): Dictionary orientation ('records', 'dict', 'series', 'index')
            
        Returns:
            Union[Dict[str, Any], list]: Dictionary representation
        """
        with self._rlock:
            return self._internal_canonical_dataframe.to_dict(orient=orient) # type: ignore

    def from_dict(self, data: Dict[str, Any], orient: str = "columns") -> None:
        """
        Load dataframe from dictionary format.
        
        Args:
            data (Dict[str, Any]): Dictionary data
            orient (str): Dictionary orientation ('columns', 'records', 'index')
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Create pandas dataframe from dictionary
            import pandas as pd
            df = pd.DataFrame.from_dict(data, orient=orient) # type: ignore
            
            # Replace internal dataframe
            self._internal_canonical_dataframe = df # type: ignore

    # ----------- Metadata Serialization ------------

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the dataframe structure.
        
        Returns:
            Dict[str, Any]: Metadata dictionary
        """
        with self._rlock:
            return {
                "column_keys": self._column_keys.copy(),
                "column_types": {str(k): str(v) for k, v in self._column_types.items()},
                "column_units": {str(k): str(v) for k, v in self._column_units.items()},
                "internal_column_strings": self._internal_dataframe_column_names.copy(),
                "shape": self._internal_dataframe.shape,
                "read_only": self._read_only
            }

    def save_metadata(self, path: Union[str, Path]) -> None:
        """
        Save dataframe metadata to JSON file.
        
        Args:
            path (Union[str, Path]): File path to save metadata
        """
        with self._rlock:
            metadata = self.get_metadata()
            with open(path, 'w') as f:
                json.dump(metadata, f, indent=2) 