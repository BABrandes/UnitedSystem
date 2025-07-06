"""
Serialization operations mixin for UnitedDataframe.

Contains all operations related to serialization and deserialization,
including JSON, CSV, HDF5, and pickle formats.

Now inherits from UnitedDataframeMixin for full IDE support and type checking.
"""

from typing import Any, Dict, Optional, Union
import json
import pickle
from pathlib import Path
from .dataframe_protocol import UnitedDataframeMixin, CK

class SerializationMixin(UnitedDataframeMixin[CK]):
    """
    Serialization operations mixin for UnitedDataframe.
    
    Provides all functionality related to serialization and deserialization,
    including JSON, CSV, HDF5, and pickle formats.
    
    Now inherits from UnitedDataframeMixin so it has full knowledge of the 
    UnitedDataframe interface with proper IDE support and type checking.
    """

    # ----------- JSON Serialization ------------

    def to_json(self, path: Optional[Union[str, Path]] = None, orient: str = "records", **kwargs) -> Optional[str]:
        """
        Serialize dataframe to JSON format.
        
        Args:
            path (Optional[Union[str, Path]]): File path to save JSON. If None, returns JSON string.
            orient (str): JSON orientation ('records', 'index', 'values', 'columns')
            **kwargs: Additional arguments passed to pandas.DataFrame.to_json()
            
        Returns:
            Optional[str]: JSON string if path is None, otherwise None
        """
        with self._rlock:  # Full IDE support!
            # Convert internal dataframe to JSON
            json_data = self._internal_canonical_dataframe.to_json(orient=orient, **kwargs)
            
            if path is None:
                return json_data
            else:
                # Save to file
                with open(path, 'w') as f:
                    f.write(json_data)
                return None

    def from_json(self, json_data: Union[str, Path], orient: str = "records", **kwargs) -> None:
        """
        Load dataframe from JSON format.
        
        Args:
            json_data (Union[str, Path]): JSON string or file path
            orient (str): JSON orientation ('records', 'index', 'values', 'columns')
            **kwargs: Additional arguments passed to pandas.read_json()
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Load pandas dataframe from JSON
            import pandas as pd
            if isinstance(json_data, (str, Path)) and Path(json_data).exists():
                # Load from file
                df = pd.read_json(json_data, orient=orient, **kwargs)
            else:
                # Assume it's a JSON string
                df = pd.read_json(json_data, orient=orient, **kwargs)
            
            # Replace internal dataframe
            self._internal_canonical_dataframe = df

    # ----------- CSV Serialization ------------

    def to_csv(self, path: Union[str, Path], **kwargs) -> None:
        """
        Serialize dataframe to CSV format.
        
        Args:
            path (Union[str, Path]): File path to save CSV
            **kwargs: Additional arguments passed to pandas.DataFrame.to_csv()
        """
        with self._rlock:
            self._internal_canonical_dataframe.to_csv(path, **kwargs)

    def from_csv(self, path: Union[str, Path], **kwargs) -> None:
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
            df = pd.read_csv(path, **kwargs)
            
            # Replace internal dataframe
            self._internal_canonical_dataframe = df

    # ----------- HDF5 Serialization ------------

    def to_hdf5(self, path: Union[str, Path], key: str, **kwargs) -> None:
        """
        Serialize dataframe to HDF5 format.
        
        Args:
            path (Union[str, Path]): File path to save HDF5
            key (str): HDF5 key/group name
            **kwargs: Additional arguments passed to pandas.DataFrame.to_hdf()
        """
        with self._rlock:
            self._internal_canonical_dataframe.to_hdf(path, key, **kwargs)

    def from_hdf5(self, path: Union[str, Path], key: str, **kwargs) -> None:
        """
        Load dataframe from HDF5 format.
        
        Args:
            path (Union[str, Path]): File path to load HDF5 from
            key (str): HDF5 key/group name
            **kwargs: Additional arguments passed to pandas.read_hdf()
        """
        with self._wlock:  # Full IDE support for _wlock!
            if self._read_only:  # And _read_only!
                raise ValueError("The dataframe is read-only. Please create a new dataframe instead.")
            
            # Load pandas dataframe from HDF5
            import pandas as pd
            df = pd.read_hdf(path, key, **kwargs)
            
            # Replace internal dataframe
            self._internal_canonical_dataframe = df

    # ----------- Pickle Serialization ------------

    def to_pickle(self, path: Union[str, Path]) -> None:
        """
        Serialize entire UnitedDataframe to pickle format.
        
        Args:
            path (Union[str, Path]): File path to save pickle
        """
        with self._rlock:
            # Serialize the entire dataframe object
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    @classmethod
    def from_pickle(cls, path: Union[str, Path]) -> "UnitedDataframe":
        """
        Load UnitedDataframe from pickle format.
        
        Args:
            path (Union[str, Path]): File path to load pickle from
            
        Returns:
            UnitedDataframe: Loaded dataframe
        """
        with open(path, 'rb') as f:
            return pickle.load(f)

    # ----------- Custom Serialization ------------

    def to_dict(self, orient: str = "records") -> Union[Dict[str, Any], list]:
        """
        Convert dataframe to dictionary format.
        
        Args:
            orient (str): Dictionary orientation ('records', 'dict', 'series', 'index')
            
        Returns:
            Union[Dict[str, Any], list]: Dictionary representation
        """
        with self._rlock:
            return self._internal_canonical_dataframe.to_dict(orient=orient)

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
            df = pd.DataFrame.from_dict(data, orient=orient)
            
            # Replace internal dataframe
            self._internal_canonical_dataframe = df

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
                "dimensions": {str(k): str(v) for k, v in self._dimensions.items()},
                "display_units": {str(k): str(v) for k, v in self._display_units.items()},
                "internal_column_strings": self._internal_dataframe_column_strings.copy(),
                "shape": self._internal_canonical_dataframe.shape,
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