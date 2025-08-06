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
    from ..._dataframe.united_dataframe import UnitedDataframe

class SerializationMixin(UnitedDataframeProtocol[CK, "UnitedDataframe[CK]"]):
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
        united_dataframe: "UnitedDataframe[CK]" = cls.create_from_dataframe(dataframe=df) # type: ignore
        return united_dataframe  # type: ignore

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
            self._internal_dataframe = df

    # ----------- HDF5 Serialization ------------

    def to_hdf5(self, group: Union[h5py.Group, tuple[str, h5py.Group]], **kwargs: Any) -> None:
        """
        Serialize dataframe to HDF5 format.

        Args:
            group (Union[h5py.Group, tuple[str, h5py.Group]]): h5py Group to save to. If a tuple is provided, the first element is the key and the second element is the parent h5py Group.
            **kwargs: Additional arguments passed to pandas.DataFrame.to_hdf()
        """
        with self._rlock:
            if isinstance(group, tuple):
                group_name, parent = group
                if group_name in parent:
                    raise ValueError(f"The key {group_name} already exists in {parent.name}") #type: ignore
                group = parent.create_group(group_name) #type: ignore

            file_path: str = group.file.filename
            key: str = kwargs.pop("key", "dataframe")
            group_name: str = str(group.name)  #type: ignore
            full_key: str = group_name + "/" + key

            self._internal_dataframe.to_hdf(file_path, key=full_key, **kwargs)

    @classmethod
    def from_hdf5(
        cls,
        group: Union[h5py.Group, tuple[str, h5py.Group]],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter = SimpleInternalDataFrameNameFormatter(),
        **kwargs: Any
    ) -> "UnitedDataframe[CK]":
        """
        Load dataframe from HDF5 format.
        
        Args:
            group (Union[h5py.Group, tuple[str, h5py.Group]]): h5py Group to load from. If a tuple is provided, the first element is the key and the second element is the parent h5py Group.
            internal_dataframe_column_name_formatter: Formatter for internal column names
            **kwargs: Additional arguments passed to pandas.read_hdf()
        """
        import pandas as pd

        if isinstance(group, tuple):
            parent_group: h5py.Group = group[1]
            key_string: str = group[0]
            group_ = parent_group[key_string]
            if not isinstance(group_, h5py.Group):
                raise ValueError(f"The key {key_string} is not a h5py Group.")
            group = group_

        file_path: str = group.file.filename
        key: str = kwargs.pop("key", "dataframe")
        group_name: str = str(group.name) #type: ignore
        full_key: str = group_name + "/" + key

        df: pd.DataFrame = pd.read_hdf(file_path, key=full_key, **kwargs) # type: ignore
        assert isinstance(df, pd.DataFrame)

        united_dataframe: "UnitedDataframe[CK]" = cls.create_from_dataframe(
            dataframe=df,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter
        )  # type: ignore
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
            return self._internal_dataframe.to_dict(orient=orient) # type: ignore

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
            self._internal_dataframe = df # type: ignore

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

    # ----------- Pickle State Management ------------

    def __getstate__(self) -> Dict[str, Any]:
        """
        Custom pickle state management - exclude thread locks.
        
        Thread locks cannot be pickled, so we exclude them from the state
        and recreate them during unpickling.
        
        Returns:
            Dict[str, Any]: Picklable state dictionary
        """
        # Get all instance attributes except locks
        state = self.__dict__.copy()
        
        # Remove the unpicklable thread lock objects
        state.pop('_lock', None)
        state.pop('_rlock', None) 
        state.pop('_wlock', None)
        
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Custom pickle state restoration - recreate thread locks.
        
        Restores the pickled state and recreates the thread locks that
        were excluded during pickling.
        
        Args:
            state (Dict[str, Any]): Restored state dictionary
        """
        # Restore all attributes from the pickled state
        self.__dict__.update(state)
        
        # Recreate the thread locks
        from readerwriterlock import rwlock
        self._lock = rwlock.RWLockFairD()
        self._rlock = self._lock.gen_rlock()
        self._wlock = self._lock.gen_wlock() 