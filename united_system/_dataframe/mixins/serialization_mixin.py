"""
Serialization operations mixin for UnitedDataframe.

Contains all operations related to serialization and deserialization,
including JSON, CSV, HDF5, and pickle formats.

Now inherits from UnitedDataframeProtocol for full IDE support and type checking.
"""

from typing import Any, Dict, Union, TYPE_CHECKING, Type, Optional
import json
import pickle
from pathlib import Path
import h5py

from ..column_type import ColumnType
from ..._units_and_dimension.dimension import Dimension
from ..._units_and_dimension.unit import Unit
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
    def from_json(cls, data: dict[str, Any], column_key_type: Type[CK], internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter[CK]=SimpleInternalDataFrameNameFormatter(), **_: Any) -> "UnitedDataframe[CK]":

        """
        Load dataframe from JSON format.
        
        Args:
            json_data (Union[str, Path]): JSON string or file path
            orient (str): JSON orientation ('records', 'index', 'values', 'columns')
            internal_dataframe_column_name_formatter (InternalDataFrameNameFormatter): Internal dataframe column name formatter
        """
        import pandas as pd

        df: pd.DataFrame = pd.read_json(data["dataframe"], orient="records") # type: ignore
        united_dataframe: "UnitedDataframe[CK]" = cls.create_from_dataframe(dataframe=df, column_key_type=column_key_type, internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter) # type: ignore
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

    def to_hdf5(self, group: Union[h5py.Group, Path, tuple[str, h5py.Group]], **kwargs: Any) -> None:
        """
        Serialize dataframe to HDF5 format.

        Args:
            group (Union[h5py.Group, Path, tuple[str, h5py.Group]]): h5py Group to save to. If a tuple is provided, the first element is the key and the second element is the parent h5py Group. If a Path is provided, it is interpreted as a file path and a new file is created.
            **kwargs: Additional arguments passed to pandas.DataFrame.to_hdf()
        """
        with self._rlock:
            key: str = kwargs.pop("key", "dataframe")
            
            # Convert pandas extension arrays to numpy arrays for HDF5 compatibility
            df_for_hdf5 = self._internal_dataframe.copy()
            for col in df_for_hdf5.columns:
                if hasattr(df_for_hdf5[col].dtype, 'numpy_dtype'):
                    # Convert extension dtype to numpy dtype
                    df_for_hdf5[col] = df_for_hdf5[col].astype(df_for_hdf5[col].dtype.numpy_dtype) # type: ignore
            
            if isinstance(group, Path):
                # Use pandas HDF5 interface directly to avoid file locking issues
                df_for_hdf5.to_hdf(str(group), key=key, **kwargs)
                return

            if isinstance(group, tuple):
                group_name, parent = group
                if group_name in parent:
                    raise ValueError(f"The key {group_name} already exists in {parent.name}") #type: ignore
                group = parent.create_group(group_name) #type: ignore

            # For h5py Group, use the file path approach
            file_path: str = group.file.filename
            group_name: str = str(group.name)  #type: ignore
            full_key: str = group_name + "/" + key

            df_for_hdf5.to_hdf(file_path, key=full_key, **kwargs)

    @classmethod
    def from_hdf5(
        cls,
        group: Union[h5py.Group, Path, tuple[str, h5py.Group]],
        column_key_type: Type[CK],
        internal_dataframe_column_name_formatter: InternalDataFrameColumnNameFormatter[CK] = SimpleInternalDataFrameNameFormatter(),
        **kwargs: Any
    ) -> "UnitedDataframe[CK]":
        """
        Load dataframe from HDF5 format.
        
        Args:
            group (Union[h5py.Group, Path, tuple[str, h5py.Group]]): h5py Group to load from. If a tuple is provided, the first element is the key and the second element is the parent h5py Group. If a Path is provided, it is interpreted as a file path and the file is opened.
            internal_dataframe_column_name_formatter: Formatter for internal column names
            **kwargs: Additional arguments passed to pandas.read_hdf()
        """
        import pandas as pd

        key: str = kwargs.pop("key", "dataframe")

        if isinstance(group, Path):
            # Use pandas HDF5 interface directly to avoid file locking issues
            df: pd.DataFrame = pd.read_hdf(str(group), key=key, **kwargs) # type: ignore
            assert isinstance(df, pd.DataFrame)

            # Reconstruct column metadata from the pandas DataFrame
            # The column names should contain the metadata we need
            columns: dict[CK, tuple[ColumnType, str, Optional[Unit|Dimension]]|tuple[ColumnType, str]] = {}
            
            for col_name in df.columns:
                # Parse the internal column name to extract metadata    
                # This assumes the column name formatter creates names that can be parsed back                
                column_key, unit = internal_dataframe_column_name_formatter.retrieve_from_internal_dataframe_column_name(col_name, column_key_type)
                
                # Infer column type from pandas dtype
                column_type: ColumnType = ColumnType.from_dtype(df[col_name].dtype, has_unit=(unit is not None)) # type: ignore
                
                columns[column_key] = (column_type, col_name, unit)
            
            united_dataframe: "UnitedDataframe[CK]" = cls.create_from_dataframe(
                dataframe=df,
                columns=columns,
                column_key_type=column_key_type,
                internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter
            )  # type: ignore
            return united_dataframe

        if isinstance(group, tuple):
            parent_group: h5py.Group = group[1]
            key_string: str = group[0]
            group_ = parent_group[key_string]
            if not isinstance(group_, h5py.Group):
                raise ValueError(f"The key {key_string} is not a h5py Group.")
            group = group_

        # For h5py Group, use the file path approach
        file_path: str = group.file.filename
        group_name: str = str(group.name) #type: ignore
        full_key: str = group_name + "/" + key

        df: pd.DataFrame = pd.read_hdf(file_path, key=full_key, **kwargs) # type: ignore
        assert isinstance(df, pd.DataFrame)

        united_dataframe: "UnitedDataframe[CK]" = cls.create_from_dataframe(# type: ignore
            dataframe=df,
            internal_dataframe_column_name_formatter=internal_dataframe_column_name_formatter
        )  # type: ignore
        return united_dataframe # type: ignore

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