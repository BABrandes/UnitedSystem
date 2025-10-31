from typing import Any, Optional, Literal, TYPE_CHECKING
import pickle
import csv
from io import StringIO

from ...._utils.general import SerializationProtocol

# Optional YAML support
try:
    import yaml as _yaml_module  # type: ignore[import-untyped]
    _yaml_available = True
except ImportError:
    _yaml_available = False
    _yaml_module = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ...._units_and_dimension.dimension.dimension import Dimension


class SerializationMixin(SerializationProtocol["Dimension"]):
    """Serialization mixin for Dimension."""

    def serialize(self, format: Optional[Literal["json", "hdf5", "pickle", "csv", "yaml"]], **kwargs: Any) -> Any:
        """Serialize the dimension to a JSON, HDF5, pickle, CSV, or YAML string."""
        if format == "json":
            return self._to_json()
        elif format == "hdf5":
            return self._to_hdf5(kwargs["hdf5_group"])
        elif format == "pickle":
            return self._to_pickle()
        elif format == "csv":
            return self._to_csv()
        elif format == "yaml":
            return self._to_yaml()
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    @classmethod
    def deserialize(cls, data: Any, format: Optional[Literal["json", "hdf5", "pickle", "csv", "yaml"]], **kwargs: Any) -> "Dimension":
        """Deserialize the dimension from a JSON, HDF5, pickle, CSV, or YAML string."""
        if format == "json":
            return cls._from_json(data)
        elif format == "hdf5":
            return cls._from_hdf5(data)
        elif format == "pickle":
            return cls._from_pickle(data)
        elif format == "csv":
            return cls._from_csv(data)
        elif format == "yaml":
            return cls._from_yaml(data)
        else:
            raise ValueError(f"Unsupported serialization format: {format}")

    ################################################################################
    # JSON serialization
    ################################################################################

    def _to_json(self) -> str:
        """
        Convert the dimension to a JSON string representation.
        
        Returns:
            JSON string representation of the dimension
        
        Examples:
            force = Dimension("M*L/T^2")
            json_str = force.to_json()  # "M*L/T^2"
        """

        _self: "Dimension" = self # type: ignore

        return _self.format_string()

    @classmethod
    def _from_json(cls, json_string: str) -> "Dimension":
        """
        Create a dimension from a JSON string representation.
        
        Args:
            json_string: JSON string representation of the dimension
        
        Returns:
            A new Dimension object
        
        Examples:
            json_str = "M*L/T^2"
            force = Dimension.from_json(json_str)
        """
        return Dimension._parse_string(json_string) # type: ignore
    
################################################################################
# HDF5 serialization
################################################################################

    def _to_hdf5(self, hdf5_group: Any) -> None:
        """
        Save the dimension to an HDF5 group.
        
        Args:
            hdf5_group: HDF5 group to save the dimension to
        
        Examples:
            import h5py
            
            with h5py.File('data.h5', 'w') as f:
                force = Dimension("M*L/T^2")
                force.to_hdf5(f)
        """

        _self: "Dimension" = self # type: ignore

        hdf5_group.attrs["dimension"] = _self.format_string()

    @classmethod
    def _from_hdf5(cls, hdf5_group: Any) -> "Dimension":
        """
        Load a dimension from an HDF5 group.
        
        Args:
            hdf5_group: HDF5 group containing the dimension
        
        Returns:
            A new Dimension object
        
        Examples:
            import h5py
            
            with h5py.File('data.h5', 'r') as f:
                force = Dimension.from_hdf5(f)
        """

        from h5py import Group

        _hdf5_group: Group = hdf5_group # type: ignore

        return Dimension._parse_string(_hdf5_group.attrs["dimension"]) # type: ignore

    ################################################################################
    # Pickle serialization
    ################################################################################

    def _to_pickle(self) -> bytes:
        """
        Serialize the dimension to pickle format.
        
        Returns:
            Pickle bytes representation of the dimension
            
        Examples:
            force = Dimension("M*L/T^2")
            pickle_data = force._to_pickle()
        """
        _self: "Dimension" = self # type: ignore
        
        # Serialize using the JSON string structure for consistency
        json_string = _self._to_json()
        return pickle.dumps(json_string)
    
    @classmethod
    def _from_pickle(cls, data: bytes) -> "Dimension":
        """
        Deserialize the dimension from pickle format.
        
        Args:
            data: Pickle bytes representation of the dimension
        
        Returns:
            A new Dimension object
            
        Examples:
            pickle_data = force._to_pickle()
            force_restored = Dimension._from_pickle(pickle_data)
        """
        # Deserialize the JSON string structure
        json_string: str = pickle.loads(data)
        return cls._from_json(json_string)

    ################################################################################
    # CSV serialization
    ################################################################################

    def _to_csv(self) -> str:
        """
        Serialize the dimension to CSV format.
        
        Returns:
            CSV string representation of the dimension
            
        Examples:
            force = Dimension("M*L/T^2")
            csv_str = force._to_csv()
        """
        _self: "Dimension" = self # type: ignore
        
        output = StringIO()
        writer = csv.writer(output)
        # Write header
        writer.writerow(["dimension"])
        # Write data row
        writer.writerow([_self.format_string()])
        return output.getvalue()
    
    @classmethod
    def _from_csv(cls, data: str) -> "Dimension":
        """
        Deserialize the dimension from CSV format.
        
        Args:
            data: CSV string representation of the dimension
        
        Returns:
            A new Dimension object
            
        Examples:
            csv_str = force._to_csv()
            force_restored = Dimension._from_csv(csv_str)
        """
        reader = csv.reader(StringIO(data))
        # Skip header row
        next(reader)
        # Read data row
        row = next(reader)
        dimension_string = row[0]
        return cls._from_json(dimension_string)

    ################################################################################
    # YAML serialization
    ################################################################################

    def _to_yaml(self) -> str:
        """
        Serialize the dimension to YAML format.
        
        Returns:
            YAML string representation of the dimension
            
        Examples:
            force = Dimension("M*L/T^2")
            yaml_str = force._to_yaml()
        
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not _yaml_available or _yaml_module is None:
            raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")
        
        _self: "Dimension" = self # type: ignore
        
        # Use the JSON string structure and convert to YAML
        json_string = _self._to_json()
        return _yaml_module.dump(json_string, default_flow_style=False)
    
    @classmethod
    def _from_yaml(cls, data: str) -> "Dimension":
        """
        Deserialize the dimension from YAML format.
        
        Args:
            data: YAML string representation of the dimension
        
        Returns:
            A new Dimension object
            
        Examples:
            yaml_str = force._to_yaml()
            force_restored = Dimension._from_yaml(yaml_str)
        
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not _yaml_available or _yaml_module is None:
            raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")
        
        # Load YAML and use JSON deserialization
        json_string: str = _yaml_module.safe_load(data)
        return cls._from_json(json_string)