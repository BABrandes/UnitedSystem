from typing import TYPE_CHECKING, Optional, Literal, Any
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
    from ...._units_and_dimension.unit.unit import Unit


class SerializationMixin(SerializationProtocol["Unit"]):

    def serialize(self, format: Optional[Literal["json", "hdf5", "pickle", "csv", "yaml"]], **kwargs: Any) -> Any:
        """Serialize the unit to a JSON, HDF5, pickle, CSV, or YAML string."""
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
    def deserialize(cls, data: Any, format: Optional[Literal["json", "hdf5", "pickle", "csv", "yaml"]], **kwargs: Any) -> "Unit":
        """Deserialize the unit from a JSON, HDF5, pickle, CSV, or YAML string."""
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

    #####################################################################################
    # JSON Serialization
    #####################################################################################
    
    def _to_json(self) -> str:
        """
        Convert the unit to JSON string representation.
        
        Returns:
            JSON string representation
        
        Examples:
            Unit("m").to_json() -> '"m"'
        """

        _self: "Unit" = self # type: ignore

        return _self.format_string()
    
    @classmethod
    def _from_json(cls, json_string: str) -> "Unit":
        """
        Create a unit from JSON string representation.
        
        Args:
            json_string: The JSON string representation
        
        Returns:
            A new Unit instance
        
        Examples:
            Unit.from_json('"m"') -> Unit("m")
        """
        return cls(json_string) # type: ignore

    #####################################################################################
    # HDF5 Serialization
    #####################################################################################

    def _to_hdf5(self, hdf5_group: Any) -> None:
        """
        Save the unit to an HDF5 group.
        
        Args:
            hdf5_group: The HDF5 group to save to
        """
        _self: "Unit" = self # type: ignore
        hdf5_group["unit"] = _self.format_string()

    @classmethod
    def _from_hdf5(cls, hdf5_group: Any) -> "Unit":
        """
        Create a unit from an HDF5 group.
        
        Args:
            hdf5_group: The HDF5 group to read from
            
        Returns:
            A new Unit instance
        """

        from h5py import Group

        _hdf5_group: Group = hdf5_group # type: ignore
        unit_string: str = _hdf5_group["unit"].asstr()[()] # type: ignore
        if isinstance(unit_string, bytes):
            unit_string = unit_string.decode("utf-8")
        return cls(unit_string) # type: ignore
    
    #####################################################################################
    # Pickle Serialization
    #####################################################################################
        
    def __getstate__(self) -> dict[str, Any]:
        """Custom pickle state management for slotted dataclass."""

        _self: "Unit" = self # type: ignore

        # Ensure _dimension is computed before pickling
        _ = _self.dimension  # This will initialize _dimension if not already set
        
        # For slotted dataclasses, manually collect field values
        # Convert MappingProxyType to dict for pickling
        return {
            "_unit_elements": dict(_self._unit_elements),  # Convert MappingProxyType to dict # type: ignore
            "_log_units": _self._log_units, # type: ignore
            "_dimension": _self._dimension # type: ignore
        }
    
    def __setstate__(self, state: dict[str, Any]) -> None:
        """Custom pickle state restoration for slotted dataclass."""

        _self: "Unit" = self # type: ignore

        # Convert dict back to MappingProxyType
        from types import MappingProxyType
        if "_unit_elements" in state:
            state["_unit_elements"] = MappingProxyType(state["_unit_elements"])
        
        # Restore all attributes
        for key, value in state.items():
            object.__setattr__(_self, key, value)
        
        # Ensure _dimension is properly set (in case it wasn't in the state)
        if not hasattr(_self, "_dimension"):
            from ...dimension.dimension import Dimension
            object.__setattr__(_self, "_dimension", Dimension(_self))

    def _to_pickle(self) -> bytes:
        """
        Serialize the unit to pickle format.
        
        Returns:
            Pickle bytes representation of the unit
            
        Examples:
            force = Unit("kg*m/s^2")
            pickle_data = force._to_pickle()
        """
        _self: "Unit" = self # type: ignore
        
        # Serialize using the JSON string structure for consistency
        json_string = _self._to_json()
        return pickle.dumps(json_string)
    
    @classmethod
    def _from_pickle(cls, data: bytes) -> "Unit":
        """
        Deserialize the unit from pickle format.
        
        Args:
            data: Pickle bytes representation of the unit
        
        Returns:
            A new Unit object
            
        Examples:
            pickle_data = force._to_pickle()
            force_restored = Unit._from_pickle(pickle_data)
        """
        # Deserialize the JSON string structure
        json_string: str = pickle.loads(data)
        return cls._from_json(json_string)

    #####################################################################################
    # CSV Serialization
    #####################################################################################

    def _to_csv(self) -> str:
        """
        Serialize the unit to CSV format.
        
        Returns:
            CSV string representation of the unit
            
        Examples:
            force = Unit("kg*m/s^2")
            csv_str = force._to_csv()
        """
        _self: "Unit" = self # type: ignore
        
        output = StringIO()
        writer = csv.writer(output)
        # Write header
        writer.writerow(["unit"])
        # Write data row
        writer.writerow([_self.format_string()])
        return output.getvalue()
    
    @classmethod
    def _from_csv(cls, data: str) -> "Unit":
        """
        Deserialize the unit from CSV format.
        
        Args:
            data: CSV string representation of the unit
        
        Returns:
            A new Unit object
            
        Examples:
            csv_str = force._to_csv()
            force_restored = Unit._from_csv(csv_str)
        """
        reader = csv.reader(StringIO(data))
        # Skip header row
        next(reader)
        # Read data row
        row = next(reader)
        unit_string = row[0]
        return cls._from_json(unit_string)

    #####################################################################################
    # YAML Serialization
    #####################################################################################

    def _to_yaml(self) -> str:
        """
        Serialize the unit to YAML format.
        
        Returns:
            YAML string representation of the unit
            
        Examples:
            force = Unit("kg*m/s^2")
            yaml_str = force._to_yaml()
        
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not _yaml_available or _yaml_module is None:
            raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")
        
        _self: "Unit" = self # type: ignore
        
        # Use the JSON string structure and convert to YAML
        json_string = _self._to_json()
        return _yaml_module.dump(json_string, default_flow_style=False)
    
    @classmethod
    def _from_yaml(cls, data: str) -> "Unit":
        """
        Deserialize the unit from YAML format.
        
        Args:
            data: YAML string representation of the unit
        
        Returns:
            A new Unit object
            
        Examples:
            yaml_str = force._to_yaml()
            force_restored = Unit._from_yaml(yaml_str)
        
        Raises:
            ImportError: If PyYAML is not installed
        """
        if not _yaml_available or _yaml_module is None:
            raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")
        
        # Load YAML and use JSON deserialization
        json_string: str = _yaml_module.safe_load(data)
        return cls._from_json(json_string)