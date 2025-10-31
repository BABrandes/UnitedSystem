"""Serialization methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Any, Optional, Literal
import h5py
import pickle
import csv
from io import StringIO

from united_system import Unit, Dimension

from .protocol import RealUnitedScalarProtocol

# Optional YAML support
try:
    import yaml as _yaml_module  # type: ignore[import-untyped]
    _yaml_available = True
except ImportError:
    _yaml_available = False
    _yaml_module = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from ...._scalars.real_united_scalar import RealUnitedScalar

class SerializationMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Serialization functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    def serialize(self, format: Optional[Literal["json", "hdf5", "pickle", "csv", "yaml"]], **kwargs: Any) -> Any:
        """Serialize to JSON, HDF5, pickle, CSV, or YAML."""
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

    @classmethod
    def deserialize(cls, data: Any, format: Optional[Literal["json", "hdf5", "pickle", "csv", "yaml"]], **kwargs: Any) -> "RealUnitedScalar":
        """Deserialize from JSON, HDF5, pickle, CSV, or YAML."""
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

    #########################################################
    # JSON Serialization
    #########################################################

    def _to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "canonical_value": self.canonical_value,
            "dimension": self.dimension.serialize("json"),
            "display_unit": self._display_unit.serialize("json") if self._display_unit else None
        }
    
    @classmethod
    def _from_json(cls, data: dict[str, Any]) -> "RealUnitedScalar":
        """Create from dictionary (JSON deserialization)."""
        from ...._scalars.real_united_scalar import RealUnitedScalar

        dimension = Dimension.deserialize(data["dimension"], "json")
        display_unit: Optional[Unit] = Unit.deserialize(data["display_unit"], "json") if data["display_unit"] is not None else None
        
        return RealUnitedScalar.create_from_canonical_value(data["canonical_value"], dimension, display_unit)
    

    #########################################################
    # HDF5 Serialization
    #########################################################

    def _to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Save to HDF5 group."""
        hdf5_group.create_dataset("canonical_value", data=self.canonical_value) # type: ignore
        
        # Save dimension
        dimension_group = hdf5_group.create_group("dimension") # type: ignore
        self.dimension.serialize("hdf5", hdf5_group=dimension_group)
        
        # Save display unit
        if self._display_unit:
            display_unit_group = hdf5_group.create_group("display_unit") # type: ignore
            self._display_unit.serialize("hdf5", hdf5_group=display_unit_group)
        else:
            hdf5_group.create_dataset("display_unit", data=None) # type: ignore

    @classmethod
    def _from_hdf5(cls, hdf5_group: h5py.Group) -> "RealUnitedScalar":
        """Load from HDF5 group."""
        from ...._scalars.real_united_scalar import RealUnitedScalar
        
        canonical_value: float = float(hdf5_group["canonical_value"][()]) # type: ignore   
        dimension: Dimension = Dimension.deserialize(hdf5_group["dimension"], "hdf5") # type: ignore
        
        # Load display unit
        display_unit: Optional[Unit] = None
        if "display_unit" in hdf5_group:
            display_unit_item = hdf5_group["display_unit"]
            if isinstance(display_unit_item, h5py.Group):
                display_unit = Unit.deserialize(display_unit_item, "hdf5") # type: ignore
        
        return RealUnitedScalar.create_from_canonical_value(canonical_value, dimension, display_unit)

    #########################################################
    # Pickle Serialization
    #########################################################

    def _to_pickle(self) -> bytes:
        """Save to pickle format."""
        # Serialize using the JSON dict structure for consistency
        json_data = self._to_json()
        return pickle.dumps(json_data)
    
    @classmethod
    def _from_pickle(cls, data: bytes) -> "RealUnitedScalar":
        """Load from pickle format."""
        # Deserialize the JSON dict structure
        json_data: dict[str, Any] = pickle.loads(data)
        return cls._from_json(json_data)

    #########################################################
    # CSV Serialization
    #########################################################

    def _to_csv(self) -> str:
        """Save to CSV format."""
        output = StringIO()
        writer = csv.writer(output)
        # Write headers
        writer.writerow(["canonical_value", "dimension", "display_unit"])
        # Write data row
        display_unit_str = self._display_unit.serialize("json") if self._display_unit else ""
        writer.writerow([str(self.canonical_value), self.dimension.serialize("json"), display_unit_str])
        return output.getvalue()
    
    @classmethod
    def _from_csv(cls, data: str) -> "RealUnitedScalar":
        """Load from CSV format."""
        from ...._scalars.real_united_scalar import RealUnitedScalar
        
        reader = csv.reader(StringIO(data))
        # Skip header row
        next(reader)
        # Read data row
        row = next(reader)
        canonical_value = float(row[0])
        dimension = Dimension.deserialize(row[1], "json")
        display_unit: Optional[Unit] = Unit.deserialize(row[2], "json") if row[2] else None
        
        return RealUnitedScalar.create_from_canonical_value(canonical_value, dimension, display_unit)

    #########################################################
    # YAML Serialization
    #########################################################

    def _to_yaml(self) -> str:
        """Save to YAML format."""
        if not _yaml_available or _yaml_module is None:
            raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")
        # Use the JSON dict structure and convert to YAML
        json_data = self._to_json()
        return _yaml_module.dump(json_data, default_flow_style=False)
    
    @classmethod
    def _from_yaml(cls, data: str) -> "RealUnitedScalar":
        """Load from YAML format."""
        if not _yaml_available or _yaml_module is None:
            raise ImportError("PyYAML is required for YAML serialization. Install it with: pip install pyyaml")
        # Load YAML and use JSON deserialization
        json_data: dict[str, Any] = _yaml_module.safe_load(data)
        return cls._from_json(json_data)