"""Serialization methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Union
import h5py

if TYPE_CHECKING:
    from ...real_united_scalar import RealUnitedScalar
    from ....units.simple.simple_unit import SimpleUnit
    from ....units.simple.simple_dimension import SimpleDimension

class SerializationMixin:
    """Serialization functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "SimpleDimension"
    display_unit: Union["SimpleUnit", None]

    def to_json(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "canonical_value": self.canonical_value,
            "canonical_unit": self.dimension.canonical_unit.format_string(no_fraction=False),
            "display_unit": self.display_unit.format_string(no_fraction=False) if self.display_unit else None
        }
    
    @classmethod
    def from_json(cls, data: dict) -> "RealUnitedScalar":
        """Create from dictionary (JSON deserialization)."""
        from ....units.simple.simple_unit import SimpleUnit
        from ..real_united_scalar import RealUnitedScalar
        
        canonical_unit = SimpleUnit.parse_string(data["canonical_unit"])
        display_unit = SimpleUnit.parse_string(data["display_unit"]) if data["display_unit"] else None
        
        return RealUnitedScalar(
            canonical_value=data["canonical_value"],
            dimension=canonical_unit.dimension,
            display_unit=display_unit
        )
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Save to HDF5 group."""
        hdf5_group.create_dataset("canonical_value", data=self.canonical_value)
        hdf5_group.create_dataset(
            "canonical_unit", 
            data=self.dimension.canonical_unit.format_string(no_fraction=False),
            dtype=h5py.string_dtype(encoding='utf-8')
        )
        
        if self.display_unit is not None:
            hdf5_group.create_dataset(
                "display_unit", 
                data=self.display_unit.format_string(no_fraction=False),
                dtype=h5py.string_dtype(encoding='utf-8')
            )

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "RealUnitedScalar":
        """Load from HDF5 group."""
        from ....units.simple.simple_unit import SimpleUnit
        from ..real_united_scalar import RealUnitedScalar
        
        canonical_value = float(hdf5_group["canonical_value"][()])
        canonical_unit = SimpleUnit.parse_string(hdf5_group["canonical_unit"][()].decode('utf-8'))
        display_unit = SimpleUnit.parse_string(hdf5_group["display_unit"][()].decode('utf-8')) if "display_unit" in hdf5_group else None
        
        return RealUnitedScalar(
            canonical_value=canonical_value,
            dimension=canonical_unit.dimension,
            display_unit=display_unit
        ) 