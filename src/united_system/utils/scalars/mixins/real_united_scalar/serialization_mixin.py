"""Serialization methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Any, Type, Optional
import h5py

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....unit import Unit
    from .....dimension import Dimension

class SerializationMixin:
    """Serialization functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "canonical_value": self.canonical_value,
            "canonical_unit": self.dimension.canonical_unit.format_string(no_fraction=False),
            "display_unit": self._display_unit.format_string(no_fraction=False) if self._display_unit else None
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any], **_: Type["Unit"]) -> "RealUnitedScalar":
        """Create from dictionary (JSON deserialization)."""
        from .....unit import Unit
        from .....real_united_scalar import RealUnitedScalar

        canonical_unit = Unit.parse_string(data["canonical_unit"])
        display_unit: Optional["Unit"] = Unit.parse_string(data["display_unit"]) if data["display_unit"] is not None else None
        
        return RealUnitedScalar(
            canonical_value=data["canonical_value"],
            dimension=canonical_unit.dimension,
            display_unit=display_unit
        )
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Save to HDF5 group."""
        hdf5_group.create_dataset("canonical_value", data=self.canonical_value) # type: ignore
        hdf5_group.create_dataset( # type: ignore
            "canonical_unit", 
            data=self.dimension.canonical_unit.format_string(no_fraction=False),
            dtype=h5py.string_dtype(encoding='utf-8') # type: ignore
        )
        hdf5_group.create_dataset( # type: ignore
            "display_unit", 
            data=self._display_unit.format_string(no_fraction=False) if self._display_unit else None,
            dtype=h5py.string_dtype(encoding='utf-8') # type: ignore
        )

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, **type_parameters: Type[Any]) -> "RealUnitedScalar":
        """Load from HDF5 group."""
        from .....unit import Unit
        from .....real_united_scalar import RealUnitedScalar
        
        canonical_value: float = float(hdf5_group["canonical_value"][()]) # type: ignore   
        canonical_unit: "Unit" = Unit.parse_string(hdf5_group["canonical_unit"][()].decode('utf-8')) # type: ignore
        display_unit: Optional["Unit"] = Unit.parse_string(hdf5_group["display_unit"][()].decode('utf-8')) if hdf5_group["display_unit"][()] is not None else None # type: ignore
        
        return RealUnitedScalar(
            canonical_value=canonical_value,
            dimension=canonical_unit.dimension,
            display_unit=display_unit
        ) 