"""Serialization methods for RealUnitedScalar."""

from typing import TYPE_CHECKING, Any, Optional
import h5py
from .protocol import RealUnitedScalarProtocol

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar
    from .....unit import Unit
    from .....dimension import Dimension

class SerializationMixin(RealUnitedScalarProtocol["RealUnitedScalar"]):
    """Serialization functionality for RealUnitedScalar."""
    
    # These will be provided by the core class
    canonical_value: float
    dimension: "Dimension"
    _display_unit: Optional["Unit"]

    def to_json(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "canonical_value": self.canonical_value,
            "dimension": self.dimension.to_json(),
            "display_unit": self._display_unit.to_json() if self._display_unit else None
        }
    
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "RealUnitedScalar":
        """Create from dictionary (JSON deserialization)."""
        from .....unit import Unit
        from .....dimension import Dimension
        from .....real_united_scalar import RealUnitedScalar

        dimension = Dimension.from_json(data["dimension"])
        display_unit: Optional[Unit] = Unit.from_json(data["display_unit"]) if data["display_unit"] is not None else None
        
        return RealUnitedScalar.create_from_canonical_value(data["canonical_value"], dimension, display_unit)
    
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        """Save to HDF5 group."""
        hdf5_group.create_dataset("canonical_value", data=self.canonical_value) # type: ignore
        
        # Save dimension
        dimension_group = hdf5_group.create_group("dimension") # type: ignore
        self.dimension.to_hdf5(dimension_group)
        
        # Save display unit
        if self._display_unit:
            display_unit_group = hdf5_group.create_group("display_unit") # type: ignore
            self._display_unit.to_hdf5(display_unit_group)
        else:
            hdf5_group.create_dataset("display_unit", data=None) # type: ignore

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> "RealUnitedScalar":
        """Load from HDF5 group."""
        from .....unit import Unit
        from .....dimension import Dimension
        from .....real_united_scalar import RealUnitedScalar
        
        canonical_value: float = float(hdf5_group["canonical_value"][()]) # type: ignore   
        dimension: Dimension = Dimension.from_hdf5(hdf5_group["dimension"]) # type: ignore
        
        # Load display unit
        display_unit: Optional[Unit] = None
        if "display_unit" in hdf5_group:
            display_unit_item = hdf5_group["display_unit"]
            if isinstance(display_unit_item, h5py.Group):
                display_unit = Unit.from_hdf5(display_unit_item) # type: ignore
        
        return RealUnitedScalar.create_from_canonical_value(canonical_value, dimension, display_unit)