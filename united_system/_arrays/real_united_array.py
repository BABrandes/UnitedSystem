from .._units_and_dimension.unit import Unit
from .base_united_array import BaseUnitedArray
from .._scalars.real_united_scalar import RealUnitedScalar
from .._units_and_dimension.dimension import Dimension
from .._units_and_dimension.named_quantity import NamedQuantity
from dataclasses import dataclass
from typing import Optional, overload
import numpy as np

@dataclass(frozen=True, slots=True, init=False)
class RealUnitedArray(BaseUnitedArray["RealUnitedArray", RealUnitedScalar, float]):
    """Array of real numbers with units."""

    @overload
    def __init__(self, np_array: np.ndarray) -> None:
        ...
    @overload
    def __init__(self, np_array: np.ndarray, unit_or_dimension: Unit) -> None:
        ...
    @overload
    def __init__(self, np_array: np.ndarray, unit_or_dimension: str) -> None:
        ...
    @overload
    def __init__(self, np_array: np.ndarray, unit_or_dimension: Dimension) -> None:
        ...
    @overload
    def __init__(self, np_array: np.ndarray, unit_or_dimension: NamedQuantity) -> None:
        ...
    @overload
    def __init__(self, np_array: np.ndarray, unit_or_dimension: Dimension, display_unit: Unit) -> None:
        ...
    @overload
    def __init__(self, np_array: np.ndarray, unit_or_dimension: NamedQuantity, display_unit: Unit) -> None:
        ...
    def __init__(self, np_array: np.ndarray, unit_or_dimension: Optional[Unit|str|Dimension|NamedQuantity] = None, display_unit: Optional[Unit] = None) -> None:

        if np.iscomplexobj(np_array):
            raise ValueError("RealUnitedArray does not support complex numbers.")
        
        if np_array.ndim != 1:
            raise ValueError("RealUnitedArray only supports 1D arrays.")

        if display_unit is None:
            if isinstance(unit_or_dimension, Unit|str):
                if isinstance(unit_or_dimension, str):
                    _display_unit: Optional[Unit] = Unit(unit_or_dimension)
                else:
                    _display_unit: Optional[Unit] = unit_or_dimension
                _dimension: Dimension = _display_unit.dimension
                _canonical_np_array: np.ndarray = _display_unit.to_canonical_value(np_array) # type: ignore
            elif isinstance(unit_or_dimension, Dimension):
                _dimension: Dimension = unit_or_dimension
                _display_unit: Optional[Unit] = None
                _canonical_np_array: np.ndarray = np_array
            elif isinstance(unit_or_dimension, NamedQuantity):
                _dimension: Dimension = Dimension(unit_or_dimension)
                _display_unit: Optional[Unit] = None
                _canonical_np_array: np.ndarray = np_array
            else:
                _dimension: Dimension = Dimension.dimensionless_dimension()
                _display_unit: Optional[Unit] = None
                _canonical_np_array: np.ndarray = np_array
        else:
            _dimension: Dimension = display_unit.dimension
            _display_unit: Optional[Unit] = display_unit
            _canonical_np_array: np.ndarray = display_unit.to_canonical_value(np_array) # type: ignore

        object.__setattr__(self, "canonical_np_array", _canonical_np_array)
        object.__setattr__(self, "dimension", _dimension)
        object.__setattr__(self, "_display_unit", _display_unit)

    def get_as_scalar(self, index: int) -> RealUnitedScalar:
        """Get a single element as a RealUnitedScalar."""
        canonical_value: float = self.canonical_np_array[index]
        return RealUnitedScalar.create_from_canonical_value(canonical_value, self.dimension, self._display_unit)

    def get_scalar_from_value(self, value: float) -> RealUnitedScalar:
        """Create a RealUnitedScalar from a primitive value with this array's dimension and display unit."""
        canonical_value: float = self.unit.to_canonical_value(value) # type: ignore
        return RealUnitedScalar.create_from_canonical_value(canonical_value, self.dimension, self._display_unit)
    
    def get_numpy_array(self, slice: slice|None = None) -> np.ndarray:
        """Get the numpy array in the display unit.
        
        If slice is provided, the array is returned as a slice of the original array.
        If no slice is provided, the array is returned as a copy of the original array.
        """
        if slice is None:
            return self.unit.from_canonical_value(self.canonical_np_array) # type: ignore
        else:
            return self.unit.from_canonical_value(self.canonical_np_array[slice]) # type: ignore
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the array."""
        return self.canonical_np_array.shape
    
    @property
    def dtype(self) -> np.dtype:
        """Get the dtype of the array."""
        return self.canonical_np_array.dtype
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the array."""
        return self.canonical_np_array.ndim
    
    @property
    def active_dimension(self) -> Dimension:
        """Get the active dimension of the array."""
        return self.dimension
    