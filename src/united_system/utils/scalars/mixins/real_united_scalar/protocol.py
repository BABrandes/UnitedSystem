from __future__ import annotations
from ...united_scalar import UnitedScalar
from ....general import JSONable, HDF5able
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from .....real_united_scalar import RealUnitedScalar

T = TypeVar("T", bound="RealUnitedScalar")

class RealUnitedScalarProtocol(UnitedScalar[T, float], JSONable[T], HDF5able[T]):
    """
    Protocol for RealUnitedScalar.
    """