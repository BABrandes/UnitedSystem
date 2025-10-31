from __future__ import annotations
from ...united_scalar import UnitedScalar
from ...._utils.general import SerializationProtocol
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from ...._scalars.real_united_scalar import RealUnitedScalar

T = TypeVar("T", bound="RealUnitedScalar")

class RealUnitedScalarProtocol(UnitedScalar[T, float], SerializationProtocol[T]):
    """
    Protocol for RealUnitedScalar.
    """