from dataclasses import dataclass
from .united_scalar import UnitedScalar
from ..units.unit import Unit

@dataclass(frozen=True, slots=True)
class ComplexUnitedScalar(UnitedScalar):
    pass

@classmethod
def create(cls, value: complex, display_unit: Unit) -> ComplexUnitedScalar:
    raise NotImplementedError("ComplexUnitedScalar.create is not implemented")
