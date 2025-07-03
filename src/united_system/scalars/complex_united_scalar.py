from dataclasses import dataclass
from .united_scalar import UnitedScalar

@dataclass(frozen=True, slots=True)
class ComplexUnitedScalar(UnitedScalar):
    pass

