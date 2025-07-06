from dataclasses import dataclass
from ..united_scalar import UnitedScalar
from ...units.simple.simple_unit import SimpleUnit, SimpleDimension
from ...units.base_classes.base_unit import BaseUnit

@dataclass(frozen=True, slots=True)
class ComplexUnitedScalar(UnitedScalar["ComplexUnitedScalar", SimpleUnit, SimpleDimension, complex]):
    
    @classmethod
    def create(cls, value: complex, display_unit: SimpleUnit) -> "ComplexUnitedScalar":
        raise NotImplementedError("ComplexUnitedScalar.create is not implemented")
