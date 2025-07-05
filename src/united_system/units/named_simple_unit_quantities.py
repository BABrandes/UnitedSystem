from .simple_unit_quantity import SimpleUnitQuantity
from dataclasses import dataclass
from .unit import UnitQuantity
from typing import Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .simple_unit import SimpleUnit

@dataclass(frozen=True, slots=True)
class NamedSimpleUnitQuantityInformation():
    unit_quantity: SimpleUnitQuantity
    has_canonical_unit: bool
    canonical_unit_string: str
    _canonical_unit: "SimpleUnit|None" = None

    @classmethod
    def create(cls, quantity_exponents: Tuple[float, float, float, float, float, float, float], pseudo_quantity_exponents: Tuple[int, int], has_canonical_unit: bool, unit_string: str) -> "NamedSimpleUnitQuantityInformation":
        canonical_quantity: SimpleUnitQuantity = SimpleUnitQuantity.create(quantity_exponents, pseudo_quantity_exponents)
        return cls(canonical_quantity, has_canonical_unit, unit_string)

    @property
    def canonical_unit(self) -> "SimpleUnit":
        if self._canonical_unit is None:
            from .simple_unit import SimpleUnit
            object.__setattr__(self, '_canonical_unit', SimpleUnit.parse(self.canonical_unit_string))
        return self._canonical_unit

#Reminder:
#MASS_INDEX: int = 0
#TIME_INDEX: int = 1
#LENGTH_INDEX: int = 2
#CURRENT_INDEX: int = 3
#TEMPERATURE_INDEX: int = 4
#AMOUNT_OF_SUBSTANCE_INDEX: int = 5
#LUMINOUS_INTENSITY_INDEX: int = 6

class NamedSimpleUnitQuantity(Enum):
    value: NamedSimpleUnitQuantityInformation

    # -------------------- Quantities with Canonical SI Base Units --------------------

    MASS = NamedSimpleUnitQuantityInformation.create(                    (1, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "kg")
    TIME = NamedSimpleUnitQuantityInformation.create(                    (0, 1, 0, 0, 0, 0, 0), (0, 0),      True,    "s")
    LENGTH = NamedSimpleUnitQuantityInformation.create(                  (0, 0, 1, 0, 0, 0, 0), (0, 0),      True,   "m")
    CURRENT = NamedSimpleUnitQuantityInformation.create(                 (0, 0, 0, 1, 0, 0, 0), (0, 0),      True,   "A")
    TEMPERATURE = NamedSimpleUnitQuantityInformation.create(             (0, 0, 0, 0, 1, 0, 0), (0, 0),      True,   "K")
    AMOUNT_OF_SUBSTANCE = NamedSimpleUnitQuantityInformation.create(     (0, 0, 0, 0, 0, 1, 0), (0, 0),      True,   "mol")
    LUMINOUS_INTENSITY = NamedSimpleUnitQuantityInformation.create(      (0, 0, 0, 0, 0, 0, 1), (0, 0),      True,   "cd")
    
    # -------------------- Quantities with Canonical SI Derived Units --------------------

    FORCE = NamedSimpleUnitQuantityInformation.create(                   (1, -2, 1, 0, 0, 0, 0), (0, 0),     True,   "N")
    ENERGY = NamedSimpleUnitQuantityInformation.create(                  (1, -2, 2, 0, 0, 0, 0), (0, 0),     True,   "J")
    POWER = NamedSimpleUnitQuantityInformation.create(                   (1, -3, 2, 0, 0, 0, 0), (0, 0),     True,   "W")
    PRESSURE = NamedSimpleUnitQuantityInformation.create(                (1, -2, -1, 0, 0, 0, 0), (0, 0),    True,   "Pa")
    VOLTAGE = NamedSimpleUnitQuantityInformation.create(                 (1, -3, 2, -1, 0, 0, 0), (0, 0),     True,   "V")
    RESISTANCE = NamedSimpleUnitQuantityInformation.create(              (1, -3, 2, -2, 0, 0, 0), (0, 0),    True,   "Ω")
    CONDUCTANCE = NamedSimpleUnitQuantityInformation.create(             (-1, 3, -2, 2, 0, 0, 0), (0, 0),    True,   "S")
    CAPACITANCE = NamedSimpleUnitQuantityInformation.create(             (-1, 4, -2, 2, 0, 0, 0), (0, 0),    True,   "F")
    CHARGE = NamedSimpleUnitQuantityInformation.create(                  (0, 1, 0, 1, 0, 0, 0), (0, 0),     True,   "C")
    INDUCTANCE = NamedSimpleUnitQuantityInformation.create(              (1, -2, 2, -2, 0, 0, 0), (0, 0),    True,   "H")
    MAGNETIC_FLUX = NamedSimpleUnitQuantityInformation.create(           (1, -2, 2, -1, 0, 0, 0), (0, 0),     True,   "Wb")
    MAGNETIC_FLUX_DENSITY = NamedSimpleUnitQuantityInformation.create(   (1, -2, 0, -1, 0, 0, 0), (0, 0),     True,   "T")
    FREQUENCY = NamedSimpleUnitQuantityInformation.create(               (0, -1, 0, 0, 0, 0, 0), (0, 0),     True,   "Hz")

    # -------------------- Quantities with Composed SI Units --------------------

    VOLTAGE_SCAN_RATE = NamedSimpleUnitQuantityInformation.create(       (1, -4, 2, -1, 0, 0, 0), (0, 0),    True,   "V/s")
    CURRENT_DENSITY = NamedSimpleUnitQuantityInformation.create(         (0, 0, -2, 1, 0, 0, 0), (0, 0),     True,   "A/m^2")
    CHARGE_DENSITY = NamedSimpleUnitQuantityInformation.create(          (0, 1, -2, 1, 0, 0, 0), (0, 0),     True,   "C/m^2")
    CAPACITANCE_DENSITY = NamedSimpleUnitQuantityInformation.create(     (-1, 4, -4, 2, 0, 0, 0), (0, 0),    True,   "F/m^2")
    
    # -------------------- Quantities with Non-SI Canonical Units --------------------

    ANGLE = NamedSimpleUnitQuantityInformation.create(                   (0, 0, 0, 0, 0, 0, 0), (0, 1),      True,   "rad")
    ACTIVITY = NamedSimpleUnitQuantityInformation.create(                (0, -1, 0, 0, 0, 0, 0), (0, 0),     True,   "Bq")
    DOSE_EQUIVALENT = NamedSimpleUnitQuantityInformation.create(         (0, -2, 2, 0, 0, 0, 0), (0, 0),     True,   "Sv")
    DOSE_ABSORTED = NamedSimpleUnitQuantityInformation.create(           (0, -2, 2, 0, 0, 0, 0), (0, 0),      True,   "Gy")
    LOGLEVEL = NamedSimpleUnitQuantityInformation.create(                (0, 0, 0, 0, 0, 0, 0), (1, 0),      True,   "np")
    FRACTION = NamedSimpleUnitQuantityInformation.create(                (0, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "")
    NUMBER = NamedSimpleUnitQuantityInformation.create(                  (0, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "")
    
    # -------------------- Quantities with Non-SI Non-canonical Units --------------------
    
    AREA = NamedSimpleUnitQuantityInformation.create(                    (0, 0, 2, 0, 0, 0, 0), (0, 0),      True,   "m^2")
    VOLUME = NamedSimpleUnitQuantityInformation.create(                  (0, 0, 3, 0, 0, 0, 0), (0, 0),      True,   "m^3")
    DENSITY = NamedSimpleUnitQuantityInformation.create(                 (1, 0, -3, 0, 0, 0, 0), (0, 0),     True,   "kg/m^3")
    VOLUMETRIC_FLOW_RATE = NamedSimpleUnitQuantityInformation.create(    (0, -1, 3, 0, 0, 0, 0), (0, 0),     True,   "m^3/s")
    AREAL_FLOW_RATE = NamedSimpleUnitQuantityInformation.create(         (0, -1, 2, 0, 0, 0, 0), (0, 0),     False,  "m^2/s")
    VOLUMETRIC_DENSITY = NamedSimpleUnitQuantityInformation.create(      (0, 0, -3, 0, 0, 0, 0), (0, 0),     False,  "1/m^3")
    AREAL_DENSITY = NamedSimpleUnitQuantityInformation.create(           (1, 0, -2, 0, 0, 0, 0), (0, 0),     False,  "kg/m^2")
    KINEMATIC_VISCOSITY = NamedSimpleUnitQuantityInformation.create(     (0, -1, 2, 0, 0, 0, 0), (0, 0),     True,   "m^2/s")
    DYNAMIC_VISCOSITY = NamedSimpleUnitQuantityInformation.create(       (1, -1, -1, 0, 0, 0, 0), (0, 0),    True,   "Pa*s")

    @property
    def simple_unit_quantity(self) -> SimpleUnitQuantity:
        return self.value.unit_quantity
    
    @property
    def canonical_unit(self) -> "SimpleUnit":
        return self.value.canonical_unit
    
    def __getitem__(self, canonical_quantity: UnitQuantity) -> "NamedSimpleUnitQuantity":
        named_canonical_quantity: NamedSimpleUnitQuantity|None = self.find_named_canonical_quantity(canonical_quantity)
        if named_canonical_quantity is None:
            raise ValueError(f"No named canonical quantity found for canonical quantity {canonical_quantity}")
        return named_canonical_quantity

    @classmethod
    def has_named_canonical_quantity(cls, canonical_quantity: SimpleUnitQuantity) -> bool:
        named_canonical_quantity: NamedSimpleUnitQuantity|None = cls.find_named_canonical_quantity(canonical_quantity)
        return named_canonical_quantity is not None

    @classmethod
    def find_named_canonical_quantity(cls, canonical_quantity: UnitQuantity) -> "NamedSimpleUnitQuantity|None":
        for named_canonical_quantity in cls:
            if named_canonical_quantity.simple_unit_quantity == canonical_quantity:
                return named_canonical_quantity
        return None