from .unit_quantity import SimpleCanonicalQuantity
from dataclasses import dataclass
from .unit import Unit, SimpleUnit, CanonicalQuantity
from typing import Tuple
from enum import Enum

@dataclass(frozen=True, slots=True)
class NamedCanonicalQuantityInformation():
    canonical_quantity: SimpleCanonicalQuantity
    has_canonical_unit: bool
    canonical_unit: SimpleUnit

    @classmethod
    def create(cls, quantity_exponents: Tuple[float, float, float, float, float, float, float], pseudo_quantity_exponents: Tuple[int, int], has_canonical_unit: bool, unit_string: str) -> "NamedCanonicalQuantityInformation":
        canonical_quantity: SimpleCanonicalQuantity = SimpleCanonicalQuantity.create(quantity_exponents, pseudo_quantity_exponents)
        unit: Unit = SimpleUnit.parse(unit_string)
        return cls(canonical_quantity, has_canonical_unit, unit)


class NamedCanonicalQuantity(Enum):
    value: NamedCanonicalQuantityInformation

    # -------------------- Quantities with Canonical SI Base Units --------------------

    MASS = NamedCanonicalQuantityInformation.create(                    (1, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "kg")
    TIME = NamedCanonicalQuantityInformation.create(                    (0, 1, 0, 0, 0, 0, 0), (0, 0),      True,    "s")
    LENGTH = NamedCanonicalQuantityInformation.create(                  (0, 0, 1, 0, 0, 0, 0), (0, 0),      True,   "m")
    CURRENT = NamedCanonicalQuantityInformation.create(                 (0, 0, 0, 1, 0, 0, 0), (0, 0),      True,   "A")
    TEMPERATURE = NamedCanonicalQuantityInformation.create(             (0, 0, 0, 0, 1, 0, 0), (0, 0),      True,   "K")
    AMOUNT_OF_SUBSTANCE = NamedCanonicalQuantityInformation.create(     (0, 0, 0, 0, 0, 1, 0), (0, 0),      True,   "mol")
    LUMINOUS_INTENSITY = NamedCanonicalQuantityInformation.create(      (0, 0, 0, 0, 0, 0, 1), (0, 0),      True,   "cd")
    
    # -------------------- Quantities with Canonical SI Derived Units --------------------

    FORCE = NamedCanonicalQuantityInformation.create(                   (1, 1, -2, 0, 0, 0, 0), (0, 0),     True,   "N")
    ENERGY = NamedCanonicalQuantityInformation.create(                  (2, 1, -2, 0, 0, 0, 0), (0, 0),     True,   "J")
    POWER = NamedCanonicalQuantityInformation.create(                   (2, 1, -3, 0, 0, 0, 0), (0, 0),     True,   "W")
    PRESSURE = NamedCanonicalQuantityInformation.create(                (1, -1, -2, 0, 0, 0, 0), (0, 0),    True,   "Pa")
    VOLTAGE = NamedCanonicalQuantityInformation.create(                 (1, 1, -3, 0, 0, 0, 0), (0, 0),     True,   "V")
    RESISTANCE = NamedCanonicalQuantityInformation.create(              (2, -1, -3, 0, 0, 0, 0), (0, 0),    True,   "Ω")
    CONDUCTANCE = NamedCanonicalQuantityInformation.create(             (-2, 1, -3, 0, 0, 0, 0), (0, 0),    True,   "S")
    CAPACITANCE = NamedCanonicalQuantityInformation.create(             (-2, -1, 4, 0, 0, 0, 0), (0, 0),    True,   "F")
    CHARGE = NamedCanonicalQuantityInformation.create(                  (1, 1, -1, 0, 0, 0, 0), (0, 0),     True,   "C")
    INDUCTANCE = NamedCanonicalQuantityInformation.create(              (2, -1, -2, 0, 0, 0, 0), (0, 0),    True,   "H")
    MAGNETIC_FLUX = NamedCanonicalQuantityInformation.create(           (2, 1, -2, 0, 0, 0, 0), (0, 0),     True,   "Wb")
    MAGNETIC_FLUX_DENSITY = NamedCanonicalQuantityInformation.create(   (1, 1, -3, 0, 0, 0, 0), (0, 0),     True,   "T")
    FREQUENCY = NamedCanonicalQuantityInformation.create(               (0, -1, 0, 0, 0, 0, 0), (0, 0),     True,   "Hz")

    # -------------------- Quantities with Composed SI Units --------------------

    VOLTAGE_SCAN_RATE = NamedCanonicalQuantityInformation.create(       (1, -1, -1, 0, 0, 0, 0), (0, 0),    True,   "V/s")
    CURRENT_DENSITY = NamedCanonicalQuantityInformation.create(         (1, -2, 0, 0, 0, 0, 0), (0, 0),     True,   "A/m^2")
    CHARGE_DENSITY = NamedCanonicalQuantityInformation.create(          (1, -2, 0, 0, 0, 0, 0), (0, 0),     True,   "C/m^2")
    CAPACITANCE_DENSITY = NamedCanonicalQuantityInformation.create(     (-2, -2, 4, 0, 0, 0, 0), (0, 0),    True,   "F/m^2")
    
    # -------------------- Quantities with Non-SI Canonical Units --------------------

    ANGLE = NamedCanonicalQuantityInformation.create(                   (0, 0, 0, 0, 0, 0, 0), (0, 1),      True,   "rad")
    ACTIVITY = NamedCanonicalQuantityInformation.create(                (0, -1, 0, 0, 0, 0, 0), (0, 0),     True,   "Bq")
    DOSE_EQUIVALENT = NamedCanonicalQuantityInformation.create(         (2, 0, -2, 0, 0, 0, 0), (0, 0),     True,   "Sv")
    DOSE_ABSORTED = NamedCanonicalQuantityInformation.create(           (2, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "Gy")
    LOGLEVEL = NamedCanonicalQuantityInformation.create(                (0, 0, 0, 0, 0, 0, 0), (1, 0),      True,   "np")
    FRACTION = NamedCanonicalQuantityInformation.create(                (0, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "")
    NUMBER = NamedCanonicalQuantityInformation.create(                  (0, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "")
    
    # -------------------- Quantities with Non-SI Non-canonical Units --------------------
    
    AREA = NamedCanonicalQuantityInformation.create(                    (2, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "m^2")
    VOLUME = NamedCanonicalQuantityInformation.create(                  (3, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "m^3")
    DENSITY = NamedCanonicalQuantityInformation.create(                 (-3, 0, 1, 0, 0, 0, 0), (0, 0),     True,   "kg/m^3")
    VOLUMETRIC_FLOW_RATE = NamedCanonicalQuantityInformation.create(    (3, 0, -1, 0, 0, 0, 0), (0, 0),     True,   "m^3/s")
    AREAL_FLOW_RATE = NamedCanonicalQuantityInformation.create(         (2, 0, -1, 0, 0, 0, 0), (0, 0),     False,  "m^2/s")
    VOLUMETRIC_DENSITY = NamedCanonicalQuantityInformation.create(      (-3, 0, 0, 0, 0, 0, 0), (0, 0),     False,  "kg/m^3")
    AREAL_DENSITY = NamedCanonicalQuantityInformation.create(           (-2, 0, 0, 0, 0, 0, 0), (0, 0),     False,  "kg/m^2")
    KINEMATIC_VISCOSITY = NamedCanonicalQuantityInformation.create(     (0, -2, 0, 0, 0, 0, 0), (0, 0),     True,   "ν")
    DYNAMIC_VISCOSITY = NamedCanonicalQuantityInformation.create(       (1, -1, -1, 0, 0, 0, 0), (0, 0),    True,   "η")

    @property
    def canonical_quantity(self) -> SimpleCanonicalQuantity:
        return self.value.canonical_quantity
    
    @property
    def canonical_unit(self) -> SimpleUnit:
        return self.value.canonical_unit
    
    def __getitem__(self, canonical_quantity: CanonicalQuantity) -> "NamedCanonicalQuantity":
        named_canonical_quantity: NamedCanonicalQuantity|None = self.find_named_canonical_quantity(canonical_quantity)
        if named_canonical_quantity is None:
            raise ValueError(f"No named canonical quantity found for canonical quantity {canonical_quantity}")
        return named_canonical_quantity

    @classmethod
    def has_named_canonical_quantity(cls, canonical_quantity: SimpleCanonicalQuantity) -> bool:
        named_canonical_quantity: NamedCanonicalQuantity|None = cls.find_named_canonical_quantity(canonical_quantity)
        return named_canonical_quantity is not None

    @classmethod
    def find_named_canonical_quantity(cls, canonical_quantity: CanonicalQuantity) -> "NamedCanonicalQuantity|None":
        for named_canonical_quantity in cls:
            if named_canonical_quantity.canonical_quantity == canonical_quantity:
                return named_canonical_quantity
        return None