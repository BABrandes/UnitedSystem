from enum import Enum
from dataclasses import dataclass, field
from .named_simple_unit_quantities import NamedSimpleUnitQuantity
from .unit import Unit, UnitQuantity
from .simple_unit_quantity import SimpleUnitQuantity
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simple_unit import SimpleUnit

FORBIDDEN_STRINGS_IN_UNIT_NAME: set[str] = {"|", ":", "__"}

def special_character_to_unicode_replacement(string: str) -> str:
    string = string.replace("Ω", "Ω")
    string = string.replace("µ", "μ")
    return string

@dataclass
class NamedUnitInformation:
    unit_string: str
    named_simple_unit_quantity: NamedSimpleUnitQuantity = field(repr=False, compare=False, hash=False)
    _unit: "SimpleUnit|None" = field(default=None, repr=False, compare=False, hash=False)

    @classmethod
    def create(cls, unit_string: str, named_simple_unit_quantity: NamedSimpleUnitQuantity) -> "NamedUnitInformation":
        return cls(unit_string, named_simple_unit_quantity)

    @property
    def unit(self) -> "SimpleUnit":
        if self._unit is None:
            from .simple_unit import SimpleUnit
            unit: SimpleUnit = SimpleUnit.parse(self.unit_string)
            if unit.unit_quantity != self.named_simple_unit_quantity.simple_unit_quantity:
                raise ValueError(f"Unit {self.unit_string} has canonical quantity {unit.unit_quantity} but expected {self.named_simple_unit_quantity.simple_unit_quantity}")
            object.__setattr__(self, '_unit', unit)
        return self._unit

class NamedUnit(Enum):
    value: NamedUnitInformation

    s = NamedUnitInformation.create(                "s",            NamedSimpleUnitQuantity.TIME)
    ms = NamedUnitInformation.create(               "ms",           NamedSimpleUnitQuantity.TIME)
    µs = NamedUnitInformation.create(               "µs",           NamedSimpleUnitQuantity.TIME)
    ns = NamedUnitInformation.create(               "ns",           NamedSimpleUnitQuantity.TIME)
    ps = NamedUnitInformation.create(               "ps",           NamedSimpleUnitQuantity.TIME)
    fs = NamedUnitInformation.create(               "fs",           NamedSimpleUnitQuantity.TIME)

    min = NamedUnitInformation.create(              "min",          NamedSimpleUnitQuantity.TIME)
    h = NamedUnitInformation.create(                "h",            NamedSimpleUnitQuantity.TIME)
    d = NamedUnitInformation.create(                "days",         NamedSimpleUnitQuantity.TIME)
    w = NamedUnitInformation.create(                "weeks",        NamedSimpleUnitQuantity.TIME)
    mo = NamedUnitInformation.create(               "months",       NamedSimpleUnitQuantity.TIME)
    y = NamedUnitInformation.create(                "years",        NamedSimpleUnitQuantity.TIME)

    m = NamedUnitInformation.create(                "m",            NamedSimpleUnitQuantity.LENGTH)
    km = NamedUnitInformation.create(               "km",           NamedSimpleUnitQuantity.LENGTH)
    mm = NamedUnitInformation.create(               "mm",           NamedSimpleUnitQuantity.LENGTH)
    µm = NamedUnitInformation.create(               "µm",           NamedSimpleUnitQuantity.LENGTH)
    nm = NamedUnitInformation.create(               "nm",           NamedSimpleUnitQuantity.LENGTH)
    pm = NamedUnitInformation.create(               "pm",           NamedSimpleUnitQuantity.LENGTH)
    fm = NamedUnitInformation.create(               "fm",           NamedSimpleUnitQuantity.LENGTH)

    kg = NamedUnitInformation.create(               "kg",           NamedSimpleUnitQuantity.MASS)
    g = NamedUnitInformation.create(                "g",            NamedSimpleUnitQuantity.MASS)
    mg = NamedUnitInformation.create(               "mg",           NamedSimpleUnitQuantity.MASS)
    µg = NamedUnitInformation.create(               "µg",           NamedSimpleUnitQuantity.MASS)
    ng = NamedUnitInformation.create(               "ng",           NamedSimpleUnitQuantity.MASS)
    pg = NamedUnitInformation.create(               "pg",           NamedSimpleUnitQuantity.MASS)
    fg = NamedUnitInformation.create(               "fg",           NamedSimpleUnitQuantity.MASS)

    A = NamedUnitInformation.create(                "A",            NamedSimpleUnitQuantity.CURRENT)
    mA = NamedUnitInformation.create(               "mA",           NamedSimpleUnitQuantity.CURRENT)
    µA = NamedUnitInformation.create(               "µA",           NamedSimpleUnitQuantity.CURRENT)
    nA = NamedUnitInformation.create(               "nA",           NamedSimpleUnitQuantity.CURRENT)
    pA = NamedUnitInformation.create(               "pA",           NamedSimpleUnitQuantity.CURRENT)
    fA = NamedUnitInformation.create(               "fA",           NamedSimpleUnitQuantity.CURRENT)
    aA = NamedUnitInformation.create(               "aA",           NamedSimpleUnitQuantity.CURRENT)

    K = NamedUnitInformation.create(                "K",            NamedSimpleUnitQuantity.TEMPERATURE)
    C = NamedUnitInformation.create(                "°C",           NamedSimpleUnitQuantity.TEMPERATURE)
    F = NamedUnitInformation.create(                "°F",           NamedSimpleUnitQuantity.TEMPERATURE)
    R = NamedUnitInformation.create(                "°R",           NamedSimpleUnitQuantity.TEMPERATURE)

    V = NamedUnitInformation.create(                "V",            NamedSimpleUnitQuantity.VOLTAGE)
    MV = NamedUnitInformation.create(               "MV",           NamedSimpleUnitQuantity.VOLTAGE)
    kV = NamedUnitInformation.create(               "kV",           NamedSimpleUnitQuantity.VOLTAGE)
    mV = NamedUnitInformation.create(               "mV",           NamedSimpleUnitQuantity.VOLTAGE)
    µV = NamedUnitInformation.create(               "µV",           NamedSimpleUnitQuantity.VOLTAGE)
    nV = NamedUnitInformation.create(               "nV",           NamedSimpleUnitQuantity.VOLTAGE)
    pV = NamedUnitInformation.create(               "pV",           NamedSimpleUnitQuantity.VOLTAGE)
    fV = NamedUnitInformation.create(               "fV",           NamedSimpleUnitQuantity.VOLTAGE)
    aV = NamedUnitInformation.create(               "aV",           NamedSimpleUnitQuantity.VOLTAGE)
    zV = NamedUnitInformation.create(               "zV",           NamedSimpleUnitQuantity.VOLTAGE)

    GΩ = NamedUnitInformation.create(               "GΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    MΩ = NamedUnitInformation.create(               "MΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    Ω = NamedUnitInformation.create(                "Ω",            NamedSimpleUnitQuantity.RESISTANCE)
    kΩ = NamedUnitInformation.create(               "kΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    mΩ = NamedUnitInformation.create(               "mΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    µΩ = NamedUnitInformation.create(               "µΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    nΩ = NamedUnitInformation.create(               "nΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    pΩ = NamedUnitInformation.create(               "pΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    fΩ = NamedUnitInformation.create(               "fΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    aΩ = NamedUnitInformation.create(               "aΩ",           NamedSimpleUnitQuantity.RESISTANCE)
    zΩ = NamedUnitInformation.create(               "zΩ",           NamedSimpleUnitQuantity.RESISTANCE)

    V_per_s = NamedUnitInformation.create(          "V/s",          NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE) 
    mV_per_s = NamedUnitInformation.create(         "mV/s",         NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE)
    µV_per_s = NamedUnitInformation.create(         "µV/s",         NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE)
    nV_per_s = NamedUnitInformation.create(         "nV/s",         NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE)
    pV_per_s = NamedUnitInformation.create(         "pV/s",         NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE)
    fV_per_s = NamedUnitInformation.create(         "fV/s",         NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE)
    aV_per_s = NamedUnitInformation.create(         "aV/s",         NamedSimpleUnitQuantity.VOLTAGE_SCAN_RATE)

    A_per_m2 = NamedUnitInformation.create(         "A/m2",         NamedSimpleUnitQuantity.CURRENT_DENSITY)
    mA_per_m2 = NamedUnitInformation.create(        "mA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    µA_per_m2 = NamedUnitInformation.create(        "µA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    nA_per_m2 = NamedUnitInformation.create(        "nA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    pA_per_m2 = NamedUnitInformation.create(        "pA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    fA_per_m2 = NamedUnitInformation.create(        "fA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    aA_per_m2 = NamedUnitInformation.create(        "aA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    zA_per_m2 = NamedUnitInformation.create(        "zA/m2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    A_per_cm2 = NamedUnitInformation.create(        "A/cm2",        NamedSimpleUnitQuantity.CURRENT_DENSITY)
    mA_per_cm2 = NamedUnitInformation.create(       "mA/cm2",       NamedSimpleUnitQuantity.CURRENT_DENSITY)
    µA_per_cm2 = NamedUnitInformation.create(       "µA/cm2",       NamedSimpleUnitQuantity.CURRENT_DENSITY)
    nA_per_cm2 = NamedUnitInformation.create(       "nA/cm2",       NamedSimpleUnitQuantity.CURRENT_DENSITY)
    pA_per_cm2 = NamedUnitInformation.create(       "pA/cm2",       NamedSimpleUnitQuantity.CURRENT_DENSITY)

    F_per_m2 = NamedUnitInformation.create(         "F/m2",         NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)
    mF_per_m2 = NamedUnitInformation.create(        "mF/m2",        NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)
    µF_per_m2 = NamedUnitInformation.create(        "µF/m2",        NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)
    nF_per_m2 = NamedUnitInformation.create(        "nF/m2",        NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)
    pF_per_m2 = NamedUnitInformation.create(        "pF/m2",        NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)
    fF_per_m2 = NamedUnitInformation.create(        "fF/m2",        NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)
    aF_per_m2 = NamedUnitInformation.create(        "aF/m2",        NamedSimpleUnitQuantity.CAPACITANCE_DENSITY)

    C_per_m2 = NamedUnitInformation.create(         "C/m2",         NamedSimpleUnitQuantity.CHARGE_DENSITY)
    mC_per_m2 = NamedUnitInformation.create(        "mC/m2",        NamedSimpleUnitQuantity.CHARGE_DENSITY)
    µC_per_m2 = NamedUnitInformation.create(        "µC/m2",        NamedSimpleUnitQuantity.CHARGE_DENSITY)
    nC_per_m2 = NamedUnitInformation.create(        "nC/m2",        NamedSimpleUnitQuantity.CHARGE_DENSITY)
    pC_per_m2 = NamedUnitInformation.create(        "pC/m2",        NamedSimpleUnitQuantity.CHARGE_DENSITY)
    fC_per_m2 = NamedUnitInformation.create(        "fC/m2",        NamedSimpleUnitQuantity.CHARGE_DENSITY)
    aC_per_m2 = NamedUnitInformation.create(        "aC/m2",        NamedSimpleUnitQuantity.CHARGE_DENSITY)

    @property
    def unit(self) -> "SimpleUnit":
        return self.value.unit

    @property
    def simple_unit_quantity(self) -> SimpleUnitQuantity:
        return self.value.unit.unit_quantity
    
    @property
    def named_simple_unit_quantity(self) -> NamedSimpleUnitQuantity:
        return self.value.named_simple_unit_quantity

    @property
    def canonical_unit(self) -> Unit:
        return self.value.named_simple_unit_quantity.canonical_unit
    
    @staticmethod
    def has_named_unit(unit_quantity: UnitQuantity) -> bool:
        return len(NamedUnit.find_named_units(unit_quantity)) > 0

    @staticmethod
    def find_named_units(unit_quantity: UnitQuantity) -> set["NamedUnit"]:
        named_units: set["NamedUnit"] = set()
        for named_unit in NamedUnit:
            if named_unit.simple_unit_quantity == unit_quantity:
                named_units.add(named_unit)
        return named_units
