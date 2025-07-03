from enum import Enum
from dataclasses import dataclass, field
from .named_canonical_quantities import NamedCanonicalQuantity
from .unit import Unit, SimpleUnit, SimpleCanonicalQuantity, CanonicalQuantity

FORBIDDEN_STRINGS_IN_UNIT_NAME: set[str] = {"|", ":", "__"}

def special_character_to_unicode_replacement(string: str) -> str:
    string = string.replace("Ω", "Ω")
    string = string.replace("µ", "μ")
    return string

@dataclass
class NamedUnitInformation:
    unit: SimpleUnit
    named_canonical_quantity: NamedCanonicalQuantity = field(repr=False, compare=False, hash=False)

    @classmethod
    def create(cls, unit_string: str, named_canonical_quantity: NamedCanonicalQuantity) -> "NamedUnitInformation":
        unit: SimpleUnit = SimpleUnit.parse(unit_string)
        if unit.canonical_quantity != named_canonical_quantity.canonical_quantity:
            raise ValueError(f"Unit {unit_string} has canonical quantity {unit.canonical_quantity} but expected {named_canonical_quantity.canonical_quantity}")
        return cls(unit, named_canonical_quantity)

class NamedUnit(Enum):
    value: NamedUnitInformation

    s = NamedUnitInformation.create(                "s",            NamedCanonicalQuantity.TIME)
    ms = NamedUnitInformation.create(               "ms",           NamedCanonicalQuantity.TIME)
    µs = NamedUnitInformation.create(               "µs",           NamedCanonicalQuantity.TIME)
    ns = NamedUnitInformation.create(               "ns",           NamedCanonicalQuantity.TIME)
    ps = NamedUnitInformation.create(               "ps",           NamedCanonicalQuantity.TIME)
    fs = NamedUnitInformation.create(               "fs",           NamedCanonicalQuantity.TIME)

    min = NamedUnitInformation.create(              "min",          NamedCanonicalQuantity.TIME)
    h = NamedUnitInformation.create(                "h",            NamedCanonicalQuantity.TIME)
    d = NamedUnitInformation.create(                "days",         NamedCanonicalQuantity.TIME)
    w = NamedUnitInformation.create(                "weeks",        NamedCanonicalQuantity.TIME)
    mo = NamedUnitInformation.create(               "months",       NamedCanonicalQuantity.TIME)
    y = NamedUnitInformation.create(                "years",        NamedCanonicalQuantity.TIME)

    m = NamedUnitInformation.create(                "m",            NamedCanonicalQuantity.LENGTH)
    km = NamedUnitInformation.create(               "km",           NamedCanonicalQuantity.LENGTH)
    mm = NamedUnitInformation.create(               "mm",           NamedCanonicalQuantity.LENGTH)
    µm = NamedUnitInformation.create(               "µm",           NamedCanonicalQuantity.LENGTH)
    nm = NamedUnitInformation.create(               "nm",           NamedCanonicalQuantity.LENGTH)
    pm = NamedUnitInformation.create(               "pm",           NamedCanonicalQuantity.LENGTH)
    fm = NamedUnitInformation.create(               "fm",           NamedCanonicalQuantity.LENGTH)

    kg = NamedUnitInformation.create(               "kg",           NamedCanonicalQuantity.MASS)
    g = NamedUnitInformation.create(                "g",            NamedCanonicalQuantity.MASS)
    mg = NamedUnitInformation.create(               "mg",           NamedCanonicalQuantity.MASS)
    µg = NamedUnitInformation.create(               "µg",           NamedCanonicalQuantity.MASS)
    ng = NamedUnitInformation.create(               "ng",           NamedCanonicalQuantity.MASS)
    pg = NamedUnitInformation.create(               "pg",           NamedCanonicalQuantity.MASS)
    fg = NamedUnitInformation.create(               "fg",           NamedCanonicalQuantity.MASS)

    A = NamedUnitInformation.create(                "A",            NamedCanonicalQuantity.CURRENT)
    mA = NamedUnitInformation.create(               "mA",           NamedCanonicalQuantity.CURRENT)
    µA = NamedUnitInformation.create(               "µA",           NamedCanonicalQuantity.CURRENT)
    nA = NamedUnitInformation.create(               "nA",           NamedCanonicalQuantity.CURRENT)
    pA = NamedUnitInformation.create(               "pA",           NamedCanonicalQuantity.CURRENT)
    fA = NamedUnitInformation.create(               "fA",           NamedCanonicalQuantity.CURRENT)
    aA = NamedUnitInformation.create(               "aA",           NamedCanonicalQuantity.CURRENT)

    K = NamedUnitInformation.create(                "K",            NamedCanonicalQuantity.TEMPERATURE)
    C = NamedUnitInformation.create(                "°C",           NamedCanonicalQuantity.TEMPERATURE)
    F = NamedUnitInformation.create(                "°F",           NamedCanonicalQuantity.TEMPERATURE)
    R = NamedUnitInformation.create(                "°R",           NamedCanonicalQuantity.TEMPERATURE)

    V = NamedUnitInformation.create(                "V",            NamedCanonicalQuantity.VOLTAGE)
    MV = NamedUnitInformation.create(               "MV",           NamedCanonicalQuantity.VOLTAGE)
    kV = NamedUnitInformation.create(               "kV",           NamedCanonicalQuantity.VOLTAGE)
    mV = NamedUnitInformation.create(               "mV",           NamedCanonicalQuantity.VOLTAGE)
    µV = NamedUnitInformation.create(               "µV",           NamedCanonicalQuantity.VOLTAGE)
    nV = NamedUnitInformation.create(               "nV",           NamedCanonicalQuantity.VOLTAGE)
    pV = NamedUnitInformation.create(               "pV",           NamedCanonicalQuantity.VOLTAGE)
    fV = NamedUnitInformation.create(               "fV",           NamedCanonicalQuantity.VOLTAGE)
    aV = NamedUnitInformation.create(               "aV",           NamedCanonicalQuantity.VOLTAGE)
    zV = NamedUnitInformation.create(               "zV",           NamedCanonicalQuantity.VOLTAGE)

    GΩ = NamedUnitInformation.create(               "GΩ",           NamedCanonicalQuantity.RESISTANCE)
    MΩ = NamedUnitInformation.create(               "MΩ",           NamedCanonicalQuantity.RESISTANCE)
    Ω = NamedUnitInformation.create(                "Ω",            NamedCanonicalQuantity.RESISTANCE)
    kΩ = NamedUnitInformation.create(               "kΩ",           NamedCanonicalQuantity.RESISTANCE)
    mΩ = NamedUnitInformation.create(               "mΩ",           NamedCanonicalQuantity.RESISTANCE)
    µΩ = NamedUnitInformation.create(               "µΩ",           NamedCanonicalQuantity.RESISTANCE)
    nΩ = NamedUnitInformation.create(               "nΩ",           NamedCanonicalQuantity.RESISTANCE)
    pΩ = NamedUnitInformation.create(               "pΩ",           NamedCanonicalQuantity.RESISTANCE)
    fΩ = NamedUnitInformation.create(               "fΩ",           NamedCanonicalQuantity.RESISTANCE)
    aΩ = NamedUnitInformation.create(               "aΩ",           NamedCanonicalQuantity.RESISTANCE)
    zΩ = NamedUnitInformation.create(               "zΩ",           NamedCanonicalQuantity.RESISTANCE)

    V_per_s = NamedUnitInformation.create(          "V/s",          NamedCanonicalQuantity.VOLTAGE_SCAN_RATE) 
    mV_per_s = NamedUnitInformation.create(         "mV/s",         NamedCanonicalQuantity.VOLTAGE_SCAN_RATE)
    µV_per_s = NamedUnitInformation.create(         "µV/s",         NamedCanonicalQuantity.VOLTAGE_SCAN_RATE)
    nV_per_s = NamedUnitInformation.create(         "nV/s",         NamedCanonicalQuantity.VOLTAGE_SCAN_RATE)
    pV_per_s = NamedUnitInformation.create(         "pV/s",         NamedCanonicalQuantity.VOLTAGE_SCAN_RATE)
    fV_per_s = NamedUnitInformation.create(         "fV/s",         NamedCanonicalQuantity.VOLTAGE_SCAN_RATE)
    aV_per_s = NamedUnitInformation.create(         "aV/s",         NamedCanonicalQuantity.VOLTAGE_SCAN_RATE)

    A_per_m2 = NamedUnitInformation.create(         "A/m2",         NamedCanonicalQuantity.CURRENT_DENSITY)
    mA_per_m2 = NamedUnitInformation.create(        "mA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    µA_per_m2 = NamedUnitInformation.create(        "µA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    nA_per_m2 = NamedUnitInformation.create(        "nA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    pA_per_m2 = NamedUnitInformation.create(        "pA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    fA_per_m2 = NamedUnitInformation.create(        "fA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    aA_per_m2 = NamedUnitInformation.create(        "aA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    zA_per_m2 = NamedUnitInformation.create(        "zA/m2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    A_per_cm2 = NamedUnitInformation.create(        "A/cm2",        NamedCanonicalQuantity.CURRENT_DENSITY)
    mA_per_cm2 = NamedUnitInformation.create(       "mA/cm2",       NamedCanonicalQuantity.CURRENT_DENSITY)
    µA_per_cm2 = NamedUnitInformation.create(       "µA/cm2",       NamedCanonicalQuantity.CURRENT_DENSITY)
    nA_per_cm2 = NamedUnitInformation.create(       "nA/cm2",       NamedCanonicalQuantity.CURRENT_DENSITY)
    pA_per_cm2 = NamedUnitInformation.create(       "pA/cm2",       NamedCanonicalQuantity.CURRENT_DENSITY)

    F_per_m2 = NamedUnitInformation.create(         "F/m2",         NamedCanonicalQuantity.CAPACITANCE_DENSITY)
    mF_per_m2 = NamedUnitInformation.create(        "mF/m2",        NamedCanonicalQuantity.CAPACITANCE_DENSITY)
    µF_per_m2 = NamedUnitInformation.create(        "µF/m2",        NamedCanonicalQuantity.CAPACITANCE_DENSITY)
    nF_per_m2 = NamedUnitInformation.create(        "nF/m2",        NamedCanonicalQuantity.CAPACITANCE_DENSITY)
    pF_per_m2 = NamedUnitInformation.create(        "pF/m2",        NamedCanonicalQuantity.CAPACITANCE_DENSITY)
    fF_per_m2 = NamedUnitInformation.create(        "fF/m2",        NamedCanonicalQuantity.CAPACITANCE_DENSITY)
    aF_per_m2 = NamedUnitInformation.create(        "aF/m2",        NamedCanonicalQuantity.CAPACITANCE_DENSITY)

    C_per_m2 = NamedUnitInformation.create(         "C/m2",         NamedCanonicalQuantity.CHARGE_DENSITY)
    mC_per_m2 = NamedUnitInformation.create(        "mC/m2",        NamedCanonicalQuantity.CHARGE_DENSITY)
    µC_per_m2 = NamedUnitInformation.create(        "µC/m2",        NamedCanonicalQuantity.CHARGE_DENSITY)
    nC_per_m2 = NamedUnitInformation.create(        "nC/m2",        NamedCanonicalQuantity.CHARGE_DENSITY)
    pC_per_m2 = NamedUnitInformation.create(        "pC/m2",        NamedCanonicalQuantity.CHARGE_DENSITY)
    fC_per_m2 = NamedUnitInformation.create(        "fC/m2",        NamedCanonicalQuantity.CHARGE_DENSITY)
    aC_per_m2 = NamedUnitInformation.create(        "aC/m2",        NamedCanonicalQuantity.CHARGE_DENSITY)

    @property
    def unit(self) -> SimpleUnit:
        return self.value.unit

    @property
    def canonical_quantity(self) -> SimpleCanonicalQuantity:
        return self.value.unit.canonical_quantity
    
    @property
    def named_canonical_quantity(self) -> NamedCanonicalQuantity:
        return self.value.named_canonical_quantity

    @property
    def canonical_unit(self) -> Unit:
        return self.value.named_canonical_quantity.canonical_unit
    
    @staticmethod
    def has_named_unit(canonical_quantity: CanonicalQuantity) -> bool:
        return len(NamedUnit.find_named_units(canonical_quantity)) > 0

    @staticmethod
    def find_named_units(canonical_quantity: CanonicalQuantity) -> set["NamedUnit"]:
        named_units: set["NamedUnit"] = set()
        for named_unit in NamedUnit:
            if named_unit.canonical_quantity == canonical_quantity:
                named_units.add(named_unit)
        return named_units
