from enum import Enum
from dataclasses import dataclass, field
from ...named_quantity import NamedQuantity

FORBIDDEN_STRINGS_IN_UNIT_NAME: set[str] = {"|", ":", "__"}

def special_character_to_unicode_replacement(string: str) -> str:
    string = string.replace("Ω", "Ω")
    string = string.replace("µ", "μ")
    return string

@dataclass(frozen=True, slots=True)
class NamedUnitInformation:
    unit_string: str
    named_quantity: NamedQuantity = field(repr=False, compare=False, hash=False)

    @classmethod
    def create(cls, unit_string: str, named_quantity: NamedQuantity) -> "NamedUnitInformation":
        return cls(unit_string, named_quantity)

class NamedUnit(Enum):
    value: NamedUnitInformation # type: ignore

    s = NamedUnitInformation.create(                "s",            NamedQuantity.TIME)
    ms = NamedUnitInformation.create(               "ms",           NamedQuantity.TIME)
    µs = NamedUnitInformation.create(               "µs",           NamedQuantity.TIME)
    ns = NamedUnitInformation.create(               "ns",           NamedQuantity.TIME)
    ps = NamedUnitInformation.create(               "ps",           NamedQuantity.TIME)
    fs = NamedUnitInformation.create(               "fs",           NamedQuantity.TIME)

    min = NamedUnitInformation.create(              "min",          NamedQuantity.TIME)
    h = NamedUnitInformation.create(                "h",            NamedQuantity.TIME)
    d = NamedUnitInformation.create(                "days",         NamedQuantity.TIME)
    w = NamedUnitInformation.create(                "weeks",        NamedQuantity.TIME)
    mo = NamedUnitInformation.create(               "months",       NamedQuantity.TIME)
    y = NamedUnitInformation.create(                "years",        NamedQuantity.TIME)

    m = NamedUnitInformation.create(                "m",            NamedQuantity.LENGTH)
    km = NamedUnitInformation.create(               "km",           NamedQuantity.LENGTH)
    mm = NamedUnitInformation.create(               "mm",           NamedQuantity.LENGTH)
    µm = NamedUnitInformation.create(               "µm",           NamedQuantity.LENGTH)
    nm = NamedUnitInformation.create(               "nm",           NamedQuantity.LENGTH)
    pm = NamedUnitInformation.create(               "pm",           NamedQuantity.LENGTH)

    kg = NamedUnitInformation.create(               "kg",           NamedQuantity.MASS)
    g = NamedUnitInformation.create(                "g",            NamedQuantity.MASS)
    mg = NamedUnitInformation.create(               "mg",           NamedQuantity.MASS)
    µg = NamedUnitInformation.create(               "µg",           NamedQuantity.MASS)
    ng = NamedUnitInformation.create(               "ng",           NamedQuantity.MASS)
    pg = NamedUnitInformation.create(               "pg",           NamedQuantity.MASS)
    fg = NamedUnitInformation.create(               "fg",           NamedQuantity.MASS)

    A = NamedUnitInformation.create(                "A",            NamedQuantity.CURRENT)
    mA = NamedUnitInformation.create(               "mA",           NamedQuantity.CURRENT)
    µA = NamedUnitInformation.create(               "µA",           NamedQuantity.CURRENT)
    nA = NamedUnitInformation.create(               "nA",           NamedQuantity.CURRENT)
    pA = NamedUnitInformation.create(               "pA",           NamedQuantity.CURRENT)
    fA = NamedUnitInformation.create(               "fA",           NamedQuantity.CURRENT)

    K = NamedUnitInformation.create(                "K",            NamedQuantity.TEMPERATURE)
    mK = NamedUnitInformation.create(               "mK",           NamedQuantity.TEMPERATURE)
    µK = NamedUnitInformation.create(               "µK",           NamedQuantity.TEMPERATURE)
    nK = NamedUnitInformation.create(               "nK",           NamedQuantity.TEMPERATURE)
    pK = NamedUnitInformation.create(               "pK",           NamedQuantity.TEMPERATURE)
    fK = NamedUnitInformation.create(               "fK",           NamedQuantity.TEMPERATURE)

    mol = NamedUnitInformation.create(              "mol",          NamedQuantity.AMOUNT_OF_SUBSTANCE)
    mmol = NamedUnitInformation.create(             "mmol",         NamedQuantity.AMOUNT_OF_SUBSTANCE)
    µmol = NamedUnitInformation.create(             "µmol",         NamedQuantity.AMOUNT_OF_SUBSTANCE)
    nmol = NamedUnitInformation.create(             "nmol",         NamedQuantity.AMOUNT_OF_SUBSTANCE)
    pmol = NamedUnitInformation.create(             "pmol",         NamedQuantity.AMOUNT_OF_SUBSTANCE)
    fmol = NamedUnitInformation.create(             "fmol",         NamedQuantity.AMOUNT_OF_SUBSTANCE)

    cd = NamedUnitInformation.create(               "cd",           NamedQuantity.LUMINOUS_INTENSITY)
    mcd = NamedUnitInformation.create(              "mcd",          NamedQuantity.LUMINOUS_INTENSITY)
    µcd = NamedUnitInformation.create(              "µcd",          NamedQuantity.LUMINOUS_INTENSITY)
    ncd = NamedUnitInformation.create(              "ncd",          NamedQuantity.LUMINOUS_INTENSITY)
    pcd = NamedUnitInformation.create(              "pcd",          NamedQuantity.LUMINOUS_INTENSITY)
    fcd = NamedUnitInformation.create(              "fcd",          NamedQuantity.LUMINOUS_INTENSITY)

    N = NamedUnitInformation.create(                "N",            NamedQuantity.FORCE)
    mN = NamedUnitInformation.create(               "mN",           NamedQuantity.FORCE)
    µN = NamedUnitInformation.create(               "µN",           NamedQuantity.FORCE)
    nN = NamedUnitInformation.create(               "nN",           NamedQuantity.FORCE)
    pN = NamedUnitInformation.create(               "pN",           NamedQuantity.FORCE)
    fN = NamedUnitInformation.create(               "fN",           NamedQuantity.FORCE)

    J = NamedUnitInformation.create(                "J",            NamedQuantity.ENERGY)
    mJ = NamedUnitInformation.create(               "mJ",           NamedQuantity.ENERGY)
    µJ = NamedUnitInformation.create(               "µJ",           NamedQuantity.ENERGY)
    nJ = NamedUnitInformation.create(               "nJ",           NamedQuantity.ENERGY)
    pJ = NamedUnitInformation.create(               "pJ",           NamedQuantity.ENERGY)
    fJ = NamedUnitInformation.create(               "fJ",           NamedQuantity.ENERGY)

    W = NamedUnitInformation.create(                "W",            NamedQuantity.POWER)
    mW = NamedUnitInformation.create(               "mW",           NamedQuantity.POWER)
    µW = NamedUnitInformation.create(               "µW",           NamedQuantity.POWER)
    nW = NamedUnitInformation.create(               "nW",           NamedQuantity.POWER)
    pW = NamedUnitInformation.create(               "pW",           NamedQuantity.POWER)
    fW = NamedUnitInformation.create(               "fW",           NamedQuantity.POWER)

    Pa = NamedUnitInformation.create(               "Pa",           NamedQuantity.PRESSURE)
    mPa = NamedUnitInformation.create(              "mPa",          NamedQuantity.PRESSURE)
    µPa = NamedUnitInformation.create(              "µPa",          NamedQuantity.PRESSURE)
    nPa = NamedUnitInformation.create(              "nPa",          NamedQuantity.PRESSURE)
    pPa = NamedUnitInformation.create(              "pPa",          NamedQuantity.PRESSURE)
    fPa = NamedUnitInformation.create(              "fPa",          NamedQuantity.PRESSURE)

    V = NamedUnitInformation.create(                "V",            NamedQuantity.VOLTAGE)
    mV = NamedUnitInformation.create(               "mV",           NamedQuantity.VOLTAGE)
    µV = NamedUnitInformation.create(               "µV",           NamedQuantity.VOLTAGE)
    nV = NamedUnitInformation.create(               "nV",           NamedQuantity.VOLTAGE)
    pV = NamedUnitInformation.create(               "pV",           NamedQuantity.VOLTAGE)
    fV = NamedUnitInformation.create(               "fV",           NamedQuantity.VOLTAGE)

    Hz = NamedUnitInformation.create(               "Hz",           NamedQuantity.FREQUENCY)
    mHz = NamedUnitInformation.create(              "mHz",          NamedQuantity.FREQUENCY)
    µHz = NamedUnitInformation.create(              "µHz",          NamedQuantity.FREQUENCY)
    nHz = NamedUnitInformation.create(              "nHz",          NamedQuantity.FREQUENCY)
    pHz = NamedUnitInformation.create(              "pHz",          NamedQuantity.FREQUENCY)
    fHz = NamedUnitInformation.create(              "fHz",          NamedQuantity.FREQUENCY)

    @property
    def unit_string(self) -> str:
        return self.value.unit_string

    @property
    def named_quantity(self) -> NamedQuantity:
        return self.value.named_quantity

    @classmethod
    def find_named_unit(cls, unit_string: str) -> "NamedUnit":
        for named_unit in cls:
            if named_unit.value.unit_string == unit_string:
                return named_unit
        raise ValueError(f"Unknown unit string: {unit_string}")

    @classmethod
    def list_unit_strings(cls) -> list[str]:
        return [named_unit.value.unit_string for named_unit in cls]
