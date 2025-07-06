from enum import Enum
from dataclasses import dataclass, field
from .named_simple_dimensions import NamedSimpleDimension
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .simple.simple_unit import SimpleUnit

FORBIDDEN_STRINGS_IN_UNIT_NAME: set[str] = {"|", ":", "__"}

def special_character_to_unicode_replacement(string: str) -> str:
    string = string.replace("Ω", "Ω")
    string = string.replace("µ", "μ")
    return string

@dataclass(frozen=True, slots=True)
class NamedUnitInformation:
    unit_string: str
    named_simple_unit_dimension: NamedSimpleDimension = field(repr=False, compare=False, hash=False)
    _unit: "SimpleUnit|None" = field(default=None, repr=False, compare=False, hash=False)

    @classmethod
    def create(cls, unit_string: str, named_simple_unit_dimension: NamedSimpleDimension) -> "NamedUnitInformation":
        return cls(unit_string, named_simple_unit_dimension)

    @property
    def unit(self) -> "SimpleUnit":
        if self._unit is None:
            from .simple.simple_unit import SimpleUnit
            unit = SimpleUnit.parse_string(self.unit_string)
            if unit.dimension != self.named_simple_unit_dimension.simple_dimension:
                raise ValueError(f"Unit {self.unit_string} has canonical dimension {unit.dimension} but expected {self.named_simple_unit_dimension.simple_dimension}")
            object.__setattr__(self, '_unit', unit)
        return self._unit

class NamedUnit(Enum):
    value: NamedUnitInformation

    s = NamedUnitInformation.create(                "s",            NamedSimpleDimension.TIME)
    ms = NamedUnitInformation.create(               "ms",           NamedSimpleDimension.TIME)
    µs = NamedUnitInformation.create(               "µs",           NamedSimpleDimension.TIME)
    ns = NamedUnitInformation.create(               "ns",           NamedSimpleDimension.TIME)
    ps = NamedUnitInformation.create(               "ps",           NamedSimpleDimension.TIME)
    fs = NamedUnitInformation.create(               "fs",           NamedSimpleDimension.TIME)

    min = NamedUnitInformation.create(              "min",          NamedSimpleDimension.TIME)
    h = NamedUnitInformation.create(                "h",            NamedSimpleDimension.TIME)
    d = NamedUnitInformation.create(                "days",         NamedSimpleDimension.TIME)
    w = NamedUnitInformation.create(                "weeks",        NamedSimpleDimension.TIME)
    mo = NamedUnitInformation.create(               "months",       NamedSimpleDimension.TIME)
    y = NamedUnitInformation.create(                "years",        NamedSimpleDimension.TIME)

    m = NamedUnitInformation.create(                "m",            NamedSimpleDimension.LENGTH)
    km = NamedUnitInformation.create(               "km",           NamedSimpleDimension.LENGTH)
    mm = NamedUnitInformation.create(               "mm",           NamedSimpleDimension.LENGTH)
    µm = NamedUnitInformation.create(               "µm",           NamedSimpleDimension.LENGTH)
    nm = NamedUnitInformation.create(               "nm",           NamedSimpleDimension.LENGTH)
    pm = NamedUnitInformation.create(               "pm",           NamedSimpleDimension.LENGTH)
    fm = NamedUnitInformation.create(               "fm",           NamedSimpleDimension.LENGTH)

    kg = NamedUnitInformation.create(               "kg",           NamedSimpleDimension.MASS)
    g = NamedUnitInformation.create(                "g",            NamedSimpleDimension.MASS)
    mg = NamedUnitInformation.create(               "mg",           NamedSimpleDimension.MASS)
    µg = NamedUnitInformation.create(               "µg",           NamedSimpleDimension.MASS)
    ng = NamedUnitInformation.create(               "ng",           NamedSimpleDimension.MASS)
    pg = NamedUnitInformation.create(               "pg",           NamedSimpleDimension.MASS)
    fg = NamedUnitInformation.create(               "fg",           NamedSimpleDimension.MASS)

    A = NamedUnitInformation.create(                "A",            NamedSimpleDimension.CURRENT)
    mA = NamedUnitInformation.create(               "mA",           NamedSimpleDimension.CURRENT)
    µA = NamedUnitInformation.create(               "µA",           NamedSimpleDimension.CURRENT)
    nA = NamedUnitInformation.create(               "nA",           NamedSimpleDimension.CURRENT)
    pA = NamedUnitInformation.create(               "pA",           NamedSimpleDimension.CURRENT)
    fA = NamedUnitInformation.create(               "fA",           NamedSimpleDimension.CURRENT)

    K = NamedUnitInformation.create(                "K",            NamedSimpleDimension.TEMPERATURE)
    mK = NamedUnitInformation.create(               "mK",           NamedSimpleDimension.TEMPERATURE)
    µK = NamedUnitInformation.create(               "µK",           NamedSimpleDimension.TEMPERATURE)
    nK = NamedUnitInformation.create(               "nK",           NamedSimpleDimension.TEMPERATURE)
    pK = NamedUnitInformation.create(               "pK",           NamedSimpleDimension.TEMPERATURE)
    fK = NamedUnitInformation.create(               "fK",           NamedSimpleDimension.TEMPERATURE)

    mol = NamedUnitInformation.create(              "mol",          NamedSimpleDimension.AMOUNT_OF_SUBSTANCE)
    mmol = NamedUnitInformation.create(             "mmol",         NamedSimpleDimension.AMOUNT_OF_SUBSTANCE)
    µmol = NamedUnitInformation.create(             "µmol",         NamedSimpleDimension.AMOUNT_OF_SUBSTANCE)
    nmol = NamedUnitInformation.create(             "nmol",         NamedSimpleDimension.AMOUNT_OF_SUBSTANCE)
    pmol = NamedUnitInformation.create(             "pmol",         NamedSimpleDimension.AMOUNT_OF_SUBSTANCE)
    fmol = NamedUnitInformation.create(             "fmol",         NamedSimpleDimension.AMOUNT_OF_SUBSTANCE)

    cd = NamedUnitInformation.create(               "cd",           NamedSimpleDimension.LUMINOUS_INTENSITY)
    mcd = NamedUnitInformation.create(              "mcd",          NamedSimpleDimension.LUMINOUS_INTENSITY)
    µcd = NamedUnitInformation.create(              "µcd",          NamedSimpleDimension.LUMINOUS_INTENSITY)
    ncd = NamedUnitInformation.create(              "ncd",          NamedSimpleDimension.LUMINOUS_INTENSITY)
    pcd = NamedUnitInformation.create(              "pcd",          NamedSimpleDimension.LUMINOUS_INTENSITY)
    fcd = NamedUnitInformation.create(              "fcd",          NamedSimpleDimension.LUMINOUS_INTENSITY)

    N = NamedUnitInformation.create(                "N",            NamedSimpleDimension.FORCE)
    mN = NamedUnitInformation.create(               "mN",           NamedSimpleDimension.FORCE)
    µN = NamedUnitInformation.create(               "µN",           NamedSimpleDimension.FORCE)
    nN = NamedUnitInformation.create(               "nN",           NamedSimpleDimension.FORCE)
    pN = NamedUnitInformation.create(               "pN",           NamedSimpleDimension.FORCE)
    fN = NamedUnitInformation.create(               "fN",           NamedSimpleDimension.FORCE)

    J = NamedUnitInformation.create(                "J",            NamedSimpleDimension.ENERGY)
    mJ = NamedUnitInformation.create(               "mJ",           NamedSimpleDimension.ENERGY)
    µJ = NamedUnitInformation.create(               "µJ",           NamedSimpleDimension.ENERGY)
    nJ = NamedUnitInformation.create(               "nJ",           NamedSimpleDimension.ENERGY)
    pJ = NamedUnitInformation.create(               "pJ",           NamedSimpleDimension.ENERGY)
    fJ = NamedUnitInformation.create(               "fJ",           NamedSimpleDimension.ENERGY)

    W = NamedUnitInformation.create(                "W",            NamedSimpleDimension.POWER)
    mW = NamedUnitInformation.create(               "mW",           NamedSimpleDimension.POWER)
    µW = NamedUnitInformation.create(               "µW",           NamedSimpleDimension.POWER)
    nW = NamedUnitInformation.create(               "nW",           NamedSimpleDimension.POWER)
    pW = NamedUnitInformation.create(               "pW",           NamedSimpleDimension.POWER)
    fW = NamedUnitInformation.create(               "fW",           NamedSimpleDimension.POWER)

    Pa = NamedUnitInformation.create(               "Pa",           NamedSimpleDimension.PRESSURE)
    mPa = NamedUnitInformation.create(              "mPa",          NamedSimpleDimension.PRESSURE)
    µPa = NamedUnitInformation.create(              "µPa",          NamedSimpleDimension.PRESSURE)
    nPa = NamedUnitInformation.create(              "nPa",          NamedSimpleDimension.PRESSURE)
    pPa = NamedUnitInformation.create(              "pPa",          NamedSimpleDimension.PRESSURE)
    fPa = NamedUnitInformation.create(              "fPa",          NamedSimpleDimension.PRESSURE)

    V = NamedUnitInformation.create(                "V",            NamedSimpleDimension.VOLTAGE)
    mV = NamedUnitInformation.create(               "mV",           NamedSimpleDimension.VOLTAGE)
    µV = NamedUnitInformation.create(               "µV",           NamedSimpleDimension.VOLTAGE)
    nV = NamedUnitInformation.create(               "nV",           NamedSimpleDimension.VOLTAGE)
    pV = NamedUnitInformation.create(               "pV",           NamedSimpleDimension.VOLTAGE)
    fV = NamedUnitInformation.create(               "fV",           NamedSimpleDimension.VOLTAGE)

    Hz = NamedUnitInformation.create(               "Hz",           NamedSimpleDimension.FREQUENCY)
    mHz = NamedUnitInformation.create(              "mHz",          NamedSimpleDimension.FREQUENCY)
    µHz = NamedUnitInformation.create(              "µHz",          NamedSimpleDimension.FREQUENCY)
    nHz = NamedUnitInformation.create(              "nHz",          NamedSimpleDimension.FREQUENCY)
    pHz = NamedUnitInformation.create(              "pHz",          NamedSimpleDimension.FREQUENCY)
    fHz = NamedUnitInformation.create(              "fHz",          NamedSimpleDimension.FREQUENCY)

    @property
    def unit_string(self) -> str:
        return self.value.unit_string

    @property
    def named_simple_unit_dimension(self) -> NamedSimpleDimension:
        return self.value.named_simple_unit_dimension

    @property
    def unit(self) -> "SimpleUnit":
        return self.value.unit

    @classmethod
    def find_named_unit(cls, unit_string: str) -> "NamedUnit":
        for named_unit in cls:
            if named_unit.value.unit_string == unit_string:
                return named_unit
        raise ValueError(f"Unknown unit string: {unit_string}")

    @classmethod
    def list_unit_strings(cls) -> list[str]:
        return [named_unit.value.unit_string for named_unit in cls]
