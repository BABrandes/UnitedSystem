from dataclasses import dataclass
from typing import Union
from enum import Enum
from .named_simple_dimensions import NamedSimpleDimension
import math

# No longer needed - we'll include all Unicode variants directly in symbol definitions

class UnitSymbolTag(Enum):
    SI_BASE_UNIT = "SI_BASE_UNIT"
    SI_DERIVED_UNIT = "SI_DERIVED_UNIT"
    SI_BASE_UNIT_EQUIVALENT = "SI_BASE_UNIT_EQUIVALENT"
    NON_SI_SYSTEM = "NON_SI_SYSTEM"
    US_CUSTOMARY_SYSTEM = "US_CUSTOMARY_SYSTEM"
    IMPERIAL_SYSTEM = "IMPERIAL_SYSTEM"
    METRIC_SYSTEM = "METRIC_SYSTEM"
    BINARY_SYSTEM = "BINARY_SYSTEM"
    DECI_SYSTEM = "DECI_SYSTEM"
    ENGINEERING_SYSTEM = "ENGINEERING_SYSTEM"
    SCIENTIFIC_SYSTEM = "SCIENTIFIC_SYSTEM"

@dataclass(frozen=True, slots=True)
class UnitSymbolInformation:
    name: str
    symbols: list[str]
    named_simple_unit_dimension: NamedSimpleDimension
    symbol_tags: set[UnitSymbolTag]
    factor: float = 1.0
    offset: float = 0.0

    @classmethod
    def create(cls, name: str, symbol: Union[str, list[str]], named_simple_unit_dimension: NamedSimpleDimension, symbol_tags: set[UnitSymbolTag], factor: float=1, offset: float=0) -> "UnitSymbolInformation":
        return cls(name, [symbol] if isinstance(symbol, str) else symbol, named_simple_unit_dimension, symbol_tags, factor, offset)

class UnitSymbol(Enum):
    value: UnitSymbolInformation

    ANGSTROM = UnitSymbolInformation.create(              "angstrom",              "Å",                 NamedSimpleDimension.LENGTH,                  symbol_tags={UnitSymbolTag.SI_BASE_UNIT},     factor=1e-10)
    ACRE = UnitSymbolInformation.create(                  "acre",                  "ac",                NamedSimpleDimension.AREA,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM},  factor=4046.86)
    AMPERE = UnitSymbolInformation.create(                "ampere",                "A",                 NamedSimpleDimension.CURRENT,                 symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    ANGLE = UnitSymbolInformation.create(                 "angle",                 "°",                 NamedSimpleDimension.ANGLE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1/180*math.pi)
    ATMOSPHERE = UnitSymbolInformation.create(            "atmosphere",            "atm",               NamedSimpleDimension.PRESSURE,                symbol_tags={UnitSymbolTag.US_CUSTOMARY_SYSTEM},  factor=101325)
    BAR = UnitSymbolInformation.create(                   "bar",                   "bar",               NamedSimpleDimension.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=100000)
    BARREL = UnitSymbolInformation.create(                "barrel",                "bbl",               NamedSimpleDimension.VOLUME,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.158987)
    BECKQUEREL = UnitSymbolInformation.create(            "becquerel",             "Bq",                NamedSimpleDimension.ACTIVITY,                symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    CANDLELA = UnitSymbolInformation.create(              "candela",               "cd",                NamedSimpleDimension.LUMINOUS_INTENSITY,      symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    COULOMB = UnitSymbolInformation.create(               "coulomb",               "C",                 NamedSimpleDimension.CHARGE,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    DAY = UnitSymbolInformation.create(                   "day",                   "days",              NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=86400)
    DECADE = UnitSymbolInformation.create(                "decade",                "dec",               NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=math.log10(10))
    DEGREE = UnitSymbolInformation.create(                "degree",                "°",                 NamedSimpleDimension.ANGLE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1/180*math.pi)
    DEGREE_CELSIUS = UnitSymbolInformation.create(        "degree celsius",        "°C",                NamedSimpleDimension.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, offset=273.15)
    TESLA = UnitSymbolInformation.create(                 "tesla",                 "T",                 NamedSimpleDimension.MAGNETIC_FLUX_DENSITY,   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    DALTON = UnitSymbolInformation.create(                "dalton",                "Da",                NamedSimpleDimension.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1.660539040e-27)
    ELECTRONVOLT = UnitSymbolInformation.create(          "electronvolt",          "eV",                NamedSimpleDimension.ENERGY,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1.602176634e-19)
    DYNAMIC_VISCOSITY = UnitSymbolInformation.create(     "dynamic viscosity",     "η",                 NamedSimpleDimension.DYNAMIC_VISCOSITY,       symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    FARAD = UnitSymbolInformation.create(                 "farad",                 "F",                 NamedSimpleDimension.CAPACITANCE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    FAHRENHEIT = UnitSymbolInformation.create(            "fahrenheit",            "°F",                NamedSimpleDimension.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=5/9, offset=459.67)
    GALLON = UnitSymbolInformation.create(                "gallon",                "gal",               NamedSimpleDimension.VOLUME,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.00378541)
    GRAM = UnitSymbolInformation.create(                  "gram",                  "g",                 NamedSimpleDimension.MASS,                    symbol_tags={UnitSymbolTag.SI_BASE_UNIT},    factor=0.001)
    GRAY = UnitSymbolInformation.create(                  "gray",                  "Gy",                NamedSimpleDimension.DOSE_ABSORTED,           symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    HERTZ = UnitSymbolInformation.create(                 "hertz",                 "Hz",                NamedSimpleDimension.FREQUENCY,               symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    HOUR = UnitSymbolInformation.create(                  "hour",                  "h",                 NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=3600)
    INCH = UnitSymbolInformation.create(                  "inch",                  "in",                NamedSimpleDimension.LENGTH,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.0254)
    JOULE = UnitSymbolInformation.create(                 "joule",                 "J",                 NamedSimpleDimension.ENERGY,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    KELVIN = UnitSymbolInformation.create(                "kelvin",                "K",                 NamedSimpleDimension.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    METER = UnitSymbolInformation.create(                 "meter",                 "m",                 NamedSimpleDimension.LENGTH,                  symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    MILLIMETER_OF_MERCURY = UnitSymbolInformation.create( "millimeter of mercury", "mmHg",              NamedSimpleDimension.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=133.322)
    MINUTE = UnitSymbolInformation.create(                "minute",                "min",               NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=60)
    MOLE = UnitSymbolInformation.create(                  "mole",                  "mol",               NamedSimpleDimension.AMOUNT_OF_SUBSTANCE,     symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    NEPER = UnitSymbolInformation.create(                 "neper",                 "Np",                NamedSimpleDimension.LOGLEVEL,                symbol_tags={UnitSymbolTag.SI_BASE_UNIT_EQUIVALENT})
    NEWTON = UnitSymbolInformation.create(                "newton",                "N",                 NamedSimpleDimension.FORCE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    KINEMATIC_VISCOSITY = UnitSymbolInformation.create(   "kinematic viscosity",   "ν",                 NamedSimpleDimension.KINEMATIC_VISCOSITY,     symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    OHM = UnitSymbolInformation.create(                   "ohm",                   ["Ω", "Ω", "Ohm"],   NamedSimpleDimension.RESISTANCE,              symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    PASCAL = UnitSymbolInformation.create(                "pascal",                "Pa",                NamedSimpleDimension.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    PSI = UnitSymbolInformation.create(                   "psi",                   "psi",               NamedSimpleDimension.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=6894.76)
    POUND = UnitSymbolInformation.create(                 "pound",                 "lb",                NamedSimpleDimension.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.453592)
    RADIAN = UnitSymbolInformation.create(                "radian",                "rad",               NamedSimpleDimension.ANGLE,                   symbol_tags={UnitSymbolTag.SI_BASE_UNIT_EQUIVALENT})
    RANKINE = UnitSymbolInformation.create(               "rankine",               "°R",                NamedSimpleDimension.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=5/9)
    SECOND = UnitSymbolInformation.create(                "second",                "s",                 NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    SIEMENS = UnitSymbolInformation.create(               "siemens",               "S",                 NamedSimpleDimension.CONDUCTANCE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    SIEVERT = UnitSymbolInformation.create(               "sievert",               "Sv",                NamedSimpleDimension.DOSE_EQUIVALENT,         symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    STONE = UnitSymbolInformation.create(                 "stone",                 "st",                NamedSimpleDimension.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=6.35029)
    TONNE = UnitSymbolInformation.create(                 "tonne",                 "t",                 NamedSimpleDimension.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1000)
    TORR = UnitSymbolInformation.create(                  "torr",                  "torr",              NamedSimpleDimension.PRESSURE,                symbol_tags={UnitSymbolTag.US_CUSTOMARY_SYSTEM}, factor=133.322)
    TURN = UnitSymbolInformation.create(                  "turn",                  "tr",                NamedSimpleDimension.ANGLE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=2*math.pi)
    VOLT = UnitSymbolInformation.create(                  "volt",                  "V",                 NamedSimpleDimension.VOLTAGE,                 symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    WATT = UnitSymbolInformation.create(                  "watt",                  "W",                 NamedSimpleDimension.POWER,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    WEEK = UnitSymbolInformation.create(                  "week",                  "weeks",             NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=604800)
    MONTH = UnitSymbolInformation.create(                 "month",                 "months",            NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=2629746)
    YEAR = UnitSymbolInformation.create(                  "year",                  "years",             NamedSimpleDimension.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=31556952)

    @classmethod
    def from_symbol(cls, symbol: str) -> "UnitSymbol":
        for unit_symbol in cls:
            if symbol in unit_symbol.value.symbols:
                return unit_symbol
        raise ValueError(f"Unknown unit symbol: {symbol}")

    @classmethod
    def from_name(cls, name: str) -> "UnitSymbol":
        for unit_symbol in cls:
            if unit_symbol.value.name == name:
                return unit_symbol
        raise ValueError(f"Unknown unit name: {name}")

    @property
    def named_simple_unit_dimension(self) -> NamedSimpleDimension:
        return self.value.named_simple_unit_dimension