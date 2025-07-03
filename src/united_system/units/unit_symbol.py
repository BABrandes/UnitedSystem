from typing import Final
from dataclasses import dataclass   
from enum import Enum
import math
from .unit_quantity import SimpleCanonicalQuantity
from .named_canonical_quantities import *

class UnitSymbolTag(Enum):
    SI_BASE_UNIT = "SI Base Unit"
    SI_DERIVED_UNIT = "SI Derived Unit"
    NON_SI_SYSTEM = "Non-SI System"
    US_CUSTOMARY_SYSTEM = "US Customary System"
    SI_BASE_UNIT_EQUIVALENT = "SI Base Unit Equivalent"

@dataclass(frozen=True, slots=True)
class UnitSymbolInformation():
    name: str
    symbol: list[str]
    named_canonical_quantity: NamedCanonicalQuantity
    symbol_tags: set[UnitSymbolTag]
    factor: float
    offset: float

    @classmethod
    def create(cls, name: str, symbol: str|list[str], named_canonical_quantity: NamedCanonicalQuantity, symbol_tags: set[UnitSymbolTag], factor: float=1, offset: float=0) -> "UnitSymbolInformation":
        return cls(name, list(symbol) if isinstance(symbol, str) else symbol, named_canonical_quantity, symbol_tags, factor, offset)

class UnitSymbol(Enum):
    value: UnitSymbolInformation

    ANGSTROM = UnitSymbolInformation.create(              "angstrom",              "Å",            NamedCanonicalQuantity.LENGTH,                  symbol_tags={UnitSymbolTag.SI_BASE_UNIT},     factor=1e-10)
    ACRE = UnitSymbolInformation.create(                  "acre",                  "ac",           NamedCanonicalQuantity.AREA,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM},  factor=4046.86)
    AMPERE = UnitSymbolInformation.create(                "ampere",                "A",            NamedCanonicalQuantity.CURRENT,                 symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    ANGLE = UnitSymbolInformation.create(                 "angle",                 "°",            NamedCanonicalQuantity.ANGLE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1/180*math.pi)
    ATMOSPHERE = UnitSymbolInformation.create(            "atmosphere",            "atm",          NamedCanonicalQuantity.PRESSURE,                symbol_tags={UnitSymbolTag.US_CUSTOMARY_SYSTEM},  factor=101325)
    BAR = UnitSymbolInformation.create(                   "bar",                   "bar",          NamedCanonicalQuantity.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=100000)
    BARREL = UnitSymbolInformation.create(                "barrel",                "bbl",          NamedCanonicalQuantity.VOLUME,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.158987)
    BECKQUEREL = UnitSymbolInformation.create(            "becquerel",             "Bq",           NamedCanonicalQuantity.ACTIVITY,                symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    CANDLELA = UnitSymbolInformation.create(              "candela",               "cd",           NamedCanonicalQuantity.LUMINOUS_INTENSITY,      symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    COULOMB = UnitSymbolInformation.create(               "coulomb",               "C",            NamedCanonicalQuantity.CHARGE,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    DECADE = UnitSymbolInformation.create(                "decade",                "dec",          NamedCanonicalQuantity.TIME,                    symbol_tags={UnitSymbolTag.NON_SI_SYSTEM}, factor=math.log10(10))
    DEGREE = UnitSymbolInformation.create(                "degree",                "°",            NamedCanonicalQuantity.ANGLE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1/180*math.pi)
    DEGREE_CELSIUS = UnitSymbolInformation.create(        "degree celsius",        "°C",           NamedCanonicalQuantity.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, offset=273.15)
    TESLA = UnitSymbolInformation.create(                 "tesla",                 "B",            NamedCanonicalQuantity.MAGNETIC_FLUX_DENSITY,   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    DALTON = UnitSymbolInformation.create(                "dalton",                "Da",           NamedCanonicalQuantity.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1.660539040e-27)
    ELECTRONVOLT = UnitSymbolInformation.create(          "electronvolt",          "eV",           NamedCanonicalQuantity.ENERGY,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1.602176634e-19)
    DYNAMIC_VISCOSITY = UnitSymbolInformation.create(     "dynamic viscosity",     "η",            NamedCanonicalQuantity.DYNAMIC_VISCOSITY,       symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    FARAD = UnitSymbolInformation.create(                 "farad",                 "F",            NamedCanonicalQuantity.CAPACITANCE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    FAHRENHEIT = UnitSymbolInformation.create(            "fahrenheit",            "°F",           NamedCanonicalQuantity.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=5/9, offset=459.67)
    GALLON = UnitSymbolInformation.create(                "gallon",                "gal",          NamedCanonicalQuantity.VOLUME,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.00378541)
    GRAM = UnitSymbolInformation.create(                  "gram",                  "g",            NamedCanonicalQuantity.MASS,                    symbol_tags={UnitSymbolTag.SI_BASE_UNIT},    factor=0.001)
    GRAY = UnitSymbolInformation.create(                  "gray",                  "Gy",           NamedCanonicalQuantity.DOSE_ABSORTED,           symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    HERTZ = UnitSymbolInformation.create(                 "hertz",                 "Hz",           NamedCanonicalQuantity.FREQUENCY,               symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    INCH = UnitSymbolInformation.create(                  "inch",                  "in",           NamedCanonicalQuantity.LENGTH,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.0254)
    JOULE = UnitSymbolInformation.create(                 "joule",                 "J",            NamedCanonicalQuantity.ENERGY,                  symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    KELVIN = UnitSymbolInformation.create(                "kelvin",                "K",            NamedCanonicalQuantity.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    METER = UnitSymbolInformation.create(                 "meter",                 "m",            NamedCanonicalQuantity.LENGTH,                  symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    MILLIMETER = UnitSymbolInformation.create(            "millimeter",            "mm",           NamedCanonicalQuantity.LENGTH,                  symbol_tags={UnitSymbolTag.SI_BASE_UNIT}, factor=0.001)
    MILLIMETER_OF_MERCURY = UnitSymbolInformation.create( "millimeter of mercury", "mmHg",         NamedCanonicalQuantity.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=133.322)
    MOLE = UnitSymbolInformation.create(                  "mole",                  "mol",          NamedCanonicalQuantity.AMOUNT_OF_SUBSTANCE,     symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    NEPER = UnitSymbolInformation.create(                 "neper",                 "Np",           NamedCanonicalQuantity.LOGLEVEL,                symbol_tags={UnitSymbolTag.SI_BASE_UNIT_EQUIVALENT})
    NEWTON = UnitSymbolInformation.create(                "newton",                "N",            NamedCanonicalQuantity.FORCE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    KINEMATIC_VISCOSITY = UnitSymbolInformation.create(   "kinematic viscosity",   "ν",            NamedCanonicalQuantity.KINEMATIC_VISCOSITY,     symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    OHM = UnitSymbolInformation.create(                   "ohm",                   ["Ω", "Ohm"],   NamedCanonicalQuantity.RESISTANCE,              symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    PASCAL = UnitSymbolInformation.create(                "pascal",                "Pa",           NamedCanonicalQuantity.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    PSI = UnitSymbolInformation.create(                   "psi",                   "psi",          NamedCanonicalQuantity.PRESSURE,                symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=6894.76)
    POUND = UnitSymbolInformation.create(                 "pound",                 "lb",           NamedCanonicalQuantity.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=0.453592)
    RADIAN = UnitSymbolInformation.create(                "radian",                "rad",          NamedCanonicalQuantity.ANGLE,                   symbol_tags={UnitSymbolTag.SI_BASE_UNIT_EQUIVALENT})
    RANKINE = UnitSymbolInformation.create(               "rankine",               "°R",           NamedCanonicalQuantity.TEMPERATURE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=5/9)
    SECOND = UnitSymbolInformation.create(                "second",                "s",            NamedCanonicalQuantity.TIME,                    symbol_tags={UnitSymbolTag.SI_BASE_UNIT})
    SIEMENS = UnitSymbolInformation.create(               "siemens",               "S",            NamedCanonicalQuantity.CONDUCTANCE,             symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    SIEVERT = UnitSymbolInformation.create(               "sievert",               "Sv",           NamedCanonicalQuantity.DOSE_EQUIVALENT,         symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    STONE = UnitSymbolInformation.create(                 "stone",                 "st",           NamedCanonicalQuantity.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=6.35029)
    TONNE = UnitSymbolInformation.create(                 "tonne",                 "t",            NamedCanonicalQuantity.MASS,                    symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=1000)
    TORR = UnitSymbolInformation.create(                  "torr",                  "torr",         NamedCanonicalQuantity.PRESSURE,                symbol_tags={UnitSymbolTag.US_CUSTOMARY_SYSTEM}, factor=133.322)
    TURN = UnitSymbolInformation.create(                  "turn",                  "tr",           NamedCanonicalQuantity.ANGLE,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT}, factor=2*math.pi)
    VOLT = UnitSymbolInformation.create(                  "volt",                  "V",            NamedCanonicalQuantity.VOLTAGE,                 symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})
    WATT = UnitSymbolInformation.create(                  "watt",                  "W",            NamedCanonicalQuantity.POWER,                   symbol_tags={UnitSymbolTag.SI_DERIVED_UNIT})

    def has_unit_symbol(self, symbol: str) -> bool:
        return not self.find_unit_symbol(symbol) == None

    def __getitem__(self, symbol: str) -> "UnitSymbol":
        result: UnitSymbol|None = self.find_unit_symbol(symbol)
        if result is None:
            raise ValueError(f"Unit symbol {symbol} not found")
        return result

    @property
    def symbol(self) -> str:
        return self.value.symbol[0]
    
    @property
    def symbols(self) -> list[str]:
        return self.value.symbol
    
    @property
    def symbol_information(self) -> UnitSymbolInformation:
        return self.value
    
    @property
    def name(self) -> str:
        return self.value.name
    
    @property
    def named_canonical_quantity(self) -> NamedCanonicalQuantity:
        return self.value.named_canonical_quantity
    
    @property
    def symbol_tags(self) -> set[UnitSymbolTag]:
        return self.value.symbol_tags
    
    @property
    def factor(self) -> float:
        return self.value.factor
    
    @property
    def offset(self) -> float:
        return self.value.offset
    
    @staticmethod
    def find_unit_symbol(symbol: str) -> "UnitSymbol|None":
        for unit_symbol in UnitSymbol:
            if symbol in unit_symbol.value.symbol:
                return unit_symbol
        return None