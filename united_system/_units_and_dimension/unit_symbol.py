from dataclasses import dataclass, field
from typing import Union, TYPE_CHECKING
from .named_quantity import NamedQuantity
from enum import Enum
import math

if TYPE_CHECKING:
    from .dimension import Dimension

class UNIT_SYMBOL_TAG(Enum):
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
    named_quantity: NamedQuantity
    symbol_tags: set[UNIT_SYMBOL_TAG]
    factor: float = 1.0
    offset: float = 0.0

    @classmethod
    def create(cls, name: str, symbol: Union[str, list[str]], named_quantity: NamedQuantity, symbol_tags: set[UNIT_SYMBOL_TAG], factor: float=1, offset: float=0) -> "UnitSymbolInformation":
        return cls(name, [symbol] if isinstance(symbol, str) else symbol, named_quantity, symbol_tags, factor, offset)

class UnitSymbol(Enum):
    value: UnitSymbolInformation # type: ignore

    ANGSTROM = UnitSymbolInformation.create(              "angstrom",              "Å",                 NamedQuantity.LENGTH,                  symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT},     factor=1e-10)
    ACRE = UnitSymbolInformation.create(                  "acre",                  "ac",                NamedQuantity.AREA,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM},  factor=4046.86)
    AMPERE = UnitSymbolInformation.create(                "ampere",                "A",                 NamedQuantity.CURRENT,                 symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT})
    ANGLE = UnitSymbolInformation.create(                 "angle",                 "°",                 NamedQuantity.ANGLE,                   symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=1/180*math.pi)
    ATMOSPHERE = UnitSymbolInformation.create(            "atmosphere",            "atm",               NamedQuantity.PRESSURE,                symbol_tags={UNIT_SYMBOL_TAG.US_CUSTOMARY_SYSTEM},  factor=101325)
    BAR = UnitSymbolInformation.create(                   "bar",                   "bar",               NamedQuantity.PRESSURE,                symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=100000)
    BARREL = UnitSymbolInformation.create(                "barrel",                "bbl",               NamedQuantity.VOLUME,                  symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=0.158987)
    BECKQUEREL = UnitSymbolInformation.create(            "becquerel",             "Bq",                NamedQuantity.ACTIVITY,                symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT})
    CANDLELA = UnitSymbolInformation.create(              "candela",               "cd",                NamedQuantity.LUMINOUS_INTENSITY,      symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    COULOMB = UnitSymbolInformation.create(               "coulomb",               "C",                 NamedQuantity.CHARGE,                  symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    DAY = UnitSymbolInformation.create(                   "day",                   "days",              NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM}, factor=86400)
    DEGREE = UnitSymbolInformation.create(                "degree",                "°",                 NamedQuantity.ANGLE,                   symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=1/180*math.pi)
    DEGREE_CELSIUS = UnitSymbolInformation.create(        "degree celsius",        "°C",                NamedQuantity.TEMPERATURE,             symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, offset=273.15)
    TESLA = UnitSymbolInformation.create(                 "tesla",                 "T",                 NamedQuantity.MAGNETIC_FLUX_DENSITY,   symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    DALTON = UnitSymbolInformation.create(                "dalton",                "Da",                NamedQuantity.MASS,                    symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=1.660539040e-27)
    ELECTRONVOLT = UnitSymbolInformation.create(          "electronvolt",          "eV",                NamedQuantity.ENERGY,                  symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=1.602176634e-19)
    DYNAMIC_VISCOSITY = UnitSymbolInformation.create(     "dynamic viscosity",     "η",                 NamedQuantity.DYNAMIC_VISCOSITY,       symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    FARAD = UnitSymbolInformation.create(                 "farad",                 "F",                 NamedQuantity.CAPACITANCE,             symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    FAHRENHEIT = UnitSymbolInformation.create(            "fahrenheit",            "°F",                NamedQuantity.TEMPERATURE,             symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=5/9, offset=459.67)
    GALLON = UnitSymbolInformation.create(                "gallon",                "gal",               NamedQuantity.VOLUME,                  symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=0.00378541)
    GRAM = UnitSymbolInformation.create(                  "gram",                  "g",                 NamedQuantity.MASS,                    symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT},    factor=0.001)
    GRAY = UnitSymbolInformation.create(                  "gray",                  "Gy",                NamedQuantity.DOSE_ABSORTED,           symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    HENRY = UnitSymbolInformation.create(                 "henry",                 "H",                 NamedQuantity.INDUCTANCE,              symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    HERTZ = UnitSymbolInformation.create(                 "hertz",                 "Hz",                NamedQuantity.FREQUENCY,               symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    HOUR = UnitSymbolInformation.create(                  "hour",                  "h",                 NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM}, factor=3600)
    INCH = UnitSymbolInformation.create(                  "inch",                  "in",                NamedQuantity.LENGTH,                  symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=0.0254)
    JOULE = UnitSymbolInformation.create(                 "joule",                 "J",                 NamedQuantity.ENERGY,                  symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    KELVIN = UnitSymbolInformation.create(                "kelvin",                "K",                 NamedQuantity.TEMPERATURE,             symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT})
    METER = UnitSymbolInformation.create(                 "meter",                 "m",                 NamedQuantity.LENGTH,                  symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT})
    MILLIMETER_OF_MERCURY = UnitSymbolInformation.create( "millimeter of mercury", "mmHg",              NamedQuantity.PRESSURE,                symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=133.322)
    MINUTE = UnitSymbolInformation.create(                "minute",                "min",               NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM}, factor=60)
    MOLE = UnitSymbolInformation.create(                  "mole",                  "mol",               NamedQuantity.AMOUNT_OF_SUBSTANCE,     symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    NEPER = UnitSymbolInformation.create(                 "neper",                 "Np",                NamedQuantity.LOGLEVEL,                symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT_EQUIVALENT})
    NEWTON = UnitSymbolInformation.create(                "newton",                "N",                 NamedQuantity.FORCE,                   symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    KINEMATIC_VISCOSITY = UnitSymbolInformation.create(   "kinematic viscosity",   "ν",                 NamedQuantity.KINEMATIC_VISCOSITY,     symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    OHM = UnitSymbolInformation.create(                   "ohm",                   ["Ω", "Ω", "Ohm"],   NamedQuantity.RESISTANCE,              symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    PASCAL = UnitSymbolInformation.create(                "pascal",                "Pa",                NamedQuantity.PRESSURE,                symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    PSI = UnitSymbolInformation.create(                   "psi",                   "psi",               NamedQuantity.PRESSURE,                symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=6894.76)
    POUND = UnitSymbolInformation.create(                 "pound",                 "lb",                NamedQuantity.MASS,                    symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=0.453592)
    RADIAN = UnitSymbolInformation.create(                "radian",                "rad",               NamedQuantity.ANGLE,                   symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT_EQUIVALENT})
    RANKINE = UnitSymbolInformation.create(               "rankine",               "°R",                NamedQuantity.TEMPERATURE,             symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=5/9)
    SECOND = UnitSymbolInformation.create(                "second",                "s",                 NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.SI_BASE_UNIT})
    SIEMENS = UnitSymbolInformation.create(               "siemens",               "S",                 NamedQuantity.CONDUCTANCE,             symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    SIEVERT = UnitSymbolInformation.create(               "sievert",               "Sv",                NamedQuantity.DOSE_EQUIVALENT,         symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    STONE = UnitSymbolInformation.create(                 "stone",                 "st",                NamedQuantity.MASS,                    symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=6.35029)
    TONNE = UnitSymbolInformation.create(                 "tonne",                 "t",                 NamedQuantity.MASS,                    symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=1000)
    TORR = UnitSymbolInformation.create(                  "torr",                  "torr",              NamedQuantity.PRESSURE,                symbol_tags={UNIT_SYMBOL_TAG.US_CUSTOMARY_SYSTEM}, factor=133.322)
    TURN = UnitSymbolInformation.create(                  "turn",                  "tr",                NamedQuantity.ANGLE,                   symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT}, factor=2*math.pi)
    VOLT = UnitSymbolInformation.create(                  "volt",                  "V",                 NamedQuantity.VOLTAGE,                 symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    WEBER = UnitSymbolInformation.create(                 "weber",                 "Wb",                NamedQuantity.MAGNETIC_FLUX,           symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    WATT = UnitSymbolInformation.create(                  "watt",                  "W",                 NamedQuantity.POWER,                   symbol_tags={UNIT_SYMBOL_TAG.SI_DERIVED_UNIT})
    WEEK = UnitSymbolInformation.create(                  "week",                  "weeks",             NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM}, factor=604800)
    MONTH = UnitSymbolInformation.create(                 "month",                 "months",            NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM}, factor=2629746)
    YEAR = UnitSymbolInformation.create(                  "year",                  "years",             NamedQuantity.TIME,                    symbol_tags={UNIT_SYMBOL_TAG.NON_SI_SYSTEM}, factor=31556952)

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
    def named_quantity(self) -> NamedQuantity:
        return self.value.named_quantity
    
    @property
    def default_unit_symbol_string(self) -> str:
        return self.value.symbols[0]
    
    @property
    def dimension(self) -> "Dimension":
        if not hasattr(self, '_dimension'):
            from .dimension import Dimension
            self._dimension: Dimension = self.named_quantity.dimension
        return self._dimension
    
    @property
    def proper_exponents(self) -> tuple[float, float, float, float, float, float, float, float]:
        if not hasattr(self, '_proper_exponents'):
            from .proper_exponents import ProperExponents
            proper_exponents: ProperExponents = self.named_quantity.value[1]
            self._proper_exponents: tuple[float, float, float, float, float, float, float, float] = proper_exponents.proper_exponents
        return self._proper_exponents

ANGLE_UNIT_SYMBOLS: set[UnitSymbol] = set()

for unit_symbol in UnitSymbol:
    if unit_symbol.value.named_quantity == NamedQuantity.ANGLE:
        ANGLE_UNIT_SYMBOLS.add(unit_symbol)

def is_angle_unit_symbol(unit_symbol: UnitSymbol) -> bool:
    return unit_symbol in ANGLE_UNIT_SYMBOLS

@dataclass(frozen=True, slots=True)
class LogDimensionSymbolInformation:
    name: str
    symbols: list[str]
    factor: float = field(default=1.0)
    offset: float = field(default=0.0)

    @classmethod
    def create(cls, name: str, symbol: Union[str, list[str]], factor: float=1) -> "LogDimensionSymbolInformation":
        return cls(name, [symbol] if isinstance(symbol, str) else symbol, factor)

class LogDimensionSymbol(Enum):
    value: LogDimensionSymbolInformation # type: ignore

    BASE_10 = LogDimensionSymbolInformation.create(name="base 10", symbol=["dec", "decade", "mag", "magnitude"], factor=math.log(10))
    BASE_2 = LogDimensionSymbolInformation.create(name="base 2", symbol="bin", factor=math.log(2))
    BASE_E = LogDimensionSymbolInformation.create(name="base e", symbol="nat", factor=1.0)

    @property
    def default_unit_symbol_string(self) -> str:
        return self.value.symbols[0]

# Export log unit symbols for convenience
LOG_UNIT_SYMBOLS = LogDimensionSymbol

BASE_10_LOG_UNIT_SYMBOL: LogDimensionSymbol = LogDimensionSymbol.BASE_10