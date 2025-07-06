from .simple.simple_dimension import SimpleDimension
from dataclasses import dataclass
from .base_classes.base_unit import BaseDimension
from typing import Tuple, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .simple.simple_unit import SimpleUnit

class DimenensionTag(Enum):
    BASE_DIMENSION = "base_unit"

@dataclass(frozen=True, slots=True)
class NamedSimpleDimensionInformation():
    unit_dimension: SimpleDimension
    has_canonical_unit: bool
    canonical_unit_string: str
    tags: set[DimenensionTag]
    _canonical_unit: "SimpleUnit|None" = None

    @classmethod
    def create(cls, dimension_exponents: Tuple[float, float, float, float, float, float, float], pseudo_dimension_exponents: Tuple[int, int], has_canonical_unit: bool, unit_string: str, tags: set[DimenensionTag] = set()) -> "NamedSimpleDimensionInformation":
        dimension: SimpleDimension = SimpleDimension.create(dimension_exponents, pseudo_dimension_exponents)
        return cls(dimension, has_canonical_unit, unit_string, tags)

    @property
    def canonical_unit(self) -> "SimpleUnit":
        if self._canonical_unit is None:
            from .simple.simple_unit import SimpleUnit
            object.__setattr__(self, '_canonical_unit', SimpleUnit.parse_string(self.canonical_unit_string))
        return self._canonical_unit

#Reminder:
#MASS_INDEX: int = 0
#TIME_INDEX: int = 1
#LENGTH_INDEX: int = 2
#CURRENT_INDEX: int = 3
#TEMPERATURE_INDEX: int = 4
#AMOUNT_OF_SUBSTANCE_INDEX: int = 5
#LUMINOUS_INTENSITY_INDEX: int = 6

class NamedSimpleDimension(Enum):
    value: NamedSimpleDimensionInformation

    # -------------------- Dimensions with Canonical SI Base Units --------------------

    MASS = NamedSimpleDimensionInformation.create(                    (1, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "kg", {DimenensionTag.BASE_DIMENSION})
    TIME = NamedSimpleDimensionInformation.create(                    (0, 1, 0, 0, 0, 0, 0), (0, 0),      True,    "s", {DimenensionTag.BASE_DIMENSION})
    LENGTH = NamedSimpleDimensionInformation.create(                  (0, 0, 1, 0, 0, 0, 0), (0, 0),      True,   "m", {DimenensionTag.BASE_DIMENSION})
    CURRENT = NamedSimpleDimensionInformation.create(                 (0, 0, 0, 1, 0, 0, 0), (0, 0),      True,   "A", {DimenensionTag.BASE_DIMENSION})
    TEMPERATURE = NamedSimpleDimensionInformation.create(             (0, 0, 0, 0, 1, 0, 0), (0, 0),      True,   "K", {DimenensionTag.BASE_DIMENSION})
    AMOUNT_OF_SUBSTANCE = NamedSimpleDimensionInformation.create(     (0, 0, 0, 0, 0, 1, 0), (0, 0),      True,   "mol", {DimenensionTag.BASE_DIMENSION})
    LUMINOUS_INTENSITY = NamedSimpleDimensionInformation.create(      (0, 0, 0, 0, 0, 0, 1), (0, 0),      True,   "cd", {DimenensionTag.BASE_DIMENSION})
    
    # -------------------- Dimensions with Canonical SI Derived Units --------------------

    FORCE = NamedSimpleDimensionInformation.create(                   (1, -2, 1, 0, 0, 0, 0), (0, 0),     True,   "N")
    ENERGY = NamedSimpleDimensionInformation.create(                  (1, -2, 2, 0, 0, 0, 0), (0, 0),     True,   "J")
    POWER = NamedSimpleDimensionInformation.create(                   (1, -3, 2, 0, 0, 0, 0), (0, 0),     True,   "W")
    PRESSURE = NamedSimpleDimensionInformation.create(                (1, -2, -1, 0, 0, 0, 0), (0, 0),    True,   "Pa")
    VOLTAGE = NamedSimpleDimensionInformation.create(                 (1, -3, 2, -1, 0, 0, 0), (0, 0),     True,   "V")
    RESISTANCE = NamedSimpleDimensionInformation.create(              (1, -3, 2, -2, 0, 0, 0), (0, 0),    True,   "Ω")
    CONDUCTANCE = NamedSimpleDimensionInformation.create(             (-1, 3, -2, 2, 0, 0, 0), (0, 0),    True,   "S")
    CAPACITANCE = NamedSimpleDimensionInformation.create(             (-1, 4, -2, 2, 0, 0, 0), (0, 0),    True,   "F")
    CHARGE = NamedSimpleDimensionInformation.create(                  (0, 1, 0, 1, 0, 0, 0), (0, 0),     True,   "C")
    INDUCTANCE = NamedSimpleDimensionInformation.create(              (1, -2, 2, -2, 0, 0, 0), (0, 0),    True,   "H")
    MAGNETIC_FLUX = NamedSimpleDimensionInformation.create(           (1, -2, 2, -1, 0, 0, 0), (0, 0),     True,   "Wb")
    MAGNETIC_FLUX_DENSITY = NamedSimpleDimensionInformation.create(   (1, -2, 0, -1, 0, 0, 0), (0, 0),     True,   "T")
    FREQUENCY = NamedSimpleDimensionInformation.create(               (0, -1, 0, 0, 0, 0, 0), (0, 0),     True,   "Hz")

    # -------------------- Dimensions with Composed SI Units --------------------

    VOLTAGE_SCAN_RATE = NamedSimpleDimensionInformation.create(       (1, -4, 2, -1, 0, 0, 0), (0, 0),    True,   "V/s")
    CURRENT_DENSITY = NamedSimpleDimensionInformation.create(         (0, 0, -2, 1, 0, 0, 0), (0, 0),     True,   "A/m^2")
    CHARGE_DENSITY = NamedSimpleDimensionInformation.create(          (0, 1, -2, 1, 0, 0, 0), (0, 0),     True,   "C/m^2")
    CAPACITANCE_DENSITY = NamedSimpleDimensionInformation.create(     (-1, 4, -4, 2, 0, 0, 0), (0, 0),    True,   "F/m^2")
    
    # -------------------- Dimensions with Non-SI Canonical Units --------------------

    ANGLE = NamedSimpleDimensionInformation.create(                   (0, 0, 0, 0, 0, 0, 0), (0, 1),      True,   "rad", {DimenensionTag.BASE_DIMENSION})
    ACTIVITY = NamedSimpleDimensionInformation.create(                (0, -1, 0, 0, 0, 0, 0), (0, 0),     True,   "Bq")
    DOSE_EQUIVALENT = NamedSimpleDimensionInformation.create(         (0, -2, 2, 0, 0, 0, 0), (0, 0),     True,   "Sv")
    DOSE_ABSORTED = NamedSimpleDimensionInformation.create(           (0, -2, 2, 0, 0, 0, 0), (0, 0),      True,   "Gy")
    LOGLEVEL = NamedSimpleDimensionInformation.create(                (0, 0, 0, 0, 0, 0, 0), (1, 0),      True,   "np", {DimenensionTag.BASE_DIMENSION})
    FRACTION = NamedSimpleDimensionInformation.create(                (0, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "")
    NUMBER = NamedSimpleDimensionInformation.create(                  (0, 0, 0, 0, 0, 0, 0), (0, 0),      True,   "")
    
    # -------------------- Dimensions with Non-SI Non-canonical Units --------------------
    
    AREA = NamedSimpleDimensionInformation.create(                    (0, 0, 2, 0, 0, 0, 0), (0, 0),      True,   "m^2")
    VOLUME = NamedSimpleDimensionInformation.create(                  (0, 0, 3, 0, 0, 0, 0), (0, 0),      True,   "m^3")
    DENSITY = NamedSimpleDimensionInformation.create(                 (1, 0, -3, 0, 0, 0, 0), (0, 0),     True,   "kg/m^3")
    VOLUMETRIC_FLOW_RATE = NamedSimpleDimensionInformation.create(    (0, -1, 3, 0, 0, 0, 0), (0, 0),     True,   "m^3/s")
    AREAL_FLOW_RATE = NamedSimpleDimensionInformation.create(         (0, -1, 2, 0, 0, 0, 0), (0, 0),     False,  "m^2/s")
    VOLUMETRIC_DENSITY = NamedSimpleDimensionInformation.create(      (0, 0, -3, 0, 0, 0, 0), (0, 0),     False,  "1/m^3")
    AREAL_DENSITY = NamedSimpleDimensionInformation.create(           (1, 0, -2, 0, 0, 0, 0), (0, 0),     False,  "kg/m^2")
    KINEMATIC_VISCOSITY = NamedSimpleDimensionInformation.create(     (0, -1, 2, 0, 0, 0, 0), (0, 0),     True,   "m^2/s")
    DYNAMIC_VISCOSITY = NamedSimpleDimensionInformation.create(       (1, -1, -1, 0, 0, 0, 0), (0, 0),    True,   "Pa*s")

    @property
    def simple_dimension(self) -> SimpleDimension:
        return self.value.unit_dimension
    
    @property
    def canonical_unit(self) -> "SimpleUnit":
        return self.value.canonical_unit
    
    def __getitem__(self, canonical_dimension: BaseDimension) -> "NamedSimpleDimension":
        named_canonical_dimension: NamedSimpleDimension|None = self.find_named_unit_dimension(canonical_dimension)
        if named_canonical_dimension is None:
            raise ValueError(f"No named canonical dimension found for canonical dimension {canonical_dimension}")
        return named_canonical_dimension

    @classmethod
    def has_named_canonical_dimension(cls, canonical_dimension: SimpleDimension) -> bool:
        named_canonical_dimension: NamedSimpleDimension|None = cls.find_named_unit_dimension(canonical_dimension)
        return named_canonical_dimension is not None

    @classmethod
    def find_named_unit_dimension(cls, canonical_dimension: BaseDimension) -> "NamedSimpleDimension|None":
        for named_canonical_dimension in cls:
            if named_canonical_dimension.simple_dimension == canonical_dimension:
                return named_canonical_dimension
        return None
    
    @classmethod
    def unit_dimensions_by_tags(cls, tags: set[DimenensionTag]) -> list["NamedSimpleDimension"]:
        return [named_canonical_dimension for named_canonical_dimension in cls if tags.issubset(named_canonical_dimension.value.tags)]