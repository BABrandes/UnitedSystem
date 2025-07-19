from typing import Optional, TYPE_CHECKING
from enum import Enum

from .utils.units.dimension_group import ProperExponents, DimensionGroup

if TYPE_CHECKING:
    from .utils.units.unit_group import UnitGroup

class NamedQuantity(Enum):
    value: tuple[str, ProperExponents] # type: ignore

    # -------------------- Dimensions with Canonical SI Base Units --------------------

    MASS =                      ("kg",      ProperExponents(mass=1)) # Mass (kg = kilogram)
    TIME =                      ("s",       ProperExponents(time=1)) # Time (s = second)
    LENGTH =                    ("m",       ProperExponents(length=1)) # Length (m = meter)
    CURRENT =                   ("A",       ProperExponents(current=1)) # Current (A = ampere)
    TEMPERATURE =               ("K",       ProperExponents(temperature=1)) # Temperature (K = Kelvin)
    AMOUNT_OF_SUBSTANCE =       ("mol",     ProperExponents(amount=1)) # Amount of substance (mol = mole)
    LUMINOUS_INTENSITY =        ("cd",      ProperExponents(luminous_intensity=1)) # Luminous intensity  (cd = candela)
    
    # -------------------- Dimensions with Canonical SI Derived Units --------------------

    FORCE =                     ("N",       ProperExponents(mass=1, length=1, time=-2)) # Mass per length per time squared
    ENERGY =                    ("J",       ProperExponents(mass=1, length=2, time=-2)) # Work    
    POWER =                     ("W",       ProperExponents(mass=1, length=2, time=-3)) # Energy per time
    PRESSURE =                  ("Pa",      ProperExponents(mass=1, length=-1, time=-2)) # Force per area
    VOLTAGE =                   ("V",       ProperExponents(mass=1, length=2, time=-3, current=-1)) # Energy per charge
    RESISTANCE =                ("Ω",       ProperExponents(mass=1, length=2, time=-3, current=-2)) # Voltage per current
    CONDUCTANCE =               ("S",       ProperExponents(mass=1, length=-2, time=3, current=2)) # Current per voltage
    CAPACITANCE =               ("F",       ProperExponents(mass=-1, length=-2, time=4, current=2)) # Charge per voltage
    CHARGE =                    ("C",       ProperExponents(current=1)) # Current per time
    INDUCTANCE =                ("H",       ProperExponents(mass=1, length=2, time=-2, current=-2)) # Voltage per current per time
    MAGNETIC_FLUX =             ("Wb",      ProperExponents(mass=1, length=2, time=-2, current=-1)) # Voltage per time
    MAGNETIC_FLUX_DENSITY =     ("T",       ProperExponents(mass=1, time=-2, current=-1)) # Force per current per length
    FREQUENCY =                 ("Hz",      ProperExponents(time=-1)) # 1 per time

    # -------------------- Dimensions with Composed SI Units --------------------

    VOLTAGE_SCAN_RATE =         ("V/s",     ProperExponents(length=1, time=-1)) # Voltage per time
    CURRENT_DENSITY =           ("A/m^2",   ProperExponents(current=1, length=-2)) # Current per area
    CHARGE_DENSITY =            ("C/m^2",   ProperExponents(current=1, length=-2)) # Charge per area
    CAPACITANCE_DENSITY =       ("F/m^2",   ProperExponents(mass=-1, length=-2, time=4, current=2)) # Charge per voltage per area
    
    # -------------------- Dimensions with Non-SI Canonical Units --------------------

    ANGLE =                     ("rad",     ProperExponents(angle=1)) # Angle
    ACTIVITY =                  ("Bq",      ProperExponents(time=-1)) # 1 per time
    DOSE_EQUIVALENT =           ("Sv",      ProperExponents(mass=1, length=2, time=-2)) # Energy per mass per area per time squared
    DOSE_ABSORTED =             ("Gy",      ProperExponents(mass=1, length=2, time=-2)) # Energy per mass per area
    LOGLEVEL = ("Np", ProperExponents(log_level=1)) # Log level
    FRACTION = ("", ProperExponents()) # Fraction
    NUMBER = ("", ProperExponents()) # Number
    
    # -------------------- Dimensions with Non-SI Non-canonical Units --------------------
    
    AREA = ("m^2", ProperExponents(length=2)) # Length squared
    VOLUME = ("m^3", ProperExponents(length=3)) # Length cubed
    DENSITY = ("kg/m^3", ProperExponents(mass=1, length=-3)) # Mass per length cubed
    AREAL_FLOW_RATE = ("m^2/s", ProperExponents(length=2, time=-1)) # Area per time
    VOLUMETRIC_DENSITY = ("1/m^3", ProperExponents(length=-3)) # 1 per length cubed
    AREAL_DENSITY = ("kg/m^2", ProperExponents(mass=1, length=-2)) # Mass per area
    KINEMATIC_VISCOSITY = ("m^2/s", ProperExponents(length=2, time=-1)) # Area per time
    DYNAMIC_VISCOSITY = ("Pa*s", ProperExponents(mass=1, length=-1, time=-1)) # Mass per length per time

    @property
    def dimension_group(self) -> DimensionGroup:

        if self._dimension_group is not None:
            exponents: tuple[float, float, float, float, float, float, float] = (
                self.value[1].mass,
                self.value[1].time,
                self.value[1].length,
                self.value[1].current,
                self.value[1].temperature,
                self.value[1].amount,
                self.value[1].luminous_intensity,
            )
            log_dimension: Optional[DimensionGroup] = None
            angle_dimension: Optional[DimensionGroup] = None
            log_dimension_exponent: float = 0.0
            angle_dimension_exponent: float = 0.0   
            if self.value[1].log_level != 0:
                log_dimension = DimensionGroup((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                log_dimension_exponent = self.value[1].log_level
            if self.value[1].angle != 0:
                angle_dimension = DimensionGroup((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
                angle_dimension_exponent = self.value[1].angle

            self._dimension_group: Optional[DimensionGroup] = DimensionGroup(
                exponents,
                log_dimension,
                angle_dimension,
                log_dimension_exponent,
                angle_dimension_exponent,
                )
        if self._dimension_group is None:
            raise AssertionError(f"Dimension group for {self.name} is not set")
        
        return self._dimension_group
    
    @property
    def unit_group(self) -> UnitGroup:
        from .utils.units.unit_group import UnitGroup

        if self._unit_group is not None:
            self._unit_group: Optional[UnitGroup] = UnitGroup(self.value[0])
        if self._unit_group is None:
            raise AssertionError(f"Unit group for {self.name} is not set")
        
        return self._unit_group