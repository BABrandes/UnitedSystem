from typing import Optional, TYPE_CHECKING
from enum import Enum
from .proper_exponents import ProperExponents   

if TYPE_CHECKING:
    from .unit_element import UnitElement
    from .dimension import Dimension
    from .unit import Unit

class QuantityTag(Enum):
    BASE_QUANTITY = "base_quantity"
    DERIVED_QUANTITY = "derived_quantity"
    COMPOSED_QUANTITY = "composed_quantity"
    SI_QUANTITY = "si_quantity"
    NON_SI_QUANTITY = "non_si_quantity"
    SINGLE_ELEMENT_QUANTITY = "single_element_quantity"
    CONSIDER_FOR_REDUCTION = "consider_for_reduction"

class NamedQuantity(Enum):
    value: tuple[str, "ProperExponents", set[QuantityTag]] # type: ignore

    # -------------------- Dimensions with Canonical SI Base Units --------------------

    MASS =                      ("kg",      ProperExponents(mass=1),                                {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Mass (kg = kilogram)
    """Dimension: M; Canonical unit: kg = kilogram"""
    TIME =                      ("s",       ProperExponents(time=1),                                {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Time (s = second)
    """Dimension: T; Canonical unit: s = second"""
    LENGTH =                    ("m",       ProperExponents(length=1),                              {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Length (m = meter)
    """Dimension: L; Canonical unit: m = meter"""
    CURRENT =                   ("A",       ProperExponents(current=1),                             {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Current (A = ampere)
    """Dimension: I; Canonical unit: A = ampere"""
    TEMPERATURE =               ("K",       ProperExponents(temperature=1),                         {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Temperature (K = Kelvin)
    """Dimension: Θ; Canonical unit: K = Kelvin"""
    AMOUNT_OF_SUBSTANCE =       ("mol",     ProperExponents(amount=1),                              {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Amount of substance (mol = mole)
    """Dimension: N; Canonical unit: mol = mole"""
    LUMINOUS_INTENSITY =        ("cd",      ProperExponents(luminous_intensity=1),                  {QuantityTag.BASE_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Luminous intensity  (cd = candela)
    """Dimension: J; Canonical unit: cd = candela"""
    
    # -------------------- Dimensions with Canonical SI Derived Units --------------------

    FORCE =                     ("N",       ProperExponents(mass=1, length=1, time=-2),             {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Mass per length per time squared
    """Dimension: M*L/T^2; Canonical unit: N = newton"""
    ENERGY =                    ("J",       ProperExponents(mass=1, length=2, time=-2),             {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Work    
    """Dimension: E; Canonical unit: J = joule"""
    POWER =                     ("W",       ProperExponents(mass=1, length=2, time=-3),             {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Energy per time
    """Dimension: M*L^2/T^3; Canonical unit: W = watt"""
    PRESSURE =                  ("Pa",      ProperExponents(mass=1, length=-1, time=-2),            {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Force per area
    """Dimension: M/L*T^2; Canonical unit: Pa = pascal"""
    VOLTAGE =                   ("V",       ProperExponents(mass=1, length=2, time=-3, current=-1), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Energy per charge
    """Dimension: M*L^2/T^3*I^-1; Canonical unit: V = volt"""
    ELECTRIC_POTENTIAL =        ("V",       ProperExponents(mass=1, length=2, time=-3, current=-1), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # Energy per charge (same as VOLTAGE)
    """Dimension: M*L^2/T^3*I^-1; Canonical unit: V = volt"""
    RESISTANCE =                ("Ω",       ProperExponents(mass=1, length=2, time=-3, current=-2), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Voltage per current
    """Dimension: M*L^2/T^3*I^-2; Canonical unit: Ω = ohm"""
    CONDUCTANCE =               ("S",       ProperExponents(mass=1, length=-2, time=3, current=2),  {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Current per voltage
    """Dimension: M^-1*L^-2*T^3*I^2; Canonical unit: S = siemens"""
    CAPACITANCE =               ("F",       ProperExponents(mass=-1, length=-2, time=4, current=2), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Charge per voltage
    """Dimension: M^-1*L^-2*T^4*I^2; Canonical unit: F = farad"""
    CHARGE =                    ("C",       ProperExponents(time=1, current=1),                     {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Current per time
    """Dimension: T*I; Canonical unit: C = coulomb"""
    INDUCTANCE =                ("H",       ProperExponents(mass=1, length=2, time=-2, current=-2), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Voltage per current per time
    """Dimension: M*L^2/T^2*I^-2; Canonical unit: H = henry"""
    MAGNETIC_FLUX =             ("Wb",      ProperExponents(mass=1, length=2, time=-2, current=-1), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Voltage per time
    """Dimension: M*L^2/T^2*I^-1; Canonical unit: Wb = weber"""
    MAGNETIC_FLUX_DENSITY =     ("T",       ProperExponents(mass=1, time=-2, current=-1),           {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY, QuantityTag.CONSIDER_FOR_REDUCTION}) # Force per current per length
    """Dimension: M/T^2*I^-1; Canonical unit: T = tesla"""
    FREQUENCY =                 ("Hz",      ProperExponents(time=-1),                               {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # 1 per time
    """Dimension: T^-1; Canonical unit: Hz = hertz"""
    
    # Additional SI derived units
    ELECTRIC_FIELD =            ("V/m",     ProperExponents(mass=1, length=1, time=-3, current=-1), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Voltage per length
    """Dimension: M*L/T^3*I^-1; Canonical unit: V/m = volt per meter"""
    ELECTRIC_DISPLACEMENT =     ("C/m^2",   ProperExponents(current=1, time=1, length=-2),          {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Charge per area
    """Dimension: T*I/L^2; Canonical unit: C/m^2 = coulomb per square meter"""
    MAGNETIC_FIELD =            ("A/m",     ProperExponents(current=1, length=-1),                  {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Current per length
    """Dimension: T*I/L; Canonical unit: A/m = ampere per meter"""
    MAGNETIZATION =             ("A/m",     ProperExponents(current=1, length=-1),                  {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Magnetic moment per volume
    """Dimension: T*I/L; Canonical unit: A/m = ampere per meter"""
    ELECTRIC_CURRENT_DENSITY =  ("A/m^2",   ProperExponents(current=1, length=-2),                  {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Current per area
    """Dimension: T*I/L^2; Canonical unit: A/m^2 = ampere per square meter"""
    ELECTRIC_CHARGE_DENSITY =   ("C/m^3",   ProperExponents(current=1, time=1, length=-3),          {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Charge per volume
    """Dimension: T*I/L^3; Canonical unit: C/m^3 = coulomb per cubic meter"""
    ELECTRIC_FLUX =             ("V*m",     ProperExponents(mass=1, length=3, time=-3, current=-1), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Electric field times area
    """Dimension: M*L^3/T^3*I^-1; Canonical unit: V*m = volt meter"""
    MAGNETIC_FLUX_DENSITY_ALT = ("Wb/m^2",  ProperExponents(mass=1, time=-2, current=-1),           {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Magnetic flux per area (same as TESLA)
    """Dimension: M/T^2*I^-1; Canonical unit: Wb/m^2 = weber per square meter"""
    ELECTRIC_DIPOLE_MOMENT =    ("C*m",     ProperExponents(current=1, time=1, length=1),           {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Charge times distance
    """Dimension: T*I*L; Canonical unit: C*m = coulomb meter"""
    MAGNETIC_DIPOLE_MOMENT =    ("A*m^2",   ProperExponents(current=1, length=2),                   {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Current times area
    """Dimension: T*I*L^2; Canonical unit: A*m^2 = ampere square meter"""
    ELECTRIC_POLARIZATION =     ("C/m^2",   ProperExponents(current=1, time=1, length=-2),          {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Electric dipole moment per volume
    """Dimension: T*I/L^2; Canonical unit: C/m^2 = coulomb per square meter"""
    MAGNETIC_POLARIZATION =     ("T",       ProperExponents(mass=1, time=-2, current=-1),           {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Magnetic field (same as TESLA)
    """Dimension: M/T^2*I^-1; Canonical unit: T = tesla"""
    ELECTRIC_SUSCEPTIBILITY =   ("",        ProperExponents(),                                      {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Dimensionless
    """Dimension: 1; Canonical unit: 1 = 1"""
    MAGNETIC_SUSCEPTIBILITY =   ("",        ProperExponents(),                                      {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Dimensionless
    """Dimension: 1; Canonical unit: 1 = 1"""
    ELECTRIC_PERMITTIVITY =     ("F/m",     ProperExponents(mass=-1, length=-3, time=4, current=2), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Capacitance per length
    """Dimension: M^-1*L^-3*T^4*I^2; Canonical unit: F/m = farad per meter"""
    MAGNETIC_PERMEABILITY =     ("H/m",     ProperExponents(mass=1, length=1, time=-2, current=-2), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Inductance per length
    """Dimension: M*L/T^2*I^-2; Canonical unit: H/m = henry per meter"""
    CONDUCTIVITY =              ("S/m",     ProperExponents(mass=-1, length=-3, time=3, current=2),  {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY}) # Conductance per length
    """Dimension: M^-1*L^-3*T^3*I^2; Canonical unit: S/m = siemens per meter"""
    RESISTIVITY =                ("Ω*m",     ProperExponents(mass=1, length=3, time=-3, current=-2), {QuantityTag.DERIVED_QUANTITY, QuantityTag.SI_QUANTITY,}) # Resistance per length
    """Dimension: M*L^3/T^3*I^-2; Canonical unit: Ω*m = ohm meter"""

    # -------------------- Dimensions with Composed SI Units --------------------

    VOLTAGE_SCAN_RATE =         ("V/s",     ProperExponents(length=1, time=-1),                     {QuantityTag.COMPOSED_QUANTITY, QuantityTag.SI_QUANTITY}) # Voltage per time
    """Dimension: M*L/T^3*I^-1; Canonical unit: V/s = volt per second"""
    CURRENT_DENSITY =           ("A/m^2",   ProperExponents(current=1, length=-2),                  {QuantityTag.COMPOSED_QUANTITY, QuantityTag.SI_QUANTITY}) # Current per area
    """Dimension: T*I/L^2; Canonical unit: A/m^2 = ampere per square meter"""
    CHARGE_DENSITY =            ("C/m^2",   ProperExponents(current=1, length=-2),                  {QuantityTag.COMPOSED_QUANTITY, QuantityTag.SI_QUANTITY}) # Charge per area
    """Dimension: T*I/L^2; Canonical unit: C/m^2 = coulomb per square meter"""
    CAPACITANCE_DENSITY =       ("F/m^2",   ProperExponents(mass=-1, length=-2, time=4, current=2), {QuantityTag.COMPOSED_QUANTITY, QuantityTag.SI_QUANTITY}) # Charge per voltage per area
    """Dimension: M^-1*L^-2*T^4*I^2; Canonical unit: F/m^2 = farad per square meter"""
    
    # -------------------- Dimensions with Non-SI Canonical Units --------------------

    ANGLE =                     ("rad",     ProperExponents(angle=1),                               {QuantityTag.BASE_QUANTITY, QuantityTag.NON_SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # Angle
    """Dimension: 1; Canonical unit: rad = radian"""
    ACTIVITY =                  ("Bq",      ProperExponents(time=-1),                               {QuantityTag.NON_SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # 1 per time
    """Dimension: T^-1; Canonical unit: Bq = becquerel"""
    DOSE_EQUIVALENT =           ("Sv",      ProperExponents(mass=1, length=2, time=-2),             {QuantityTag.NON_SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # Energy per mass per area per time squared
    """Dimension: M*L^2/T^-2; Canonical unit: Sv = sievert"""
    DOSE_ABSORTED =             ("Gy",      ProperExponents(mass=1, length=2, time=-2),             {QuantityTag.NON_SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # Energy per mass per area
    """Dimension: M*L^2/T^-2; Canonical unit: Gy = gray"""
    LOGLEVEL =                  ("Np",      ProperExponents(),                                      {QuantityTag.NON_SI_QUANTITY, QuantityTag.SINGLE_ELEMENT_QUANTITY}) # Log level
    """Dimension: 1; Canonical unit: Np = neper"""
    FRACTION =                  ("",        ProperExponents(),                                      {QuantityTag.NON_SI_QUANTITY}) # Fraction
    """Dimension: 1; Canonical unit: 1 = 1"""
    
    # -------------------- Dimensions with Non-SI Non-canonical Units --------------------
    
    AREA = ("m^2", ProperExponents(length=2)) # Length squared
    """Dimension: L^2; Canonical unit: m^2 = square meter"""
    VOLUME = ("m^3", ProperExponents(length=3)) # Length cubed
    """Dimension: L^3; Canonical unit: m^3 = cubic meter"""
    DENSITY = ("kg/m^3", ProperExponents(mass=1, length=-3)) # Mass per length cubed
    """Dimension: M/L^3; Canonical unit: kg/m^3 = kilogram per cubic meter"""
    AREAL_FLOW_RATE = ("m^2/s", ProperExponents(length=2, time=-1)) # Area per time
    """Dimension: L^2/T; Canonical unit: m^2/s = square meter per second"""
    VOLUMETRIC_DENSITY = ("1/m^3", ProperExponents(length=-3)) # 1 per length cubed
    """Dimension: L^-3; Canonical unit: 1/m^3 = 1 per cubic meter"""
    AREAL_DENSITY = ("kg/m^2", ProperExponents(mass=1, length=-2)) # Mass per area
    """Dimension: M/L^2; Canonical unit: kg/m^2 = kilogram per square meter"""
    KINEMATIC_VISCOSITY = ("m^2/s", ProperExponents(length=2, time=-1)) # Area per time
    """Dimension: L^2/T; Canonical unit: m^2/s = square meter per second"""
    DYNAMIC_VISCOSITY = ("Pa*s", ProperExponents(mass=1, length=-1, time=-1)) # Mass per length per time
    """Dimension: M/L*T; Canonical unit: Pa*s = pascal second"""

    @property
    def dimension(self) -> "Dimension":
        if not hasattr(self, '_dimension') or self._dimension is None:
            from .dimension import Dimension
            self._dimension: Optional["Dimension"] = Dimension(self)
        return self._dimension
    
    @property
    def unit_element(self) -> Optional["UnitElement"]:
        if not hasattr(self, '_unit_element'):
            from .unit_element import UnitElement
            unit_element_str = self.value[0]  # First element is the unit symbol string
            # Check if this quantity has tags and if it's marked as single element
            has_single_element_tag = (len(self.value) >= 3 and 
                                    QuantityTag.SINGLE_ELEMENT_QUANTITY in self.value[2])
            if has_single_element_tag:
                self._unit_element: Optional["UnitElement"] = UnitElement.parse_string(unit_element_str, "nominator")
            else:
                self._unit_element: Optional["UnitElement"] = None
        return self._unit_element

    @property
    def unit(self) -> "Unit":
        if not hasattr(self, '_unit') or self._unit is None:
            from .unit import Unit
            self._unit: Optional[Unit] = Unit(self.value[0])
        return self._unit
    
    @property
    def proper_exponents(self) -> ProperExponents:
        if not hasattr(self, '_proper_exponents'):
            self._proper_exponents: ProperExponents = self.value[1]
        return self._proper_exponents

    @property
    def proper_exponents_as_tuple(self) -> tuple[float, float, float, float, float, float, float, float]:
        if not hasattr(self, '_proper_exponents_as_tuple'):
            self._proper_exponents_as_tuple: tuple[float, float, float, float, float, float, float, float] = self.value[1].proper_exponents
        return self._proper_exponents_as_tuple
    
    @property
    def tags(self) -> set[QuantityTag]:
        return self.value[2]
    
    @classmethod
    def find_named_quantity(cls, dimension: "Dimension") -> Optional["NamedQuantity"]:
        for named_quantity in cls:
            if named_quantity.dimension == dimension:
                return named_quantity
        return None

SI_BASE_QUANTITIES: set[NamedQuantity] = set()
SI_DERIVED_QUANTITIES: set[NamedQuantity] = set()
SINGLE_ELEMENT_QUANTITIES: set[NamedQuantity] = set()
SINGLE_ELEMENT_SI_QUANTITIES: set[NamedQuantity] = set()
CONSIDER_FOR_REDUCTION_QUANTITIES: set[NamedQuantity] = set()
for quantity in NamedQuantity:
    value_tuple = quantity.value
    if len(value_tuple) >= 3 and QuantityTag.BASE_QUANTITY in value_tuple[2]:
        SI_BASE_QUANTITIES.add(quantity)
    if len(value_tuple) >= 3 and QuantityTag.DERIVED_QUANTITY in value_tuple[2]:
        SI_DERIVED_QUANTITIES.add(quantity)
    if len(value_tuple) >= 3 and QuantityTag.SINGLE_ELEMENT_QUANTITY in value_tuple[2]:
        SINGLE_ELEMENT_QUANTITIES.add(quantity)
    if len(value_tuple) >= 3 and QuantityTag.SI_QUANTITY in value_tuple[2] and QuantityTag.SINGLE_ELEMENT_QUANTITY in value_tuple[2]:
        SINGLE_ELEMENT_SI_QUANTITIES.add(quantity)
    if len(value_tuple) >= 3 and QuantityTag.CONSIDER_FOR_REDUCTION in value_tuple[2]:
        CONSIDER_FOR_REDUCTION_QUANTITIES.add(quantity)