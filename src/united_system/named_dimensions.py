from .units.named_simple_dimensions import NamedSimpleDimension
from enum import Enum
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class DimensionExponents():
    mass: float = field(default=0.0)
    time: float = field(default=0.0)
    length: float = field(default=0.0)
    current: float = field(default=0.0)
    temperature: float = field(default=0.0)
    amount: float = field(default=0.0)
    luminous_intensity: float = field(default=0.0)
    angle: int = field(default=0)
    log_level: int = field(default=0)

class NamedDimension(Enum):
    value: NamedSimpleDimension|DimensionExponents
    """User-friendly named dimensions wrapping all NamedSimpleUnitDimension values."""
    
    # Base dimensions
    MASS = NamedSimpleDimension.MASS
    TIME = NamedSimpleDimension.TIME
    LENGTH = NamedSimpleDimension.LENGTH
    CURRENT = NamedSimpleDimension.CURRENT
    TEMPERATURE = NamedSimpleDimension.TEMPERATURE
    AMOUNT = NamedSimpleDimension.AMOUNT_OF_SUBSTANCE
    LUMINOUS_INTENSITY = NamedSimpleDimension.LUMINOUS_INTENSITY
    ANGLE = NamedSimpleDimension.ANGLE
    LOG_LEVEL = NamedSimpleDimension.LOGLEVEL
    
    # Derived mechanical dimensions
    FORCE = NamedSimpleDimension.FORCE
    ENERGY = NamedSimpleDimension.ENERGY
    POWER = NamedSimpleDimension.POWER
    PRESSURE = NamedSimpleDimension.PRESSURE
    
    # Derived electrical dimensions
    VOLTAGE = NamedSimpleDimension.VOLTAGE
    RESISTANCE = NamedSimpleDimension.RESISTANCE
    CONDUCTANCE = NamedSimpleDimension.CONDUCTANCE
    CAPACITANCE = NamedSimpleDimension.CAPACITANCE
    CHARGE = NamedSimpleDimension.CHARGE
    INDUCTANCE = NamedSimpleDimension.INDUCTANCE
    MAGNETIC_FLUX = NamedSimpleDimension.MAGNETIC_FLUX
    MAGNETIC_FLUX_DENSITY = NamedSimpleDimension.MAGNETIC_FLUX_DENSITY
    FREQUENCY = NamedSimpleDimension.FREQUENCY
    
    # Specialized electrical dimensions
    VOLTAGE_SCAN_RATE = NamedSimpleDimension.VOLTAGE_SCAN_RATE
    CURRENT_DENSITY = NamedSimpleDimension.CURRENT_DENSITY
    CHARGE_DENSITY = NamedSimpleDimension.CHARGE_DENSITY
    CAPACITANCE_DENSITY = NamedSimpleDimension.CAPACITANCE_DENSITY
    
    # Radiation dimensions
    ACTIVITY = NamedSimpleDimension.ACTIVITY
    DOSE_EQUIVALENT = NamedSimpleDimension.DOSE_EQUIVALENT
    DOSE_ABSORBED = NamedSimpleDimension.DOSE_ABSORTED
    
    # Geometric dimensions
    AREA = NamedSimpleDimension.AREA
    VOLUME = NamedSimpleDimension.VOLUME
    
    # Density and flow dimensions
    DENSITY = NamedSimpleDimension.DENSITY
    VOLUMETRIC_FLOW_RATE = NamedSimpleDimension.VOLUMETRIC_FLOW_RATE
    AREAL_FLOW_RATE = NamedSimpleDimension.AREAL_FLOW_RATE
    VOLUMETRIC_DENSITY = NamedSimpleDimension.VOLUMETRIC_DENSITY
    AREAL_DENSITY = NamedSimpleDimension.AREAL_DENSITY
    
    # Viscosity dimensions
    KINEMATIC_VISCOSITY = NamedSimpleDimension.KINEMATIC_VISCOSITY
    DYNAMIC_VISCOSITY = NamedSimpleDimension.DYNAMIC_VISCOSITY
    
    # Dimensionless quantities
    FRACTION = NamedSimpleDimension.FRACTION
    NUMBER = NamedSimpleDimension.NUMBER

    # Additional common dimensions
    VELOCITY = DimensionExponents(length=1, time=-1)             # length/time
    ACCELERATION = DimensionExponents(length=1, time=-2)         # length/time²
    MOMENTUM = DimensionExponents(mass=1, length=1, time=-1)     # mass×length/time
    ANGULAR_VELOCITY = DimensionExponents(angle=1, time=-1)      # angle/time
    ANGULAR_ACCELERATION = DimensionExponents(angle=1, time=-2)  # angle/time²