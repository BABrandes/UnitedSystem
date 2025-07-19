from dataclasses import dataclass, field
from typing import Final, Tuple, Optional, TYPE_CHECKING
import h5py

from .dimension_symbol import DimensionSymbol

if TYPE_CHECKING:
    from .unit_group import UnitGroup

EPSILON: Final[float] = 1e-10

# Cache for canonical units to avoid repeated expensive calculations
_CANONICAL_UNIT_CACHE: dict["DimensionGroup", "UnitGroup"] = {}

@dataclass(frozen=True, slots=True)
class ProperExponents:
    mass: float = field(default=0.0)
    time: float = field(default=0.0)
    length: float = field(default=0.0)
    current: float = field(default=0.0)
    temperature: float = field(default=0.0)
    amount: float = field(default=0.0)
    luminous_intensity: float = field(default=0.0)
    angle: float = field(default=0.0)
    log_level: float = field(default=0.0)

@dataclass(frozen=True, slots=True, init=False)
class DimensionGroup:
    proper_exponents: Final[Tuple[float, float, float, float, float, float, float]]
    log_dimension: Final[Optional["DimensionGroup"]]
    log_exponent: Final[float]
    angle_dimension: Final[Optional["DimensionGroup"]]
    angle_exponent: Final[float]

    def __init__(
            self,
            proper_exponents: Tuple[float, float, float, float, float, float, float]|list[float],
            log_dimension: Optional["DimensionGroup"] = None,
            angle_dimension: Optional["DimensionGroup"] = None,
            log_exponent: float = 0.0,
            angle_exponent: float = 0.0,
            ):
        if isinstance(proper_exponents, list):
            pe = tuple(proper_exponents)
        else:
            pe = proper_exponents
        self.__setattr__("proper_exponents", pe)
        self.__setattr__("log_dimension", log_dimension)
        self.__setattr__("log_exponent", log_exponent)
        self.__setattr__("angle_dimension", angle_dimension)
        self.__setattr__("angle_exponent", angle_exponent)

################################################################################
# Arithmetic operations
################################################################################

    def __mul__(self, other: "DimensionGroup") -> "DimensionGroup":
        dimension_exponents: Tuple[float, float, float, float, float, float, float] = (self.proper_exponents[0] + other.proper_exponents[0], self.proper_exponents[1] + other.proper_exponents[1], self.proper_exponents[2] + other.proper_exponents[2], self.proper_exponents[3] + other.proper_exponents[3], self.proper_exponents[4] + other.proper_exponents[4], self.proper_exponents[5] + other.proper_exponents[5], self.proper_exponents[6] + other.proper_exponents[6])
        log_dimension: Optional["DimensionGroup"] = None
        angle_dimension: Optional["DimensionGroup"] = None
        log_exponent: float = 0.0
        angle_exponent: float = 0.0
        match self.log_dimension, other.log_dimension:
            case None, None:
                pass
            case None, _:
                log_dimension = other.log_dimension
            case _, None:
                log_dimension = self.log_dimension
            case _, _:
                if self.log_dimension == other.log_dimension:
                    log_dimension = self.log_dimension
                    log_exponent = self.log_exponent + other.log_exponent
                else:
                    raise ValueError("Cannot multiply two dimension with log dimensions")
        match self.angle_dimension, other.angle_dimension:
            case None, None:
                pass
            case None, _:
                angle_dimension = other.angle_dimension
            case _, None:
                angle_dimension = self.angle_dimension
            case _, _:
                if self.angle_dimension == other.angle_dimension:
                    angle_dimension = self.angle_dimension
                    angle_exponent = self.angle_exponent + other.angle_exponent
                else:
                    raise ValueError("Cannot multiply two dimension with angle dimensions")
        return DimensionGroup(dimension_exponents, log_dimension, angle_dimension, log_exponent, angle_exponent)

    def __truediv__(self, other: "DimensionGroup") -> "DimensionGroup":
        dimension_exponents: Tuple[float, float, float, float, float, float, float] = (self.proper_exponents[0] - other.proper_exponents[0], self.proper_exponents[1] - other.proper_exponents[1], self.proper_exponents[2] - other.proper_exponents[2], self.proper_exponents[3] - other.proper_exponents[3], self.proper_exponents[4] - other.proper_exponents[4], self.proper_exponents[5] - other.proper_exponents[5], self.proper_exponents[6] - other.proper_exponents[6])
        log_dimension: Optional["DimensionGroup"] = None
        angle_dimension: Optional["DimensionGroup"] = None
        log_exponent: float = 0.0
        angle_exponent: float = 0.0
        match self.log_dimension, other.log_dimension:
            case None, None:
                pass
            case None, _:
                log_dimension = other.log_dimension
            case _, None:
                log_dimension = self.log_dimension
            case _, _:
                if self.log_dimension == other.log_dimension:
                    log_dimension = self.log_dimension
                    log_exponent = self.log_exponent - other.log_exponent
                else:
                    raise ValueError("Cannot divide two dimension with log dimensions")
        match self.angle_dimension, other.angle_dimension:
            case None, None:
                pass
            case None, _:
                angle_dimension = other.angle_dimension
            case _, None:
                angle_dimension = self.angle_dimension
            case _, _:
                if self.angle_dimension == other.angle_dimension:
                    angle_dimension = self.angle_dimension
                    angle_exponent = self.angle_exponent - other.angle_exponent
                else:
                    raise ValueError("Cannot divide two dimension with angle dimensions")
        return DimensionGroup(dimension_exponents, log_dimension, angle_dimension, log_exponent, angle_exponent)
    
    def __pow__(self, exponent: float|int) -> "DimensionGroup":
        dimension_exponents: Tuple[float, float, float, float, float, float, float] = (self.proper_exponents[0] * exponent, self.proper_exponents[1] * exponent, self.proper_exponents[2] * exponent, self.proper_exponents[3] * exponent, self.proper_exponents[4] * exponent, self.proper_exponents[5] * exponent, self.proper_exponents[6] * exponent)
        log_dimension: Optional["DimensionGroup"] = self.log_dimension
        angle_dimension: Optional["DimensionGroup"] = self.angle_dimension
        log_exponent: float = self.log_exponent * exponent
        angle_exponent: float = self.angle_exponent * exponent
        return DimensionGroup(dimension_exponents, log_dimension, angle_dimension, log_exponent, angle_exponent)
    
    def __invert__(self) -> "DimensionGroup":
        dimension_exponents: Tuple[float, float, float, float, float, float, float] = (-self.proper_exponents[0], -self.proper_exponents[1], -self.proper_exponents[2], -self.proper_exponents[3], -self.proper_exponents[4], -self.proper_exponents[5], -self.proper_exponents[6])
        angle_dimension: Optional["DimensionGroup"] = self.angle_dimension
        log_dimension: Optional["DimensionGroup"] = self.log_dimension
        log_exponent: float = -self.log_exponent
        angle_exponent: float = -self.angle_exponent
        return DimensionGroup(dimension_exponents, log_dimension, angle_dimension, log_exponent, angle_exponent)

    def invert(self) -> "DimensionGroup":
        return ~self
    
    def exp(self) -> "DimensionGroup":
        """
        Return the log dimension of the current dimension as a new dimension group.
        """
        if self.log_dimension is None:
            if all(exp == 0 for exp in self.proper_exponents) and self.angle_dimension is None:
                return DimensionGroup(
                    proper_exponents=(0, 0, 0, 0, 0, 0, 0),
                    log_dimension=None,
                    angle_dimension=None,
                    log_exponent=0.0,
                    angle_exponent=0.0
                )
            else:
                raise ValueError("Cannot exponentiate a dimension with non-zero proper exponents or angle dimension")
        else:
            if all(exp == 0 for exp in self.proper_exponents) and self.angle_dimension is None:
                log_exponent = self.log_exponent - 1
                if abs(self.log_exponent) < EPSILON:
                    log_exponent = 0.0
                else:
                    raise ValueError("The dimension must have a log dimension of one.")
                return DimensionGroup(
                    proper_exponents=self.log_dimension.proper_exponents,
                    log_dimension=self.log_dimension.log_dimension,
                    angle_dimension=self.log_dimension.angle_dimension,
                    log_exponent=log_exponent,
                    angle_exponent=self.log_dimension.angle_exponent
                )
            else:
                raise ValueError("Cannot exponentiate a dimension with non-zero proper exponents or angle dimension")
            
    def log(self) -> "DimensionGroup":
        """
        Returns the current dimension as a log dimension of a new dimension group.
        """
        return DimensionGroup(
            proper_exponents=(0, 0, 0, 0, 0, 0, 0),
            log_dimension=self,
            angle_dimension=None,
            log_exponent=1.0,
            angle_exponent=0.0
        )
        
    def trig(self) -> "DimensionGroup":
        """
        (sin, cos, tan, cot)
        Return the angle dimension of the current dimension as a new dimension group.
        """
        if self.angle_dimension is None:
            if all(exp == 0 for exp in self.proper_exponents) and self.log_dimension is None:
                return DimensionGroup(
                    proper_exponents=(0, 0, 0, 0, 0, 0, 0),
                    log_dimension=None,
                    angle_dimension=None,
                    log_exponent=0.0,
                    angle_exponent=0.0
                )
            else:
                raise ValueError("Cannot take the trigonometric functions of a dimension with non-zero proper exponents or log dimension")
        else:
            if all(exp == 0 for exp in self.proper_exponents) and self.log_dimension is None:
                angle_exponent = self.angle_exponent - 1
                if abs(self.angle_exponent) < EPSILON:
                    angle_exponent = 0.0
                else:
                    raise ValueError("The dimension must have an angle dimension of one.")
                return DimensionGroup(
                    proper_exponents=self.angle_dimension.proper_exponents,
                    log_dimension=self.angle_dimension.log_dimension,
                    angle_dimension=self.angle_dimension.angle_dimension,
                    log_exponent=self.angle_dimension.log_exponent,
                    angle_exponent=angle_exponent
                )
            else:
                raise ValueError("Cannot exponentiate a dimension with non-zero proper exponents or angle dimension")
    
    def arc(self) -> "DimensionGroup":
        """
        (arcsin, arccos, arctan, arccot)
        Return the current dimension as an angle dimension of a new dimension group.
        """
        return DimensionGroup(
            proper_exponents=(0, 0, 0, 0, 0, 0, 0),
            log_dimension=None,
            angle_dimension=self,
            log_exponent=0.0,
            angle_exponent=1.0
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DimensionGroup):
            return False
        import numpy as np
        return bool(np.array_equal(self.proper_exponents, other.proper_exponents))

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, DimensionGroup):
            return True
        return not self.__eq__(other)

################################################################################
# Hash operations
################################################################################

    def __hash__(self) -> int:
        return hash((self.proper_exponents, self.angle_dimension, self.log_dimension))

################################################################################
# Properties
################################################################################

    @property
    def is_dimensionless(self) -> bool:
        return all(exp == 0 for exp in self.proper_exponents) and self.angle_dimension is None and self.log_dimension is None
    
    @property
    def is_proper_exponents_zero(self) -> bool:
        return all(exp == 0 for exp in self.proper_exponents)
    
    @property
    def is_angle_dimension(self) -> bool:
        return self.angle_dimension is not None

    @property
    def includes_angle(self) -> bool:
        return self.angle_dimension is not None

    @property
    def includes_log_level(self) -> bool:
        return self.log_dimension is not None
    
    @property
    def canonical_unit(self) -> "UnitGroup":
        if self in _CANONICAL_UNIT_CACHE:
            return _CANONICAL_UNIT_CACHE[self]
        else:
            unit_group = UnitGroup(self)
            _CANONICAL_UNIT_CACHE[self] = unit_group
            return unit_group
        
    @classmethod
    def dimensionless_dimension_group(cls) -> "DimensionGroup":
        return DIMENSIONLESS_DIMENSION_GROUP
    
################################################################################
# Compatibility
################################################################################

    def compatible_to(self, other: "DimensionGroup") -> bool:
        result = self.__eq__(other)
        return bool(result)

################################################################################
# String representation
################################################################################

    def __str__(self) -> str:
        return self.format_string()

    def format_string(self, subscript: str = "", as_fraction: bool = True) -> str:
        """
        String representation of the dimension group.

        Args:
            subscript: The subscript to use for the dimension group.
            as_fraction: Whether to represent the dimension group as a fraction.

        Returns:
            The string representation of the dimension group.
        """
        nominator_parts: list[str] = []
        denominator_parts: list[str] = []

        # Add proper dimensions
        for i, exp in enumerate(self.proper_exponents):
            if exp != 0:
                if exp == 1:
                    nominator_parts.append(DimensionSymbol.from_index(i).symbol)
                else:
                    exp_string: str = str(abs(exp))
                    if exp > 0:
                        nominator_parts.append(DimensionSymbol.from_index(i).symbol + subscript + f"^{exp_string}")
                    else:
                        if as_fraction:
                            if abs(exp) == 1:
                                denominator_parts.append(DimensionSymbol.from_index(i).symbol + subscript)
                            else:
                                denominator_parts.append(DimensionSymbol.from_index(i).symbol + subscript + f"^{exp_string}")
                        else:
                            if exp_string == "1":
                                nominator_parts.append(DimensionSymbol.from_index(i).symbol + subscript)
                            else:
                                nominator_parts.append(DimensionSymbol.from_index(i).symbol + subscript + f"^-{exp_string}")

        result_string: str = ""
        if len(nominator_parts) == 0 and len(denominator_parts) == 0:
            return ""
        elif len(nominator_parts) == 0 and len(denominator_parts) > 0:
            result_string = "1"
        else:
            result_string += "*".join(nominator_parts)

        if denominator_parts:
            if len(denominator_parts) == 1:
                result_string += " / " + denominator_parts[0]
            else:
                result_string += " / (" + "*".join(denominator_parts) + ")"

        def fun(dim_string: str, dim_exponent: float, s: str) -> str:
            if dim_string == "":
                dim_string = s
            else:
                dim_string = f"{s}(" + dim_string + ")"
            if dim_exponent == 1:
                dim_string = dim_string
            elif dim_exponent > 0:
                dim_string = dim_string + " ^ " + str(dim_exponent)
            elif dim_exponent == -1 and as_fraction:
                dim_string = "1 / " + dim_string
            else:
                dim_string = dim_string + " ^ " + str(dim_exponent)
            return dim_string

        if self.log_dimension is not None:
            result_string = fun(self.log_dimension.format_string(subscript, as_fraction), self.log_exponent, "dec")
        if self.angle_dimension is not None:
            result_string = fun(self.angle_dimension.format_string(subscript, as_fraction), self.angle_exponent, "deg")

        return result_string
    
    def __repr__(self) -> str:
        return self.format_string()
    
    @classmethod
    def parse_string(cls, string: str, relevant_subscript: str = "") -> "DimensionGroup":
        return DimensionGroup(
            proper_exponents=(0, 0, 0, 0, 0, 0, 0),
            log_dimension=None,
            angle_dimension=None,
            log_exponent=0.0,
            angle_exponent=0.0
        )

################################################################################
# JSON serialization
################################################################################

    def to_json(self, subscript: str = "") -> str:
        return self.format_string(subscript, as_fraction=True)

    @classmethod
    def from_json(cls, string: str) -> "DimensionGroup":
        return cls.parse_string(string)

################################################################################
# HDF5 serialization
################################################################################

    def to_hdf5(self, hdf5_group: h5py.Group, subscript: str = "") -> None:
        hdf5_group.attrs[subscript] = self.format_string(subscript, as_fraction=True)

    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group, subscript: str = "") -> "DimensionGroup":
        return cls.parse_string(hdf5_group.attrs[subscript]) # type: ignore
    
DIMENSIONLESS_DIMENSION_GROUP: Final[DimensionGroup] = DimensionGroup(
    proper_exponents=(0, 0, 0, 0, 0, 0, 0),
    log_dimension=None,
    angle_dimension=None,
    log_exponent=0.0,
    angle_exponent=0.0
)

DECADE_DIMENSION_GROUP: Final[DimensionGroup] = DimensionGroup(
    proper_exponents=(0, 0, 0, 0, 0, 0, 0),
    log_dimension=DIMENSIONLESS_DIMENSION_GROUP,
    angle_dimension=None,
    log_exponent=1.0,
    angle_exponent=0.0
)

ANGULAR_DIMENSION_GROUP: Final[DimensionGroup] = DimensionGroup(
    proper_exponents=(0, 0, 0, 0, 0, 0, 0),
    log_dimension=None,
    angle_dimension=DIMENSIONLESS_DIMENSION_GROUP,
    log_exponent=0.0,
    angle_exponent=1.0
)
