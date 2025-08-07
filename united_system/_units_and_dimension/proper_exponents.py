from dataclasses import dataclass, field
from typing import Tuple, Union, TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .unit_element import UnitElement
    from .named_quantity import NamedQuantity

@dataclass(frozen=True, slots=True)
class ProperExponents:
    """
    Dataclass representing the exponents of base dimensions.
    
    Each field represents the exponent of a base dimension:
    - mass: Exponent of mass (M)
    - time: Exponent of time (T)
    - length: Exponent of length (L)
    - current: Exponent of electric current (I)
    - temperature: Exponent of temperature (Θ)
    - amount: Exponent of amount of substance (N)
    - luminous_intensity: Exponent of luminous intensity (J)
    - angle: Exponent of angle (A)
    - log_level: Exponent for logarithmic dimensions
    """
    mass: float = field(default=0.0)
    time: float = field(default=0.0)
    length: float = field(default=0.0)
    current: float = field(default=0.0)
    temperature: float = field(default=0.0)
    amount: float = field(default=0.0)
    luminous_intensity: float = field(default=0.0)
    angle: float = field(default=0.0)

    @property
    def proper_exponents(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """Return the exponents as a tuple (excluding angle and log_level)."""
        return (self.mass, self.time, self.length, self.current, self.temperature, self.amount, self.luminous_intensity, self.angle)
    
    @staticmethod
    def same_proper_exponents(proper_exponents_1: Union[Tuple[float, float, float, float, float, float, float, float], "ProperExponents"], proper_exponents_2: Union[Tuple[float, float, float, float, float, float, float, float], "ProperExponents"]) -> bool:
        """Check if two proper exponents are the same."""
        if isinstance(proper_exponents_1, ProperExponents):
            proper_exponents_1 = proper_exponents_1.proper_exponents
        if isinstance(proper_exponents_2, ProperExponents):
            proper_exponents_2 = proper_exponents_2.proper_exponents
        return proper_exponents_1 == proper_exponents_2
    
    @staticmethod
    def proper_exponents_of_unit_elements(elements: Sequence["UnitElement"]|Sequence["NamedQuantity"]) -> tuple[float, float, float, float, float, float, float, float]:
        """
        Calculate the proper exponents for a group of unit elements.
        
        Args:
            elements: Tuple of unit elements or named quantities
            
        Returns:
            Tuple of proper exponents
        """

        from .unit_symbol import UnitSymbol, LogDimensionSymbol
        from .unit_element import UnitElement
        from .named_quantity import NamedQuantity

        exponents: list[float] = [0.0] * 8
        for element in elements:
            if isinstance(element, UnitElement):
                unit_symbol: UnitSymbol|LogDimensionSymbol = element.unit_symbol
                element_exponent = element.exponent
            elif isinstance(element, NamedQuantity): # type: ignore
                unit_symbol = element.unit_element.unit_symbol # type: ignore
                element_exponent = 1.0
            else:
                raise TypeError(f"Unknown element type: {type(element)}")
            for i, exponent in enumerate(unit_symbol.proper_exponents): # type: ignore
                exponents[i] += exponent * element_exponent
        return tuple(exponents) # type: ignore