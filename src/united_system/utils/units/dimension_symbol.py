from enum import Enum
from typing import Optional

from ...named_quantity import NamedQuantity
from .unit_symbol import UnitSymbol

class DimensionSymbol(Enum):
    MASS = ("M", 0, NamedQuantity.MASS)
    TIME = ("T", 1, NamedQuantity.TIME)
    LENGTH = ("L", 2, NamedQuantity.LENGTH)
    CURRENT = ("I", 3, NamedQuantity.CURRENT)
    TEMPERATURE = ("Θ", 4, NamedQuantity.TEMPERATURE)
    AMOUNT = ("N", 5, NamedQuantity.AMOUNT_OF_SUBSTANCE)
    LUMINOUS = ("J", 6, NamedQuantity.LUMINOUS_INTENSITY)
    ANGLE = ("A", 7, NamedQuantity.ANGLE)

    @property
    def symbol(self) -> str:
        return self.value[0]
    
    @property
    def index(self) -> int:
        return self.value[1]
    
    @property
    def named_quantity(self) -> NamedQuantity:
        return self.value[2]
    
    @property
    def base_unit_prefix(self) -> str:
        if not hasattr(self, "_base_unit_prefix"):
            from .unit_element import UnitElement
            unit_element: Optional[UnitElement] = self.named_quantity.unit_element
            if unit_element is not None:
                self._base_unit_prefix: str = unit_element.prefix
            else:
                raise AssertionError("Named quantity has no unit element.")
        return self._base_unit_prefix
    
    @property
    def base_unit_symbol(self) -> UnitSymbol:
        if not hasattr(self, "_base_unit_symbol"):
            from .unit_symbol import UnitSymbol
            from .unit_element import UnitElement
            from .unit_symbol import LogDimensionSymbol
            unit_element: Optional[UnitElement] = self.named_quantity.unit_element
            if unit_element is not None:
                unit_symbol: UnitSymbol|LogDimensionSymbol = unit_element.unit_symbol
                if isinstance(unit_symbol, LogDimensionSymbol):
                    raise AssertionError("Named quantity has no unit element.")
                self._base_unit_symbol: UnitSymbol = unit_symbol
            else:
                raise AssertionError("Named quantity has no unit element.")
        return self._base_unit_symbol

    @classmethod
    def get_symbol(cls, index: int) -> str:
        """Get the symbol for a given index."""
        for dimension in cls:
            if dimension.value[1] == index:
                return dimension.value[0]
        raise ValueError("Invalid index.")
    
    @classmethod
    def get_index(cls, symbol: str) -> int:
        """Get the index for a given symbol."""
        for dimension in cls:
            if dimension.value[0] == symbol:
                return dimension.value[1]
        raise ValueError("Invalid symbol.")
    
    @classmethod
    def is_dimension_symbol(cls, symbol: str) -> bool:
        """Check if a symbol is a dimension symbol."""
        for dimension in cls:
            if dimension.value[0] == symbol:
                return True
        return False
    
BASE_DIMENSION_SYMBOLS = [
    DimensionSymbol.MASS,
    DimensionSymbol.TIME,
    DimensionSymbol.LENGTH,
    DimensionSymbol.CURRENT,
    DimensionSymbol.TEMPERATURE,
    DimensionSymbol.AMOUNT,
    DimensionSymbol.LUMINOUS,
    DimensionSymbol.ANGLE,
]