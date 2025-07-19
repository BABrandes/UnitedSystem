from enum import Enum

class DimensionSymbol(Enum):
    MASS = ("M", 0)
    TIME = ("T", 1)
    LENGTH = ("L", 2)
    CURRENT = ("I", 3)
    TEMPERATURE = ("Θ", 4)
    AMOUNT = ("N", 5)
    LUMINOUS = ("J", 6)

    @property
    def symbol(self) -> str:
        return self.value[0]
    
    @property
    def index(self) -> int:
        return self.value[1]
    
    @classmethod
    def from_index(cls, index: int) -> "DimensionSymbol":
        for symbol in cls:
            if symbol.index == index:
                return symbol
        raise ValueError(f"No dimension symbol found for index {index}")