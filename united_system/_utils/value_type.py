from typing import TypeAlias
from pandas import Timestamp

# Basic value types that don't depend on scalar classes
VALUE_TYPE: TypeAlias = float|complex|str|bool|int|Timestamp

# Runtime type tuples for basic types
VALUE_TYPE_RUNTIME: tuple[type, ...] = (float, complex, str, bool, int, Timestamp) 