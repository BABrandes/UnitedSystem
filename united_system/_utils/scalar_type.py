from typing import TypeAlias, TYPE_CHECKING, Union
from .value_type import VALUE_TYPE
from pandas import Timestamp

if TYPE_CHECKING:
    from .._scalars.real_united_scalar import RealUnitedScalar
    from .._scalars.complex_united_scalar import ComplexUnitedScalar

# Use string literals to avoid circular imports
SCALAR_TYPE: TypeAlias = Union[VALUE_TYPE, "RealUnitedScalar", "ComplexUnitedScalar", int, float, complex, Timestamp]

SCALAR_TYPE_RUNTIME: tuple[type, ...] = (float, complex, str, bool, int, Timestamp)  # Runtime types without circular imports

NUMERIC_SCALAR_TYPE: TypeAlias = Union["RealUnitedScalar", "ComplexUnitedScalar", int, float, complex, Timestamp]