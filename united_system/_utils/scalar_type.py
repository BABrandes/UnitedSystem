from typing import TypeAlias, TYPE_CHECKING, Union, Any
from .._scalars.base_scalar import BaseScalar
from .value_type import VALUE_TYPE
from pandas import Timestamp

if TYPE_CHECKING:
    
    from .._scalars.real_united_scalar import RealUnitedScalar
    from .._scalars.complex_united_scalar import ComplexUnitedScalar


SCALAR_TYPE: TypeAlias = Union[VALUE_TYPE, "RealUnitedScalar", "ComplexUnitedScalar", int, float, complex, Timestamp]
SCALAR_TYPE_RUNTIME: tuple[type, ...] = (float, complex, str, bool, int, Timestamp, BaseScalar)

NUMERIC_SCALAR_TYPE: TypeAlias = Union["RealUnitedScalar", "ComplexUnitedScalar", int, float, complex, Timestamp]
NUMERIC_SCALAR_TYPE_RUNTIME: tuple[type, ...] = (float, complex, int, Timestamp, BaseScalar)

@staticmethod
def is_scalar(item: Any) -> bool:
    return isinstance(item, SCALAR_TYPE_RUNTIME)
    
@staticmethod
def is_numeric_scalar(item: Any) -> bool:
    return isinstance(item, NUMERIC_SCALAR_TYPE_RUNTIME)
    