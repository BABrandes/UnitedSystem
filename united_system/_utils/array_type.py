from typing import TypeAlias, TYPE_CHECKING, Any
from .._arrays.base_array import BaseArray

if TYPE_CHECKING:
    from .._arrays.real_united_array import RealUnitedArray
    from .._arrays.complex_united_array import ComplexUnitedArray
    from .._arrays.string_array import StringArray
    from .._arrays.int_array import IntArray
    from .._arrays.float_array import FloatArray
    from .._arrays.bool_array import BoolArray
    from .._arrays.timestamp_array import TimestampArray
    from .._arrays.complex_array import ComplexArray


ARRAY_TYPE: TypeAlias = "RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray|ComplexArray"
ARRAY_TYPE_RUNTIME: tuple[type, ...] = (BaseArray,)

@staticmethod
def is_array(item: Any) -> bool:
    return isinstance(item, ARRAY_TYPE_RUNTIME)