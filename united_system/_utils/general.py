import numpy as np
from typing import Protocol, runtime_checkable, Any, Generic, TypeVar, TypeAlias
from enum import Enum
from typing import Callable
import locale
import h5py
from pandas import Timestamp
from .._scalars.base_scalar import BaseScalar
from .._arrays.real_united_array import RealUnitedArray
from .._arrays.complex_united_array import ComplexUnitedArray
from .._arrays.string_array import StringArray
from .._arrays.int_array import IntArray
from .._arrays.float_array import FloatArray
from .._arrays.bool_array import BoolArray
from .._arrays.timestamp_array import TimestampArray
from .._arrays.complex_array import ComplexArray

VALUE_TYPE: TypeAlias = float|complex|str|bool|int|Timestamp
SCALAR_TYPE: TypeAlias = VALUE_TYPE|BaseScalar
ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray|ComplexArray

T = TypeVar("T", covariant=True)

@runtime_checkable
class JSONable(Protocol, Generic[T]):
    """
    Protocol for JSONable objects.
    """

    def to_json(self) -> dict[str, Any]:
        ...
    @classmethod
    def from_json(cls, data: dict[str, Any]) -> T:
        ...

@runtime_checkable
class HDF5able(Protocol, Generic[T]):
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        ...
    @classmethod
    def from_hdf5(cls, hdf5_group: h5py.Group) -> T:
        ...

class Standard_Shape_Conserving_Filters(Enum):
    value: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]] # type: ignore

    ONLY_POSITIVE_INCLUDING_ZERO = lambda array: np.where(array >= 0, array, np.nan) # type: ignore
    ONLY_NEGATIVE_INCLUDING_ZERO = lambda array: np.where(array <= 0, array, np.nan) # type: ignore
    ONLY_POSITIVE_EXCLUDING_ZERO = lambda array: np.where(array > 0, array, np.nan) # type: ignore
    ONLY_NEGATIVE_EXCLUDING_ZERO = lambda array: np.where(array < 0, array, np.nan) # type: ignore
    ONLY_NON_ZERO = lambda array: np.where(array != 0, array, np.nan) # type: ignore

class Decimal_Seperator_Keys(Enum):
    COMMA = "Comma"
    POINT = "Point"

def str_to_float(value: str, decimal_separator: Decimal_Seperator_Keys|None = None) -> float:
    """
    If no decimal separator is provided, the function will try to detect the decimal separator automatically.
    *** Attention: This function will not work on non-decimal strings with thousand separators! ***

    If a decimal separator is provided, the function will use the provided decimal separator to convert the  string to a float.

    """

    current_locale: str|None = None

    try:
        #Remove any whitespace
        value = value.strip()

        #First, check if the string contains neither '.' nor ','
        if not "." in value and not "," in value:
            return float(value)
        
        POINT_AS_DECIMAL_SEPARATOR_LOCALE: str = 'en_US.UTF-8'
        COMMA_AS_DECIMAL_SEPARATOR_LOCALE: str = 'de_DE.UTF-8'

        current_locale_tuple: tuple[str|None, str|None] = locale.getlocale(locale.LC_NUMERIC)
        current_locale: str|None = current_locale_tuple[0] if current_locale_tuple else None
        target_locale: str|None = None

        #Second, check if the decimal separator is provided
        if decimal_separator:
            match decimal_separator:
                case Decimal_Seperator_Keys.COMMA:
                    target_locale = COMMA_AS_DECIMAL_SEPARATOR_LOCALE
                case Decimal_Seperator_Keys.POINT:
                    target_locale = POINT_AS_DECIMAL_SEPARATOR_LOCALE

        #Third, check if the string contains '.' and ',
        else:
            #Check if the string contains '.' and ',
            if "." in value and "," in value:
                #If the string contains both '.' and ',', check which comes first
                if value.index(".") < value.index(","):
                    #If '.' comes first, use the locale to convert the string to a float
                    target_locale = POINT_AS_DECIMAL_SEPARATOR_LOCALE
                else:
                    #If ',' comes first, use the locale to convert the string to a float
                    target_locale = COMMA_AS_DECIMAL_SEPARATOR_LOCALE
            #Check if the string contains '.' or ','
            elif "." in value:
                target_locale = POINT_AS_DECIMAL_SEPARATOR_LOCALE
            elif "," in value:
                target_locale = COMMA_AS_DECIMAL_SEPARATOR_LOCALE
            #Just added for completeness, but should be unreachable
            else:
                raise ValueError("Should never ever happen, but it did.")

        #Set the locale to the target locale
        if target_locale:
            locale.setlocale(locale.LC_NUMERIC, target_locale)
        else:
            raise ValueError(f"No locale found for string: {value}")

        #Convert the string to a float
        return locale.atof(value)

    except locale.Error as e:
        raise ValueError(f"Could not convert '{value}' to float due to locale error", e)
    finally:
        #Reset the locale to the original locale
        if current_locale is not None:
            locale.setlocale(locale.LC_NUMERIC, current_locale)

def segment_numpy_arrays_by_key_array(key_array: np.ndarray, *data_arrays: np.ndarray) -> list[tuple[np.ndarray, ...]]:
    """
    Splits multiple data arrays into segments, based on contiguous blocks of the same value in key_array.

    Parameters:
    - *data_arrays: multiple numpy arrays of the same length as key_array.
    - key_array: a 1D numpy array of labels, whose contiguous values define segment boundaries.

    Returns:
    - A list of tuples, each containing one segment from each data array, including the key array as the first element (in the same order).
    """
    if not all(len(arr) == len(key_array) for arr in data_arrays):
        raise ValueError("All input arrays must have the same length as the key_array")

    # Find start and end indices of each segment
    change_indices = np.flatnonzero(np.diff(key_array)) + 1
    segment_indices = np.concatenate(([0], change_indices, [len(key_array)]))

    segments: list[tuple[np.ndarray, ...]] = []
    for start, end in zip(segment_indices[:-1], segment_indices[1:]):
        segment = (key_array[start:end],) + tuple(arr[start:end] for arr in data_arrays)
        segments.append(segment)

    return segments