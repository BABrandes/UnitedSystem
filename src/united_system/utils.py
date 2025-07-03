import numpy as np
from typing import Type, Protocol, runtime_checkable
from enum import Enum
from typing import Callable
import locale
import h5py

@runtime_checkable
class JSONable(Protocol):
    def to_json(self) -> dict:
        ...
    @classmethod
    def from_json(cls: Type['JSONable'], data: dict) -> 'JSONable':
        ...

@runtime_checkable
class HDF5able(Protocol):
    def to_hdf5(self, hdf5_group: h5py.Group) -> None:
        ...
    @classmethod
    def from_hdf5(cls: Type['HDF5able'], hdf5_group: h5py.Group) -> 'HDF5able':
        ...

class Standard_Shape_Conserving_Filters(Enum):
    value: Callable[[np.ndarray], np.ndarray]

    ONLY_POSITIVE_INCLUDING_ZERO = lambda array: np.where(array >= 0, array, np.nan)
    ONLY_NEGATIVE_INCLUDING_ZERO = lambda array: np.where(array <= 0, array, np.nan)
    ONLY_POSITIVE_EXCLUDING_ZERO = lambda array: np.where(array > 0, array, np.nan)
    ONLY_NEGATIVE_EXCLUDING_ZERO = lambda array: np.where(array < 0, array, np.nan)
    ONLY_NON_ZERO = lambda array: np.where(array != 0, array, np.nan)

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
                case _:
                    raise ValueError(f"Invalid decimal separator: {decimal_separator}")
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
            raise LoggedException(f"No locale found for string: {value}")

        #Convert the string to a float
        return locale.atof(value)

    except locale.Error as e:
        raise ValueError(f"Could not convert '{value}' to float due to locale error", e)
    finally:
        #Reset the locale to the original locale
        if current_locale is not None:
            locale.setlocale(locale.LC_NUMERIC, current_locale)