"""
String utility functions to avoid circular imports.
"""

import locale
from enum import Enum

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
                target_locale = POINT_AS_DECIMAL_SEPARATOR_LOCALE

        #Fourth, convert the string to a float using the target locale
        if target_locale:
            #Set the locale to the target locale
            locale.setlocale(locale.LC_NUMERIC, target_locale)
            #Convert the string to a float
            result = locale.atof(value)
            #Reset the locale to the original locale
            if current_locale:
                locale.setlocale(locale.LC_NUMERIC, current_locale)
            return result
        else:
            return float(value)

    except Exception as e:
        #Reset the locale to the original locale
        if current_locale:
            locale.setlocale(locale.LC_NUMERIC, current_locale)
        raise e 