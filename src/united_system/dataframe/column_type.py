from typing import TypeAlias, Literal, overload, Any, NamedTuple
from enum import Enum
from dataclasses import dataclass
import math

import pandas as pd
from pandas._typing import Dtype
from pandas import Timestamp
import numpy as np

from ..unit import Unit
from ..dimension import Dimension
from ..real_united_scalar import RealUnitedScalar
from ..complex_united_scalar import ComplexUnitedScalar
from ..real_united_array import RealUnitedArray
from ..complex_united_array import ComplexUnitedArray
from ..string_array import StringArray
from ..int_array import IntArray
from ..float_array import FloatArray
from ..bool_array import BoolArray
from ..timestamp_array import TimestampArray

PYTHON_SCALAR_TYPE: TypeAlias = float|complex|str|bool|int|Timestamp
SCALAR_TYPE: TypeAlias = PYTHON_SCALAR_TYPE|RealUnitedScalar|ComplexUnitedScalar|str|bool|int|Timestamp
NUMERIC_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar|int|float|complex|Timestamp
ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray|StringArray|IntArray|FloatArray|BoolArray|TimestampArray
UNITED_ARRAY_TYPE: TypeAlias = RealUnitedArray|ComplexUnitedArray
UNITED_SCALAR_TYPE: TypeAlias = RealUnitedScalar|ComplexUnitedScalar
ARRAY_STORAGE_TYPE: TypeAlias = float|complex|str|int|bool|Timestamp
NUMPY_STORAGE_TYPE: TypeAlias = np.float64|np.float32|np.float16|np.complex128|np.complex64|np.int64|np.int32|np.int16|np.int8|np.bool_|np.datetime64

@dataclass(frozen=True, slots=True)
class ColumnTypeInformation(NamedTuple):
    name: str
    has_unit: bool
    python_scalar_type: PYTHON_SCALAR_TYPE
    scalar_type: SCALAR_TYPE
    array_type: ARRAY_TYPE
    dataframe_storage_type: Dtype
    array_storage_type: ARRAY_STORAGE_TYPE
    numpy_storage_options: NUMPY_STORAGE_TYPE
    missing_values_in_dataframe: Any
    non_values_in_scalar: Any
    precision: int|None

class ColumnType(Enum):
    value: ColumnTypeInformation
    REAL_NUMBER_64 = ColumnTypeInformation(         name="Real United Numbers (64bit)",     has_unit=True,  python_scalar_type=float,         scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_types=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options={np.float64, np.float32, np.float16},     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=64)
    REAL_NUMBER_32 = ColumnTypeInformation(         name="Real United Numbers (32bit)",     has_unit=True,  python_scalar_type=float,         scalar_type=RealUnitedScalar,       array_type=RealUnitedArray,     dataframe_storage_types=pd.Float32Dtype(),  array_storage_type=float,       numpy_storage_options={np.float32, np.float64, np.float16},     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=32)
    COMPLEX_NUMBER_128 = ColumnTypeInformation(     name="Complex United Numbers (128bit)", has_unit=True,  python_scalar_type=complex,       scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_types=np.complex128,      array_storage_type=complex,     numpy_storage_options={np.complex128, np.complex64},            missing_values_in_dataframe=math.nan+1j, non_values_in_scalar=math.nan+1j,   precision=128)
    STRING = ColumnTypeInformation(                 name="Strings",                         has_unit=False, python_scalar_type=str,           scalar_type=str,                    array_type=StringArray,         dataframe_storage_types=pd.StringDtype(),   array_storage_type=str,         numpy_storage_options={np.str_},                                missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=None)
    INTEGER_64 = ColumnTypeInformation(             name="Integers",                        has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int64Dtype(),    array_storage_type=int,         numpy_storage_options={np.int64, np.int32, np.int16, np.int8},  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=64)
    INTEGER_32 = ColumnTypeInformation(             name="Integers",                        has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int32Dtype(),    array_storage_type=int,         numpy_storage_options={np.int32, np.int64, np.int16, np.int8},  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=32)
    INTEGER_16 = ColumnTypeInformation(             name="Integers",                        has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int16Dtype(),    array_storage_type=int,         numpy_storage_options={np.int16, np.int64, np.int32, np.int8},  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=16)
    INTEGER_8 = ColumnTypeInformation(              name="Integers",                        has_unit=False, python_scalar_type=int,           scalar_type=int,                    array_type=IntArray,            dataframe_storage_types=pd.Int8Dtype(),     array_storage_type=int,         numpy_storage_options={np.int8, np.int64, np.int32, np.int16},  missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=8)
    FLOAT_64 = ColumnTypeInformation(               name="Floats (64bit)",                  has_unit=True,  python_scalar_type=float,         scalar_type=float,                  array_type=FloatArray,          dataframe_storage_types=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options={np.float64, np.float32, np.float16},     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=64)
    FLOAT_32 = ColumnTypeInformation(               name="Floats (32bit)",                  has_unit=True,  python_scalar_type=float,         scalar_type=float,                  array_type=FloatArray,          dataframe_storage_types=pd.Float64Dtype(),  array_storage_type=float,       numpy_storage_options={np.float32, np.float64, np.float16},     missing_values_in_dataframe=pd.NA,       non_values_in_scalar=math.nan,      precision=32)
    COMPLEX_128 = ColumnTypeInformation(            name="Complex (128bit)",                has_unit=True,  python_scalar_type=complex,       scalar_type=ComplexUnitedScalar,    array_type=ComplexUnitedArray,  dataframe_storage_types=np.complex128,      array_storage_type=complex,     numpy_storage_options={np.complex128, np.complex64},            missing_values_in_dataframe=math.nan+1j, non_values_in_scalar=math.nan+1j,   precision=128)
    BOOL = ColumnTypeInformation(                   name="Bools",                           has_unit=False, python_scalar_type=bool,          scalar_type=bool,                   array_type=BoolArray,           dataframe_storage_types=pd.BooleanDtype(),  array_storage_type=bool,        numpy_storage_options={np.bool_},                               missing_values_in_dataframe=pd.NA,       non_values_in_scalar=None,          precision=None)
    TIMESTAMP = ColumnTypeInformation(              name="Timestamps",                      has_unit=False, python_scalar_type=Timestamp,     scalar_type=Timestamp,              array_type=TimestampArray,      dataframe_storage_types=pd.Timestamp,       array_storage_type=Timestamp,   numpy_storage_options={np.datetime64},                          missing_values_in_dataframe=pd.NaT,      non_values_in_scalar=None,          precision=None)

    @property
    def has_unit(self) -> bool:
        return self.value.has_unit
    
    @property
    def is_numeric(self) -> bool:
        return self in {ColumnType.REAL_NUMBER_64, ColumnType.REAL_NUMBER_32, ColumnType.FLOAT_64, ColumnType.FLOAT_32, ColumnType.COMPLEX_128, ColumnType.COMPLEX_NUMBER_128, ColumnType.INTEGER_64, ColumnType.INTEGER_32, ColumnType.INTEGER_16, ColumnType.INTEGER_8, ColumnType.TIMESTAMP}
    
    @property
    def is_not_numeric(self) -> bool:
        return not self.is_numeric

    def get_as_python_scalar_values(self, series: pd.Series) -> list[PYTHON_SCALAR_TYPE]:
        """
        Return the series as a list of the correct python scalar type defined in ColumnType.
        """
        casted_series = self.cast_for_dataframe(series)
        non_value = self.value.non_values_in_scalar
        if non_value is None:
            if casted_series.isna().any():
                raise ValueError(f"Missing values detected in series, but ColumnType {self.name} does not define a non-value scalar representation.")
            return casted_series.tolist()
        else:
            return [v if pd.notna(v) else non_value for v in casted_series]

    def get_as_numpy_array(self, series: pd.Series) -> np.ndarray:
        """
        Return the series as a NumPy array with the correct storage dtype.
        """
        casted_series = self.cast_for_dataframe(series)
        return casted_series.to_numpy()

    @overload
    def cast_for_dataframe(self, value: PYTHON_SCALAR_TYPE) -> Any: ...
    @overload
    def cast_for_dataframe(self, value: SCALAR_TYPE) -> Any: ...
    @overload
    def cast_for_dataframe(self, value: pd.Series) -> pd.Series: ...
    @overload
    def cast_for_dataframe(self, value: tuple[Any, ...]) -> tuple[Any, ...]: ...
    @overload
    def cast_for_dataframe(self, value: np.ndarray) -> np.ndarray: ...
    def cast_for_dataframe(self, value: PYTHON_SCALAR_TYPE|SCALAR_TYPE|pd.Series|tuple[Any, ...]|np.ndarray) -> Any|pd.Series|tuple[Any, ...]|np.ndarray:

        if isinstance(value, pd.Series):
            series: pd.Series = value
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.FLOAT_64:
                    return series.astype("Float64")  # pandas nullable float
                case ColumnType.REAL_NUMBER_32 | ColumnType.FLOAT_32:
                    return series.astype("Float32")  # pandas nullable float32
                case ColumnType.INTEGER_64:
                    return series.astype("Int64")
                case ColumnType.INTEGER_32:
                    return series.astype("Int32")
                case ColumnType.INTEGER_16:
                    return series.astype("Int16")
                case ColumnType.INTEGER_8:
                    return series.astype("Int8")
                case ColumnType.COMPLEX_128 | ColumnType.COMPLEX_NUMBER_128:
                    return series.astype(np.complex128)
                case ColumnType.STRING:
                    return series.astype("string")
                case ColumnType.BOOL:
                    return series.astype("boolean")
                case ColumnType.TIMESTAMP:
                    return pd.to_datetime(series, errors="coerce")  # preserves NaT
                case _:
                    raise TypeError(f"Unsupported column type: {self}")
        elif isinstance(value, tuple[Any, ...]):
            return tuple(self.cast_for_dataframe_storage(v) for v in value)
        elif isinstance(value, np.ndarray):
            array: np.ndarray = value
            match self:
                    case ColumnType.REAL_NUMBER_64 | ColumnType.FLOAT_64:
                        return array.astype("Float64")
                    case ColumnType.REAL_NUMBER_32 | ColumnType.FLOAT_32:
                        return array.astype("Float32")
                    case ColumnType.INTEGER_64:
                        return array.astype("Int64")
                    case ColumnType.INTEGER_32:
                        return array.astype("Int32")
                    case ColumnType.INTEGER_16:
                        return array.astype("Int16")
                    case ColumnType.INTEGER_8:
                        return array.astype("Int8")
                    case ColumnType.COMPLEX_128 | ColumnType.COMPLEX_NUMBER_128:
                        return array.astype(np.complex128)
                    case ColumnType.STRING:
                        return array.astype("string")
                    case ColumnType.BOOL:
                        return array.astype("boolean")
                    case ColumnType.TIMESTAMP:
                        return pd.to_datetime(array, errors="coerce").to_numpy()
                    case _:
                        raise TypeError(f"Unsupported column type: {self}")
        elif isinstance(value, SCALAR_TYPE):
            match self:
                case ColumnType.REAL_NUMBER_64:
                    return value.canonical_value
                case ColumnType.REAL_NUMBER_32:
                    return np.float32(value.float_value_canoninical_units)
                case ColumnType.COMPLEX_NUMBER_128:
                    return value.canonical_value
                case ColumnType.INTEGER_64:
                    return pd.Int64Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.INTEGER_32:
                    return pd.Int32Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.INTEGER_16:
                    return pd.Int16Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.INTEGER_8:
                    return pd.Int8Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.COMPLEX_128 | ColumnType.COMPLEX_NUMBER_128:
                    return complex(value)
                case ColumnType.STRING:
                    return str(value) if value is not pd.NA else pd.NA
                case ColumnType.BOOL:
                    return bool(value) if value is not pd.NA else pd.NA
                case ColumnType.TIMESTAMP:
                    return pd.Timestamp(value) if value is not pd.NA else pd.NaT
                case ColumnType.FLOAT_64:
                    return value.canonical_value
                case ColumnType.FLOAT_32:
                    return np.float32(value)
                case ColumnType.COMPLEX_128:
                    return value.canonical_value
                case _:
                    raise TypeError(f"Unsupported column type: {self}")
        else:
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.FLOAT_64:
                    return float(value)
                case ColumnType.REAL_NUMBER_32 | ColumnType.FLOAT_32:
                    return np.float32(value)
                case ColumnType.INTEGER_64:
                    return pd.Int64Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.INTEGER_32:
                    return pd.Int32Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.INTEGER_16:
                    return pd.Int16Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.INTEGER_8:
                    return pd.Int8Dtype().type(value) if value is not pd.NA else pd.NA
                case ColumnType.COMPLEX_128 | ColumnType.COMPLEX_NUMBER_128:
                    return complex(value)
                case ColumnType.STRING:
                    return str(value) if value is not pd.NA else pd.NA
                case ColumnType.BOOL:
                    return bool(value) if value is not pd.NA else pd.NA
                case ColumnType.TIMESTAMP:
                    return pd.Timestamp(value) if value is not pd.NA else pd.NaT
                case _:
                    raise TypeError(f"Unsupported column type: {self}")
                
    @classmethod
    def from_dtype(cls, dtype: pd.Series | pd.api.extensions.ExtensionDtype | np.dtype | str, has_unit: bool = False) -> "ColumnType":
        """
        Infer the ColumnType from a pandas Series, dtype, or dtype string.
        Uses `has_unit` to distinguish between united and raw types.
        """
        if isinstance(dtype, pd.Series):
            dtype = dtype.dtype

        dtype_str = str(dtype).lower()

        candidates = [
            col_type
            for col_type in cls
            if col_type.value.has_unit == has_unit
        ]

        for column_type in candidates:
            expected_dtype = column_type.value.dataframe_storage_type

            if isinstance(expected_dtype, pd.api.extensions.ExtensionDtype):
                if dtype_str == str(expected_dtype).lower():
                    return column_type
            elif isinstance(expected_dtype, type):
                try:
                    if isinstance(dtype, expected_dtype):
                        return column_type
                except TypeError:
                    pass  # e.g., str vs dtype mismatch
            elif dtype_str == str(expected_dtype).lower():
                return column_type

        raise ValueError(f"No matching ColumnType for dtype: {dtype} with has_unit={has_unit}")

    @overload
    @classmethod
    def infer_approbiate_column_type(cls, scalar_type: SCALAR_TYPE, precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": ...
    @overload
    @classmethod
    def infer_approbiate_column_type(cls, array_type: ARRAY_TYPE, precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType": ...
    @classmethod
    def infer_approbiate_column_type(cls, Scalar_or_array_type: SCALAR_TYPE|ARRAY_TYPE, precision: Literal[8, 16, 32, 64, 128, 256]|None=None) -> "ColumnType":
        options: list[ColumnType] = []
        for column_type in cls:
            if column_type.value.scalar_type == Scalar_or_array_type:
                options.append(column_type)
            if column_type.value.array_type == Scalar_or_array_type:
                options.append(column_type)
        if len(options) == 0:
            raise ValueError(f"No column type found for UnitedScalar type {Scalar_or_array_type}.")
        elif len(options) == 1:
            return options[0]
        else:
            if precision is None:
                raise ValueError(f"Multiple column types found for UnitedScalar type {Scalar_or_array_type}.")
            for option in options:
                if option.value.precision == precision:
                    return option
            raise ValueError(f"No column type found for UnitedScalar type {Scalar_or_array_type} with precision {precision}. Tried: {[ct.name for ct in options]}")
        
    @overload
    def check_compatibility(self, scalar_type: SCALAR_TYPE) -> bool: ...
    @overload
    def check_compatibility(self, array_type: ARRAY_TYPE) -> bool: ...
    def check_compatibility(self, scalar_or_array_type: SCALAR_TYPE | ARRAY_TYPE) -> bool:
        return (
            scalar_or_array_type == self.value.scalar_type
            or scalar_or_array_type == self.value.array_type
        )
    
    def create_scalar_from_value(self, canonical_value: float|complex|str|bool|int|Timestamp|None, dimension_or_display_unit: Dimension|Unit|None = None) -> SCALAR_TYPE:

        if canonical_value is None:
            match self:
                case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                    return RealUnitedScalar.create_from_canonical_value(np.nan, dimension_or_display_unit)
                case ColumnType.COMPLEX_NUMBER_128:
                    raise ValueError(f"Complex United Scalar is not yet supported.")
                case ColumnType.STRING:
                    raise ValueError(f"Missing values are not supported for string types.")
                case ColumnType.BOOL:
                    raise ValueError(f"Missing values are not supported for boolean types.")
                case ColumnType.TIMESTAMP:
                    return pd.NaT
                case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                    raise ValueError(f"Missing values are not supported for integer types.")
                case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                    return math.nan
                case ColumnType.COMPLEX_128:
                    return math.nan + 1j
                case _:
                    raise ValueError(f"Unsupported column type: {self}")
        
        if isinstance(canonical_value, float) and self is not ColumnType.FLOAT_64 and self is not ColumnType.FLOAT_32 and self is not ColumnType.REAL_NUMBER_64 and self is not ColumnType.REAL_NUMBER_32:
            raise ValueError(f"Invalid canonical value: {canonical_value} for column type {self.name}.")

        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if isinstance(dimension_or_display_unit, Unit):
                    display_unit: Unit = dimension_or_display_unit
                    return RealUnitedScalar.create_from_canonical_value(canonical_value, display_unit)
                elif isinstance(dimension_or_display_unit, Dimension):
                    dimension: Dimension = dimension_or_display_unit
                    return RealUnitedScalar.create_from_canonical_value(canonical_value, dimension)
                else:
                    raise ValueError(f"Invalid dimension or display unit: {dimension_or_display_unit}")
            case ColumnType.COMPLEX_NUMBER_128:
                raise ValueError(f"Complex United Scalar is not yet supported.")
            case ColumnType.STRING:
                return str(canonical_value)
            case ColumnType.BOOL:
                return bool(canonical_value)
            case ColumnType.TIMESTAMP:
                return Timestamp(canonical_value)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return int(canonical_value)
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return float(canonical_value)
            case ColumnType.COMPLEX_128:
                return complex(canonical_value)
            case _:
                raise ValueError(f"Unsupported column type: {self}")
            
    def create_array_from_canonical_values(self, canonical_values: np.ndarray, dimension_or_display_unit: Dimension|Unit|None = None) -> ARRAY_TYPE:
        match self:
            case ColumnType.REAL_NUMBER_64 | ColumnType.REAL_NUMBER_32:
                if isinstance(dimension_or_display_unit, Unit):
                    display_unit: Unit = dimension_or_display_unit
                    return RealUnitedArray.create_from_canonical_values(canonical_values, display_unit)
                elif isinstance(dimension_or_display_unit, Dimension):
                    dimension: Dimension = dimension_or_display_unit
                    return RealUnitedArray.create_from_canonical_values(canonical_values, dimension)
                else:
                    raise ValueError(f"Invalid dimension or display unit: {dimension_or_display_unit}")
            case ColumnType.COMPLEX_NUMBER_128:
                raise ValueError(f"Complex United Array is not yet supported.")
            case ColumnType.STRING:
                return StringArray(canonical_values)
            case ColumnType.BOOL:
                return BoolArray(canonical_values)
            case ColumnType.TIMESTAMP:
                return TimestampArray(canonical_values)
            case ColumnType.COMPLEX_128:
                raise ValueError(f"Complex United Array is not yet supported.")
            case ColumnType.FLOAT_64 | ColumnType.FLOAT_32:
                return FloatArray(canonical_values)
            case ColumnType.INTEGER_64 | ColumnType.INTEGER_32 | ColumnType.INTEGER_16 | ColumnType.INTEGER_8:
                return IntArray(canonical_values)
            case _:
                raise ValueError(f"Unsupported column type: {self}")