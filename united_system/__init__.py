"""
UnitedSystem: A comprehensive physical units system for Python.

This package provides type-safe, efficient handling of physical quantities
with automatic unit conversion and dimensional analysis.
"""

from ._scalars.real_united_scalar import RealUnitedScalar
from ._arrays.real_united_array import RealUnitedArray
from ._units_and_dimension.dimension import Dimension
from ._units_and_dimension.unit import Unit
from ._dataframe.united_dataframe import UnitedDataframe
from ._arrays.int_array import IntArray
from ._arrays.string_array import StringArray
from ._arrays.timestamp_array import TimestampArray
from ._arrays.bool_array import BoolArray
from ._arrays.complex_array import ComplexArray
from ._arrays.float_array import FloatArray
from ._dataframe.column_type import ColumnType as DataframeColumnType
from ._dataframe.column_key import ColumnKey as DataframeColumnKey
from ._units_and_dimension.named_quantity import NamedQuantity
from ._units_and_dimension.unit_symbol import UnitSymbol
from ._units_and_dimension.unit_prefix import UnitPrefix
from ._units_and_dimension.has_unit_protocol import HasUnit
from ._utils.value_type import VALUE_TYPE
from ._utils.scalar_type import SCALAR_TYPE
from ._utils.array_type import ARRAY_TYPE
from ._dataframe.internal_dataframe_name_formatter import InternalDataFrameColumnNameFormatter, SimpleInternalDataFrameNameFormatter
from ._units_and_dimension.dimension import DIMENSIONLESS_DIMENSION, ANGLE_DIMENSION


__all__ = [
    'RealUnitedScalar',
    'RealUnitedArray',
    'Dimension',
    'Unit',
    'UnitedDataframe',
    'NamedQuantity',
    'IntArray',
    'StringArray',
    'TimestampArray',
    'BoolArray',
    'ComplexArray',
    'FloatArray',
    'DataframeColumnType',
    'DataframeColumnKey',
    'UnitSymbol',
    'UnitPrefix',
    'HasUnit',
    'VALUE_TYPE',
    'SCALAR_TYPE',
    'ARRAY_TYPE',
    'InternalDataFrameColumnNameFormatter',
    'SimpleInternalDataFrameNameFormatter',
    'DIMENSIONLESS_DIMENSION',
    'ANGLE_DIMENSION'
]
__version__ = '0.1.1'
__author__ = 'Benedikt Axel Brandes'
__year__ = '2025'
