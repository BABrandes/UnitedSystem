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
from ._dataframe.column_key import ColumnKey
from ._dataframe.column_type import ColumnType, ARRAY_TYPE, LOWLEVEL_TYPE, SCALAR_TYPE
from ._units_and_dimension.named_quantity import NamedDimension

__all__ = [
    'RealUnitedScalar',
    'RealUnitedArray',
    'Dimension',
    'Unit',
    'UnitedDataframe',
    'NamedDimension',
    'IntArray',
    'StringArray',
    'TimestampArray',
    'BoolArray',
    'ComplexArray',
    'FloatArray',
    'ColumnKey',
    'ColumnType',
    'ARRAY_TYPE',
    'LOWLEVEL_TYPE',
    'SCALAR_TYPE',
]
__version__ = '0.1.0'
__author__ = 'Benedikt Axel Brandes'
__year__ = '2025'
