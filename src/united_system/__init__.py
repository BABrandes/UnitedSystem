"""
UnitedSystem: A comprehensive physical units system for Python.

This package provides type-safe, efficient handling of physical quantities
with automatic unit conversion and dimensional analysis.
"""

from ..depreciated.real_united_scalar_depreciated import RealUnitedScalar
from ..depreciated.real_united_array_depreciated import RealUnitedArray
from ..depreciated.dimension import Dimension
from ..depreciated.unit import Unit
from .named_dimensions import NamedDimension, DimensionExponents
from .united_dataframe import UnitedDataframe
from .int_array import IntArray
from .string_array import StringArray
from .timestamp_array import TimestampArray
from .bool_array import BoolArray
from .complex_array import ComplexArray
from .float_array import FloatArray

__all__ = [
    'RealUnitedScalar',
    'RealUnitedArray',
    'Dimension',
    'Unit',
    'NamedDimension',
    'DimensionExponents',
    'UnitedDataframe',
    'IntArray',
    'StringArray',
    'TimestampArray',
    'BoolArray',
    'ComplexArray',
    'FloatArray'
]
__version__ = '0.1.0'
__author__ = 'Benedikt Axel Brandes'
__year__ = '2025'
