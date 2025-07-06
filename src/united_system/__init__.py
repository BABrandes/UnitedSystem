"""
UnitedSystem: A comprehensive physical units system for Python.

This package provides type-safe, efficient handling of physical quantities
with automatic unit conversion and dimensional analysis.
"""

from .real_united_scalar import RealUnitedScalar
from .real_united_array import RealUnitedArray
from .dimension import Dimension
from .unit import Unit
from .named_dimensions import NamedDimension, DimensionExponents
from .dataframe import UnitedDataframe  # Import from new modular dataframe package
from .arrays.int_array import IntArray
from .arrays.string_array import StringArray
from .arrays.timestamp_array import TimestampArray
from .arrays.bool_array import BoolArray
from .arrays.complex_array import ComplexArray
from .arrays.float_array import FloatArray

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
