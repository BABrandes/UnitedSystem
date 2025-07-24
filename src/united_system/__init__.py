"""
UnitedSystem: A comprehensive physical units system for Python.

This package provides type-safe, efficient handling of physical quantities
with automatic unit conversion and dimensional analysis.
"""

from .real_united_scalar import RealUnitedScalar
from .real_united_array import RealUnitedArray
from .dimension import Dimension
from .unit import Unit
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
