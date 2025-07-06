"""
UnitedSystem: A comprehensive physical units system for Python.

This package provides type-safe, efficient handling of physical quantities
with automatic unit conversion and dimensional analysis.
"""

from .real_scalar import RealScalar
from .real_array import RealArray
from .dimension import Dimension
from .unit import Unit
from .named_dimensions import NamedDimension, DimensionExponents

__all__ = ['RealScalar', 'RealArray', 'Dimension', 'Unit', 'NamedDimension', 'DimensionExponents', 'int_array', 'string_array']
__version__ = '0.1.0'
__author__ = 'Benedikt Axel Brandes'
__year__ = '2025'
