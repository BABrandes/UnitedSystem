"""
Core module for UnitedDataframe.

This module provides the essential infrastructure that all other modules depend on.
"""

from .base import UnitedDataframeCore
from .validation import ValidationMixin

__all__ = [
    'UnitedDataframeCore',
    'ValidationMixin',
]
