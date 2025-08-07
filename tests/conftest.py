"""
Pytest configuration file for UnitedSystem tests.

This file ensures that the tests directory is properly recognized as a pytest package
and handles any necessary setup for test imports.
"""

import sys
import os
from pathlib import Path

# Add the tests directory to the Python path
tests_dir = Path(__file__).parent
sys.path.insert(0, str(tests_dir))

# Import TestColumnKey so it's available to all tests
from test_dataframe import TestColumnKey 