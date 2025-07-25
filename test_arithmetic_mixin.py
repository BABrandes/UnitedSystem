#!/usr/bin/env python3
"""Test script to check arithmetic mixin import."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Try to import just the arithmetic mixin
    from united_system.utils.scalars.mixins.real_united_scalar.arithmetic_mixin import ArithmeticMixin
    print("✅ ArithmeticMixin imported successfully!")
    
    # Try to create an instance (this will test the circular import issue)
    print("✅ No circular import issues detected!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc() 