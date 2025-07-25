#!/usr/bin/env python3
"""Isolated test script to check arithmetic mixin import."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import dependencies first
    from united_system.unit import Unit
    from united_system.dimension import Dimension
    from united_system.utils.units.unit_symbol import UnitSymbol
    from united_system.utils.scalars.mixins.real_united_scalar.protocol import RealUnitedScalarProtocol
    
    # Now try to import the arithmetic mixin
    from united_system.utils.scalars.mixins.real_united_scalar.arithmetic_mixin import ArithmeticMixin
    print("✅ ArithmeticMixin imported successfully!")
    
    # Test that we can access the class
    print(f"✅ ArithmeticMixin class: {ArithmeticMixin}")
    print(f"✅ Base classes: {ArithmeticMixin.__bases__}")
    
    print("✅ No circular import issues detected!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Other error: {e}")
    import traceback
    traceback.print_exc() 