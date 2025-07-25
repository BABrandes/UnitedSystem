#!/usr/bin/env python3
"""Direct test script to check arithmetic mixin import without main package."""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Import the protocol first
    from united_system.utils.scalars.mixins.real_united_scalar.protocol import RealUnitedScalarProtocol
    print("✅ Protocol imported successfully!")
    
    # Now try to import the arithmetic mixin directly
    import united_system.utils.scalars.mixins.real_united_scalar.arithmetic_mixin as arithmetic_module
    print("✅ Arithmetic module imported successfully!")
    
    # Get the ArithmeticMixin class
    ArithmeticMixin = arithmetic_module.ArithmeticMixin
    print("✅ ArithmeticMixin class accessed successfully!")
    
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