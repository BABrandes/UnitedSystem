#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES, QuantityTag
from united_system.utils.units.proper_exponents import ProperExponents
from united_system.utils.units.old_reduce_unit_elements import reduce_unit_elements as old_reduce_unit_elements

def debug_w_per_m():
    print("=== DEBUGGING kg*m/s^3 -> W/m ===")
    
    # Test the specific case
    test_unit = Unit("kg*m/s^3")
    print(f"Original unit: {test_unit}")
    print(f"Unit elements: {test_unit.unit_elements}")
    print(f"Dimension: {test_unit.dimension}")
    
    # Check what W and m are
    w = NamedQuantity.POWER
    m = NamedQuantity.LENGTH
    print(f"\nW (POWER) dimension: {w.dimension}")
    print(f"m (LENGTH) dimension: {m.dimension}")
    
    # Calculate what W/m should be
    w_per_m = w.dimension / m.dimension
    print(f"W/m dimension: {w_per_m}")
    print(f"Original unit dimension: {test_unit.dimension}")
    print(f"Dimensions match: {w_per_m == test_unit.dimension}")
    
    # Try to manually construct W/m
    w_unit = Unit("W")
    m_unit = Unit("m")
    w_per_m_unit = w_unit / m_unit
    print(f"\nManually constructed W/m: {w_per_m_unit}")
    print(f"W/m unit dimension: {w_per_m_unit.dimension}")
    
    # Now test the reduction
    print(f"\n=== TESTING REDUCTION ===")
    # Extract the actual unit elements from the dictionary
    unit_elements = test_unit.unit_elements[""]
    print(f"Unit elements to reduce: {unit_elements}")
    
    # Debug: Check each unit element's proper exponents
    print("\nDEBUG: Individual unit element proper exponents:")
    for i, element in enumerate(unit_elements):
        print(f"  Element {i}: {element} -> proper_exponents: {element.unit_symbol.named_quantity.value[1].proper_exponents}")
    
    result = old_reduce_unit_elements(unit_elements)
    print(f"Raw reduction result: {result}")
    print(f"Result type: {type(result)}")
    
    # The result is a dictionary, so let's extract the elements
    if isinstance(result, dict) and "" in result:
        elements = result[""]
        print(f"Elements: {elements}")
        print(f"Elements type: {type(elements)}")
        print(f"Number of elements: {len(elements)}")
        
        # Create a unit from the result
        result_unit = Unit._construct(result, [])
        print(f"Reduction result: {result_unit}")
        print(f"Expected: W/m")
        print(f"Match: {result_unit == w_per_m_unit}")
        
        # Check if the reduction actually changed anything
        print(f"Original: {test_unit}")
        print(f"Reduced: {result_unit}")
        print(f"Same as original: {test_unit == result_unit}")
    else:
        print("Unexpected result format")
    
    # Check if W/m is in the candidates
    print(f"\n=== CHECKING CANDIDATES ===")
    # We need to see what candidates are being generated
    # Let's add some debug output to the reduction function
    
if __name__ == "__main__":
    debug_w_per_m() 