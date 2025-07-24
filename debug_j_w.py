#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES, QuantityTag
from united_system.utils.units.proper_exponents import ProperExponents
from united_system.utils.units.old_reduce_unit_elements import reduce_unit_elements as old_reduce_unit_elements

def debug_j_w():
    print("=== DEBUGGING kg*m^2/s^2*A*V -> J*W ===")
    
    # Test the specific case
    test_unit = Unit("kg*m^2/s^2*A*V")
    print(f"Original unit: {test_unit}")
    print(f"Unit elements: {test_unit.unit_elements}")
    print(f"Dimension: {test_unit.dimension}")
    
    # Check what J and W are
    j = NamedQuantity.ENERGY
    w = NamedQuantity.POWER
    print(f"\nJ (ENERGY) dimension: {j.value[1]}")
    print(f"W (POWER) dimension: {w.value[1]}")
    
    # Try to manually construct J*W
    j_unit = Unit("J")
    w_unit = Unit("W")
    j_w_unit = j_unit * w_unit
    print(f"\nManually constructed J*W: {j_w_unit}")
    print(f"J*W unit dimension: {j_w_unit.dimension}")
    print(f"Original unit dimension: {test_unit.dimension}")
    print(f"Dimensions match: {test_unit.dimension == j_w_unit.dimension}")
    
    # Extract the actual unit elements from the dictionary
    unit_elements = test_unit.unit_elements[""]
    print(f"\nUnit elements to reduce: {unit_elements}")
    
    # Debug: Check each unit element's proper exponents
    print("\nDEBUG: Individual unit element proper exponents:")
    for i, element in enumerate(unit_elements):
        print(f"  Element {i}: {element} -> proper_exponents: {element.unit_symbol.named_quantity.value[1].proper_exponents}")
    
    result = old_reduce_unit_elements(unit_elements)
    print(f"\nRaw reduction result: {result}")
    print(f"Result type: {type(result)}")
    
    # Try to create a unit from the result
    try:
        result_unit = Unit._construct({"": tuple(result)}, [])
        print(f"Reduction result: {result_unit}")
        print(f"Expected: J*W")
        print(f"Match: {result_unit == j_w_unit}")
    except Exception as e:
        print(f"Error creating unit: {e}")
        print("Unexpected result format")

if __name__ == "__main__":
    debug_j_w() 