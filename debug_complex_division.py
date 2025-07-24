#!/usr/bin/env python3

from united_system.unit import Unit

def debug_complex_division():
    print("=== DEBUGGING COMPLEX DIVISION ===")
    
    # Test the failing case
    test_string = "kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)"
    print(f"Test string: {test_string}")
    
    # Parse the unit
    test_unit = Unit(test_string)
    print(f"Parsed unit: {test_unit}")
    print(f"Unit elements: {test_unit.unit_elements}")
    
    # Print each element
    for subscript, elements in test_unit.unit_elements.items():
        print(f"  Subscript '{subscript}':")
        for element in elements:
            print(f"    {element.unit_symbol.value.symbols[0]} (exponent: {element.exponent})")
    
    print("\n--- Testing reduction ---")
    reduced = Unit.reduce_unit(test_unit)
    print(f"Reduced unit: {reduced}")
    print(f"Reduced unit elements: {reduced.unit_elements}")
    
    # Print each element of reduced unit
    for subscript, elements in reduced.unit_elements.items():
        print(f"  Subscript '{subscript}':")
        for element in elements:
            print(f"    {element.unit_symbol.value.symbols[0]} (exponent: {element.exponent})")

if __name__ == "__main__":
    debug_complex_division() 