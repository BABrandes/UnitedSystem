#!/usr/bin/env python3

from united_system.unit import Unit
from united_system.utils.units.utils import seperate_string

def debug_formatting():
    print("=== DEBUGGING UNIT FORMATTING ===")
    
    # Test the failing case
    test_string = "kg_thermal*m/s^2/m^2"
    print(f"Original unit string: {test_string}")
    
    # Test seperate_string first
    print("\n--- Testing seperate_string ---")
    parts = seperate_string(test_string, "nominator")
    print(f"seperate_string result: {parts}")
    
    # Test the failing case
    test_unit = Unit(test_string)
    print(f"\nParsed unit: {test_unit}")
    print(f"Unit elements: {test_unit.unit_elements}")
    
    # Print each element
    for subscript, elements in test_unit.unit_elements.items():
        print(f"  Subscript '{subscript}':")
        for element in elements:
            print(f"    {element.unit_symbol.value.symbols[0]} (exponent: {element.exponent})")
    
    print("\n--- Testing format_string method ---")
    formatted = test_unit.format_string(as_fraction=True)
    print(f"Formatted result: {formatted}")
    
    # Test the expected case
    expected_unit = Unit("kg_thermal/m/s^2")
    print(f"\nExpected unit: {expected_unit}")
    print(f"Expected formatted: {expected_unit.format_string(as_fraction=True)}")
    
    # Test individual elements
    print("\n--- Testing individual element formatting ---")
    for subscript, elements in test_unit.unit_elements.items():
        for element in elements:
            part_str, position = element.format_string(as_fraction=True)
            print(f"  {element.unit_symbol.value.symbols[0]} (exponent: {element.exponent}) -> '{part_str}' ({position})")
    
    # Test the new failing case
    print("\n=== TESTING NEW FAILING CASE ===")
    failing_case = "1/s_geo*1/s_thermal"
    print(f"Failing case: {failing_case}")
    
    parts_failing = seperate_string(failing_case, "nominator")
    print(f"seperate_string result: {parts_failing}")
    
    failing_unit = Unit(failing_case)
    print(f"Parsed unit: {failing_unit}")
    print(f"Formatted result: {failing_unit.format_string(as_fraction=True)}")

if __name__ == "__main__":
    debug_formatting() 