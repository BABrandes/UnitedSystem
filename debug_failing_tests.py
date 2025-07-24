#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.utils.units.old_reduce_unit_elements import reduce_unit_elements as old_reduce_unit_elements

def debug_failing_tests():
    print("=== DEBUGGING ORIGINAL VERSION FAILING TESTS ===")
    
    # Test cases from the failing unit tests
    test_cases = [
        # Group 1: Unit Reduction Not Working
        ("kg*m/s^3", "W/m"),  # Should reduce to W/m
        ("kg*m^2/s^2*mol", "J*mol"),  # Should reduce to J*mol  
        ("kg*m^2/s^2*A*V", "J*W"),  # Should reduce to J*W
        ("kg*m/s^2/m^2", "Pa"),  # Should reduce to Pa
        ("kg*m^2/s^2/K", "J/K"),  # Should reduce to J/K
        ("A*s", "C"),  # Should reduce to C
        ("kg*m^2/s^3", "W"),  # Should reduce to W
        ("N*m", "J"),  # Should reduce to J
        
        # Additional test cases
        ("A*V", "W"),  # Basic power reduction
        ("kg*m/s^2", "N"),  # Basic force reduction
        ("kg*m^2/s^2", "J"),  # Basic energy reduction
        ("A^2*V^2", "W^2"),  # Power squared
        ("kg^2*m^2/s^4", "N^2"),  # Force squared
    ]
    
    print(f"Testing {len(test_cases)} cases...\n")
    
    passed = 0
    failed = []
    
    for i, (input_unit_str, expected_output) in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {input_unit_str} -> {expected_output} ---")
        
        try:
            # Create the unit
            unit = Unit(input_unit_str)
            elements = unit.unit_elements.get("", ())
            
            # Test old version
            old_result = old_reduce_unit_elements(elements)
            
            # Create unit from result for comparison
            old_unit = Unit._construct({"": tuple(old_result)}, [])
            
            print(f"  Input: {input_unit_str}")
            print(f"  Result: {old_unit}")
            print(f"  Expected: {expected_output}")
            
            old_match = str(old_unit) == expected_output
            
            print(f"  Match: {old_match}")
            
            if old_match:
                passed += 1
                print(f"  ✅ PASSED")
            else:
                failed.append((i+1, input_unit_str, str(old_unit), expected_output))
                print(f"  ❌ FAILED")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed.append((i+1, input_unit_str, f"ERROR: {e}", expected_output))
        
        print()
    
    print(f"=== SUMMARY ===")
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {len(failed)}/{len(test_cases)}")
    
    if failed:
        print(f"\n=== FAILED TESTS ===")
        for test_num, input_unit, result, expected in failed:
            print(f"Test {test_num}: {input_unit}")
            print(f"  Got: {result}")
            print(f"  Expected: {expected}")
            print()

if __name__ == "__main__":
    debug_failing_tests() 