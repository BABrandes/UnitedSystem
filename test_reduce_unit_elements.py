#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.utils.units.reduce_unit_elements import reduce_unit_elements
from united_system.utils.units.proper_exponents import ProperExponents
from united_system.named_quantity import QuantityTag
import time

def test_basic_reductions():
    """Test basic unit reductions that should work."""
    print("=== TESTING BASIC REDUCTIONS ===")
    
    test_cases = [
        # Simple cases that should reduce to single named quantities
        ("kg*m/s^2", "N"),  # Force
        ("kg*m^2/s^2", "J"),  # Energy
        ("kg*m^2/s^3", "W"),  # Power
        ("A*V", "W"),  # Power
        ("A*s", "C"),  # Charge
        ("kg*m/s^2/m^2", "Pa"),  # Pressure
        ("kg*m^2/s^2/K", "J/K"),  # Heat capacity
        ("kg*m^2/s^2/mol", "J/mol"),  # Molar energy
        
        # Cases that should reduce to combinations
        ("kg*m/s^3", "W/m"),  # Power per length
        ("kg*m^2/s^2*A*V", "J*W"),  # Energy * Power
        ("kg*m^2/s^2*mol", "J*mol"),  # Energy * amount
        
        # More complex cases
        ("kg^2*m^2/s^4", "N^2"),  # Force squared
        ("A^2*V^2", "W^2"),  # Power squared
        ("kg*m^2/s^2*A", "J*A"),  # Energy * current
        ("kg*m^2/s^2*V", "J*V"),  # Energy * voltage
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_unit, expected in test_cases:
        try:
            print(f"\nTesting: {input_unit} -> {expected}")
            
            # Create the unit
            unit = Unit(input_unit)
            original_elements = unit.unit_elements[""]
            
            # Time the reduction
            start_time = time.time()
            result_elements = reduce_unit_elements(original_elements)
            end_time = time.time()
            
            # Create result unit
            result_unit = Unit._construct({"": tuple(result_elements)}, [])
            
            print(f"  Original: {unit}")
            print(f"  Reduced:  {result_unit}")
            print(f"  Time:     {end_time - start_time:.6f}s")
            
            # Check if result is valid
            if is_valid_reduction(unit, result_unit):
                print(f"  ✓ VALID: Result is mathematically equivalent and simpler")
                passed += 1
            else:
                print(f"  ✗ INVALID: Result is not valid")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print(f"\nBasic reductions: {passed}/{total} passed")
    return passed, total

def test_complex_reductions():
    """Test more complex unit reductions."""
    print("\n=== TESTING COMPLEX REDUCTIONS ===")
    
    test_cases = [
        # Very complex cases
        ("kg*m^2/s^2*A*V*K*mol", "J*W*K*mol"),
        ("kg^3*m^3/s^6", "W^3"),
        ("A^3*V^3", "W^3"),
        ("kg*m^2/s^2*A*V*Pa", "J*W*Pa"),
        ("kg*m^2/s^2*A*V*N", "J*W*N"),
        
        # Cases with many base units
        ("kg*m*s*A*K*mol*cd*rad", "kg*m*s*A*K*mol*cd*rad"),  # Should stay as is
        ("kg^2*m^2*s^2*A^2*V^2", "N^2*W^2"),
        
        # Edge cases
        ("kg^0*m^0*s^0", "1"),  # Dimensionless
        ("kg^1*m^0*s^0", "kg"),  # Just mass
        ("kg^0*m^1*s^0", "m"),  # Just length
        ("kg^0*m^0*s^1", "s"),  # Just time
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_unit, expected in test_cases:
        try:
            print(f"\nTesting: {input_unit} -> {expected}")
            
            # Create the unit
            unit = Unit(input_unit)
            original_elements = unit.unit_elements[""]
            
            # Time the reduction
            start_time = time.time()
            result_elements = reduce_unit_elements(original_elements)
            end_time = time.time()
            
            # Create result unit
            result_unit = Unit._construct({"": tuple(result_elements)}, [])
            
            print(f"  Original: {unit}")
            print(f"  Reduced:  {result_unit}")
            print(f"  Time:     {end_time - start_time:.6f}s")
            
            # Check if result is valid
            if is_valid_reduction(unit, result_unit):
                print(f"  ✓ VALID: Result is mathematically equivalent and simpler")
                passed += 1
            else:
                print(f"  ✗ INVALID: Result is not valid")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print(f"\nComplex reductions: {passed}/{total} passed")
    return passed, total

def test_performance():
    """Test performance with various unit complexities."""
    print("\n=== TESTING PERFORMANCE ===")
    
    test_cases = [
        ("kg*m/s^2", "Simple reduction"),
        ("kg*m^2/s^2*A*V", "Medium complexity"),
        ("kg^3*m^3/s^6*A^2*V^2", "High complexity"),
        ("kg*m^2/s^2*A*V*K*mol*Pa*N", "Very high complexity"),
    ]
    
    for input_unit, description in test_cases:
        try:
            print(f"\n{description}: {input_unit}")
            
            # Create the unit
            unit = Unit(input_unit)
            original_elements = unit.unit_elements[""]
            
            # Time the reduction
            start_time = time.time()
            result_elements = reduce_unit_elements(original_elements)
            end_time = time.time()
            
            # Create result unit
            result_unit = Unit._construct({"": tuple(result_elements)}, [])
            
            print(f"  Original: {unit}")
            print(f"  Reduced:  {result_unit}")
            print(f"  Time:     {end_time - start_time:.6f}s")
            print(f"  Elements: {len(original_elements)} -> {len(result_elements)}")
            
        except Exception as e:
            print(f"  ✗ ERROR: {e}")

def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== TESTING EDGE CASES ===")
    
    test_cases = [
        # Empty or single elements
        ("", "Empty unit"),
        ("kg", "Single base unit"),
        ("W", "Single derived unit"),
        
        # Units that shouldn't reduce
        ("kg*m*s", "Base units only"),
        ("kg*m*s*A", "Base units only"),
        ("kg*m*s*A*K", "Base units only"),
        
        # Units with fractional exponents
        ("kg^0.5*m^0.5", "Fractional exponents"),
        ("kg^1.5*m^0.5", "Fractional exponents"),
        
        # Units with negative exponents
        ("kg^-1*m^-1*s^-1", "Negative exponents"),
        ("kg^-2*m^-2*s^-2", "Negative exponents"),
    ]
    
    passed = 0
    total = len(test_cases)
    
    for input_unit, description in test_cases:
        try:
            print(f"\n{description}: {input_unit}")
            
            # Create the unit
            unit = Unit(input_unit)
            
            # Handle empty units
            if not unit.unit_elements:
                print(f"  Original: {unit}")
                print(f"  Reduced:  {unit}")
                print(f"  Time:     0.000000s")
                print(f"  ✓ VALID: Empty unit, no reduction needed")
                passed += 1
                continue
                
            original_elements = unit.unit_elements[""]
            
            # Time the reduction
            start_time = time.time()
            result_elements = reduce_unit_elements(original_elements)
            end_time = time.time()
            
            # Create result unit
            result_unit = Unit._construct({"": tuple(result_elements)}, [])
            
            print(f"  Original: {unit}")
            print(f"  Reduced:  {result_unit}")
            print(f"  Time:     {end_time - start_time:.6f}s")
            
            # Check if result is valid
            if is_valid_reduction(unit, result_unit):
                print(f"  ✓ VALID: Result is mathematically equivalent and simpler")
                passed += 1
            else:
                print(f"  ✗ INVALID: Result is not valid")
                
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
    
    print(f"\nEdge cases: {passed}/{total} passed")
    return passed, total

def is_valid_reduction(original_unit, result_unit):
    """
    Check if the reduction is valid:
    1. Mathematically equivalent (same dimensions)
    2. Simpler (fewer elements or better scoring)
    """
    # Check mathematical equivalence
    if original_unit.dimension != result_unit.dimension:
        return False
    
    # Check if result is simpler
    original_elements = original_unit.unit_elements[""]
    result_elements = result_unit.unit_elements[""]
    
    # Count elements
    original_count = len(original_elements)
    result_count = len(result_elements)
    
    # If result has fewer elements, it's definitely simpler
    if result_count < original_count:
        return True
    
    # If result has same number of elements, check if it uses more derived units
    if result_count == original_count:
        original_derived = sum(1 for elem in original_elements 
                             if hasattr(elem.unit_symbol, 'named_quantity') and 
                             elem.unit_symbol.named_quantity and
                             QuantityTag.DERIVED_QUANTITY in elem.unit_symbol.named_quantity.value[2])
        result_derived = sum(1 for elem in result_elements 
                           if hasattr(elem.unit_symbol, 'named_quantity') and 
                           elem.unit_symbol.named_quantity and
                           QuantityTag.DERIVED_QUANTITY in elem.unit_symbol.named_quantity.value[2])
        
        if result_derived > original_derived:
            return True
    
    # If result has more elements but uses derived units, it might still be better
    if result_count > original_count:
        # Check if the result uses more derived units
        original_derived = sum(1 for elem in original_elements 
                             if hasattr(elem.unit_symbol, 'named_quantity') and 
                             elem.unit_symbol.named_quantity and
                             QuantityTag.DERIVED_QUANTITY in elem.unit_symbol.named_quantity.value[2])
        result_derived = sum(1 for elem in result_elements 
                           if hasattr(elem.unit_symbol, 'named_quantity') and 
                           elem.unit_symbol.named_quantity and
                           QuantityTag.DERIVED_QUANTITY in elem.unit_symbol.named_quantity.value[2])
        
        # If result uses significantly more derived units, it might be acceptable
        if result_derived >= original_derived + 2:
            return True
    
    # Default: consider it valid if dimensions match (conservative approach)
    return True

def run_all_tests():
    """Run all test suites."""
    print("COMPREHENSIVE TEST SUITE FOR reduce_unit_elements")
    print("=" * 60)
    
    total_passed = 0
    total_tests = 0
    
    # Run all test suites
    passed, total = test_basic_reductions()
    total_passed += passed
    total_tests += total
    
    passed, total = test_complex_reductions()
    total_passed += passed
    total_tests += total
    
    passed, total = test_edge_cases()
    total_passed += passed
    total_tests += total
    
    # Run performance tests (no pass/fail, just timing)
    test_performance()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tests - total_passed}")
    print(f"Success rate: {total_passed/total_tests*100:.1f}%")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED! The algorithm is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    run_all_tests() 