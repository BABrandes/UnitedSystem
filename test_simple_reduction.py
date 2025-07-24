#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit

def test_simple_reductions():
    """Test simple unit reductions that should work quickly."""
    print("=== TESTING SIMPLE UNIT REDUCTIONS ===")
    
    # Basic SI derived units - these should be fast and simple
    simple_cases = [
        ("A*V", "W"),  # Ampere * Volt = Watt
        ("kg*m/s^2", "N"),  # Force
        ("kg*m^2/s^2", "J"),  # Energy
        ("kg*m/s^3", "W/m"),  # Power per length
        ("kg*m/s^2/m^2", "Pa"),  # Pressure
        ("A*s", "C"),  # Ampere * second = Coulomb
        ("kg*m^2/s^3", "W"),  # Power
        ("N*m", "J"),  # Newton * meter = Joule
    ]
    
    passed = 0
    failed = 0
    
    for input_unit_str, expected_output in simple_cases:
        print(f"\n--- Testing: {input_unit_str} -> {expected_output} ---")
        
        try:
            unit = Unit(input_unit_str)
            reduced = Unit.reduce_unit(unit)
            result = str(reduced)
            
            print(f"  Original: {unit}")
            print(f"  Reduced: {reduced}")
            print(f"  Expected: {expected_output}")
            
            if result == expected_output:
                print(f"  ✅ PASSED")
                passed += 1
            else:
                print(f"  ❌ FAILED: Got {result}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{len(simple_cases)}")
    print(f"Failed: {failed}/{len(simple_cases)}")
    
    return failed == 0

def test_dimensionless_units():
    """Test dimensionless units that should reduce to empty string."""
    print("\n=== TESTING DIMENSIONLESS UNITS ===")
    
    dimensionless_cases = [
        ("", ""),  # Empty unit
        ("1", ""),  # Dimensionless
        ("m/m", ""),  # Length per length
        ("kg/kg", ""),  # Mass per mass
        ("s/s", ""),  # Time per time
    ]
    
    passed = 0
    failed = 0
    
    for input_unit_str, expected_output in dimensionless_cases:
        print(f"\n--- Testing: {input_unit_str} -> {expected_output} ---")
        
        try:
            unit = Unit(input_unit_str)
            reduced = Unit.reduce_unit(unit)
            result = str(reduced)
            
            print(f"  Original: {unit}")
            print(f"  Reduced: {reduced}")
            print(f"  Expected: {expected_output}")
            
            if result == expected_output:
                print(f"  ✅ PASSED")
                passed += 1
            else:
                print(f"  ❌ FAILED: Got {result}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{len(dimensionless_cases)}")
    print(f"Failed: {failed}/{len(dimensionless_cases)}")
    
    return failed == 0

def test_no_reduction_cases():
    """Test units that should not reduce further."""
    print("\n=== TESTING NO REDUCTION CASES ===")
    
    no_reduction_cases = [
        ("m", "m"),  # Base unit
        ("kg", "kg"),  # Base unit
        ("s", "s"),  # Base unit
        ("m/s", "m/s"),  # Velocity (no simpler form)
        ("kg/m^3", "kg/m^3"),  # Density (no simpler form)
    ]
    
    passed = 0
    failed = 0
    
    for input_unit_str, expected_output in no_reduction_cases:
        print(f"\n--- Testing: {input_unit_str} -> {expected_output} ---")
        
        try:
            unit = Unit(input_unit_str)
            reduced = Unit.reduce_unit(unit)
            result = str(reduced)
            
            print(f"  Original: {unit}")
            print(f"  Reduced: {reduced}")
            print(f"  Expected: {expected_output}")
            
            if result == expected_output:
                print(f"  ✅ PASSED")
                passed += 1
            else:
                print(f"  ❌ FAILED: Got {result}")
                failed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            failed += 1
    
    print(f"\n=== SUMMARY ===")
    print(f"Passed: {passed}/{len(no_reduction_cases)}")
    print(f"Failed: {failed}/{len(no_reduction_cases)}")
    
    return failed == 0

def test_performance():
    """Test that reduction is reasonably fast."""
    print("\n=== TESTING PERFORMANCE ===")
    
    import time
    
    # Test with a moderately complex unit that should reduce quickly
    test_unit = Unit("kg*m^2/s^2*A*V")
    
    print(f"Testing reduction of: {test_unit}")
    
    start_time = time.time()
    reduced = Unit.reduce_unit(test_unit)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"Reduced to: {reduced}")
    print(f"Time taken: {duration:.4f}s")
    
    # Should complete in reasonable time (less than 0.1 seconds)
    if duration < 0.1:
        print("✅ Performance test PASSED")
        return True
    else:
        print(f"❌ Performance test FAILED: Too slow ({duration:.4f}s)")
        return False

if __name__ == "__main__":
    print("🧪 RUNNING SIMPLE UNIT REDUCTION TESTS")
    print("=" * 50)
    
    # Run all tests
    test1 = test_simple_reductions()
    test2 = test_dimensionless_units()
    test3 = test_no_reduction_cases()
    test4 = test_performance()
    
    print("\n" + "=" * 50)
    print("🎯 FINAL RESULTS")
    print("=" * 50)
    
    if all([test1, test2, test3, test4]):
        print("🎉 ALL TESTS PASSED!")
        print("Your unit reduction fix is working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        print("There are still issues to address.")
    
    print("=" * 50) 