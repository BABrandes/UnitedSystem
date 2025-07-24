#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES, SI_BASE_QUANTITIES, QuantityTag
from united_system.utils.units.proper_exponents import ProperExponents
from united_system.utils.units.unit_element import UnitElement

def debug_reduce_unit_elements():
    print("=== DEBUGGING COMPLETELY REWRITTEN reduce_unit_elements ===")
    
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
    
    for i, (input_unit_str, expected_output) in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {input_unit_str} -> {expected_output} ---")
        
        try:
            # Create the unit
            unit = Unit(input_unit_str)
            print(f"  Original unit: {unit}")
            print(f"  Unit elements: {unit.unit_elements}")
            
            # Get the unit elements for the main subscript (usually "")
            elements = unit.unit_elements.get("", ())
            print(f"  Main elements: {elements}")
            
            # Calculate proper exponents
            proper_exponents = ProperExponents.proper_exponents_of_unit_elements(elements)
            print(f"  Proper exponents: {proper_exponents}")
            
            # Test the reduction
            reduced = Unit.reduce_unit(unit)
            print(f"  Reduced unit: {reduced}")
            print(f"  Expected: {expected_output}")
            print(f"  Match: {str(reduced) == expected_output}")
            
            if str(reduced) != expected_output:
                print(f"  ❌ FAILED: Expected {expected_output}, got {reduced}")
            else:
                print(f"  ✅ PASSED")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()

def debug_algorithm_details():
    print("=== DEBUGGING ALGORITHM DETAILS ===")
    
    # Test a simple case to understand the algorithm
    test_unit = Unit("A*V")
    elements = test_unit.unit_elements.get("", ())
    
    print(f"Test unit: {test_unit}")
    print(f"Elements: {elements}")
    
    # Calculate proper exponents
    proper_exponents = ProperExponents.proper_exponents_of_unit_elements(elements)
    print(f"Proper exponents: {proper_exponents}")
    
    # Check what's in CONSIDER_FOR_REDUCTION_QUANTITIES
    print(f"\nCONSIDER_FOR_REDUCTION_QUANTITIES:")
    for quantity in CONSIDER_FOR_REDUCTION_QUANTITIES:
        print(f"  {quantity.name}: {quantity.value[0]} -> {quantity.dimension}")
        print(f"    Proper exponents: {quantity.value[1].proper_exponents}")
        print(f"    Unit element: {quantity.unit_element}")
        print(f"    Tags: {quantity.tags}")
    
    # Test the subtract function logic
    print(f"\n--- Testing subtract function ---")
    
    def subtract(index, value_1, value_2):
        """Test the subtract function from the algorithm"""
        factor = value_1[index] / value_2[index]
        result = tuple(value_1[i] - value_2[i] * factor for i in range(8))
        return factor, result
    
    def is_zero(value):
        """Test the is_zero function from the algorithm"""
        return all(abs(v) < 1e-10 for v in value)
    
    # Test with POWER (W) which should match A*V
    power_quantity = None
    for quantity in CONSIDER_FOR_REDUCTION_QUANTITIES:
        if quantity.name == "POWER":
            power_quantity = quantity
            break
    
    if power_quantity:
        print(f"Testing with POWER: {power_quantity.value[1].proper_exponents}")
        
        # Test different indices
        for index in range(8):
            if power_quantity.value[1].proper_exponents[index] != 0:
                factor, new_exponents = subtract(index, proper_exponents, power_quantity.value[1].proper_exponents)
                print(f"  Index {index}: factor={factor}, new_exponents={new_exponents}")
                print(f"    Is zero: {is_zero(new_exponents)}")

def debug_scoring_system():
    print("=== DEBUGGING SCORING SYSTEM ===")
    
    # Test the scoring function
    def score_named_quantities_candidates(named_quantities_candidate):
        """Test the scoring function from the algorithm"""
        score = len(named_quantities_candidate) * 2
        for named_quantity, exponent in named_quantities_candidate:
            score += abs(exponent)
            if exponent < 0:
                score += 1
            if QuantityTag.BASE_QUANTITY in named_quantity.tags:
                score += 0
            if QuantityTag.DERIVED_QUANTITY in named_quantity.tags:
                score += 1
        return score
    
    # Test with some example candidates
    test_candidates = [
        [(NamedQuantity.POWER, 1.0)],  # Single POWER
        [(NamedQuantity.POWER, 2.0)],  # POWER squared
        [(NamedQuantity.FORCE, 1.0)],  # Single FORCE
        [(NamedQuantity.POWER, 1.0), (NamedQuantity.FORCE, 1.0)],  # POWER + FORCE
    ]
    
    for candidate in test_candidates:
        score = score_named_quantities_candidates(candidate)
        print(f"Candidate {candidate}: score = {score}")

def debug_edge_cases():
    print("=== DEBUGGING EDGE CASES ===")
    
    edge_cases = [
        ("", ""),  # Empty unit
        ("1", ""),  # Dimensionless
        ("m/m", ""),  # Canceling units
        ("kg*kg", "kg^2"),  # Same unit multiplied
        ("kg/kg", ""),  # Same unit divided
        ("m^2/m", "m"),  # Partial cancellation
        ("m^3/m^2", "m"),  # More partial cancellation
    ]
    
    for input_unit_str, expected_output in edge_cases:
        print(f"--- Edge Case: {input_unit_str} -> {expected_output} ---")
        
        try:
            unit = Unit(input_unit_str)
            reduced = Unit.reduce_unit(unit)
            print(f"  Original: {unit}")
            print(f"  Reduced: {reduced}")
            print(f"  Expected: {expected_output}")
            print(f"  Match: {str(reduced) == expected_output}")
            
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

def debug_performance():
    print("=== DEBUGGING PERFORMANCE ===")
    
    import time
    
    # Test with complex units
    complex_units = [
        "kg*m^2/s^3*A*V*N*m",
        "kg*m^2/s^2*A*V*kg*m/s^2/m^2",
        "kg*m^2/s^2*kg*m/s^2*A*V*kg*m^2/s^2/K",
    ]
    
    for unit_str in complex_units:
        print(f"--- Performance Test: {unit_str} ---")
        
        start_time = time.time()
        unit = Unit(unit_str)
        reduced = Unit.reduce_unit(unit)
        end_time = time.time()
        
        print(f"  Time: {end_time - start_time:.4f}s")
        print(f"  Original: {unit}")
        print(f"  Reduced: {reduced}")
        print(f"  Reduction ratio: {len(str(unit)) / len(str(reduced)):.2f}")

if __name__ == "__main__":
    debug_reduce_unit_elements()
    print("\n" + "="*80 + "\n")
    debug_algorithm_details()
    print("\n" + "="*80 + "\n")
    debug_scoring_system()
    print("\n" + "="*80 + "\n")
    debug_edge_cases()
    print("\n" + "="*80 + "\n")
    debug_performance() 