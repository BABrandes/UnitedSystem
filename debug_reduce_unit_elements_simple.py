#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES, QuantityTag
from united_system.utils.units.proper_exponents import ProperExponents

def debug_simple_cases():
    print("=== DEBUGGING SIMPLE CASES ===")
    
    # Test the most basic cases first
    simple_cases = [
        ("A*V", "W"),  # Should reduce to W
        ("kg*m/s^2", "N"),  # Should reduce to N
        ("kg*m^2/s^2", "J"),  # Should reduce to J
        ("kg*m/s^3", "W/m"),  # Should reduce to W/m
    ]
    
    for input_unit_str, expected_output in simple_cases:
        print(f"\n--- {input_unit_str} -> {expected_output} ---")
        
        try:
            unit = Unit(input_unit_str)
            reduced = Unit.reduce_unit(unit)
            print(f"  Original: {unit}")
            print(f"  Reduced: {reduced}")
            print(f"  Expected: {expected_output}")
            print(f"  Match: {str(reduced) == expected_output}")
            
            if str(reduced) != expected_output:
                print(f"  ❌ FAILED")
            else:
                print(f"  ✅ PASSED")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

def debug_problematic_cases():
    print("\n=== DEBUGGING PROBLEMATIC CASES ===")
    
    # Test the cases that are producing unexpected results
    problematic_cases = [
        ("kg*m^2/s^2*mol", "J*mol"),  # Getting mol*C^2/F instead
        ("kg*m^2/s^2*A*V", "J*W"),  # Getting kg^6*A^13/Pa^4/C^13 instead
        ("kg*m/s^2/m^2", "Pa"),  # Getting J/m^3 instead
    ]
    
    for input_unit_str, expected_output in problematic_cases:
        print(f"\n--- {input_unit_str} -> {expected_output} ---")
        
        try:
            unit = Unit(input_unit_str)
            reduced = Unit.reduce_unit(unit)
            print(f"  Original: {unit}")
            print(f"  Reduced: {reduced}")
            print(f"  Expected: {expected_output}")
            print(f"  Match: {str(reduced) == expected_output}")
            
            if str(reduced) != expected_output:
                print(f"  ❌ FAILED: Got {reduced}")
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

def debug_algorithm_logic():
    print("\n=== DEBUGGING ALGORITHM LOGIC ===")
    
    # Test a simple case to understand what's happening
    test_unit = Unit("kg*m^2/s^2*mol")
    elements = test_unit.unit_elements.get("", ())
    
    print(f"Test unit: {test_unit}")
    print(f"Elements: {len(elements)} elements")
    
    # Calculate proper exponents
    proper_exponents = ProperExponents.proper_exponents_of_unit_elements(elements)
    print(f"Proper exponents: {proper_exponents}")
    
    # Check what quantities are available for reduction
    print(f"\nAvailable quantities for reduction:")
    for quantity in CONSIDER_FOR_REDUCTION_QUANTITIES:
        if quantity.name in ["ENERGY", "FORCE", "POWER", "PRESSURE", "CHARGE", "CAPACITANCE"]:
            print(f"  {quantity.name}: {quantity.value[0]} -> {quantity.dimension}")
            print(f"    Proper exponents: {quantity.value[1].proper_exponents}")

def debug_scoring():
    print("\n=== DEBUGGING SCORING SYSTEM ===")
    
    # Test the scoring function with some examples
    def score_candidate(candidate):
        """Test the scoring function"""
        score = len(candidate) * 2
        for named_quantity, exponent in candidate:
            score += abs(exponent)
            if exponent < 0:
                score += 1
            if QuantityTag.BASE_QUANTITY in named_quantity.tags:
                score += 0
            if QuantityTag.DERIVED_QUANTITY in named_quantity.tags:
                score += 1
        return score
    
    # Find ENERGY and FORCE quantities
    energy_quantity = None
    force_quantity = None
    for quantity in CONSIDER_FOR_REDUCTION_QUANTITIES:
        if quantity.name == "ENERGY":
            energy_quantity = quantity
        elif quantity.name == "FORCE":
            force_quantity = quantity
    
    if energy_quantity and force_quantity:
        test_candidates = [
            [(energy_quantity, 1.0)],  # Single ENERGY
            [(force_quantity, 1.0)],   # Single FORCE
            [(energy_quantity, 1.0), (force_quantity, 1.0)],  # ENERGY + FORCE
        ]
        
        for candidate in test_candidates:
            score = score_candidate(candidate)
            print(f"Candidate {[q.name for q, _ in candidate]}: score = {score}")

if __name__ == "__main__":
    debug_simple_cases()
    debug_problematic_cases()
    debug_algorithm_logic()
    debug_scoring() 