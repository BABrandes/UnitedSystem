#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit
from united_system.named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES, QuantityTag
from united_system.utils.units.proper_exponents import ProperExponents
from united_system.utils.units.unit_element import UnitElement
from united_system.utils.units.chatgpt_reduce_unit_elements import reduce_unit_elements as chatgpt_reduce_unit_elements
from united_system.utils.units.old_reduce_unit_elements import reduce_unit_elements as old_reduce_unit_elements
import time

def test_chatgpt_version():
    print("=== TESTING CHATGPT VERSION OF reduce_unit_elements ===")
    
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
    
    chatgpt_passed = 0
    old_passed = 0
    
    for i, (input_unit_str, expected_output) in enumerate(test_cases):
        print(f"--- Test Case {i+1}: {input_unit_str} -> {expected_output} ---")
        
        try:
            # Create the unit
            unit = Unit(input_unit_str)
            elements = unit.unit_elements.get("", ())
            
            # Test ChatGPT version
            start_time = time.time()
            chatgpt_result = chatgpt_reduce_unit_elements(elements)
            chatgpt_time = time.time() - start_time
            
            # Test old version
            start_time = time.time()
            old_result = old_reduce_unit_elements(elements)
            old_time = time.time() - start_time
            
            # Create units from results for comparison
            chatgpt_unit = Unit._construct({"": tuple(chatgpt_result)}, [])
            old_unit = Unit._construct({"": tuple(old_result)}, [])
            
            print(f"  ChatGPT: {chatgpt_unit} (time: {chatgpt_time:.4f}s)")
            print(f"  Old:     {old_unit} (time: {old_time:.4f}s)")
            print(f"  Expected: {expected_output}")
            
            chatgpt_match = str(chatgpt_unit) == expected_output
            old_match = str(old_unit) == expected_output
            
            print(f"  ChatGPT match: {chatgpt_match}")
            print(f"  Old match: {old_match}")
            
            if chatgpt_match:
                chatgpt_passed += 1
            if old_match:
                old_passed += 1
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print(f"=== RESULTS ===")
    print(f"ChatGPT version: {chatgpt_passed}/{len(test_cases)} passed")
    print(f"Old version: {old_passed}/{len(test_cases)} passed")

def test_performance():
    print("\n=== PERFORMANCE TESTING ===")
    
    # Test with complex units
    complex_units = [
        "kg*m^2/s^3*A*V*N*m",
        "kg*m^2/s^2*A*V*kg*m/s^2/m^2",
        "kg*m^2/s^2*kg*m/s^2*A*V*kg*m^2/s^2/K",
    ]
    
    for unit_str in complex_units:
        print(f"\n--- Performance Test: {unit_str} ---")
        
        unit = Unit(unit_str)
        elements = unit.unit_elements.get("", ())
        
        # Test ChatGPT version
        start_time = time.time()
        chatgpt_result = chatgpt_reduce_unit_elements(elements)
        chatgpt_time = time.time() - start_time
        
        # Test old version
        start_time = time.time()
        old_result = old_reduce_unit_elements(elements)
        old_time = time.time() - start_time
        
        print(f"  ChatGPT time: {chatgpt_time:.4f}s")
        print(f"  Old time: {old_time:.4f}s")
        print(f"  Speedup: {old_time/chatgpt_time:.2f}x")
        
        # Show results
        chatgpt_unit = Unit._construct({"": tuple(chatgpt_result)}, [])
        old_unit = Unit._construct({"": tuple(old_result)}, [])
        print(f"  ChatGPT result: {chatgpt_unit}")
        print(f"  Old result: {old_unit}")

if __name__ == "__main__":
    test_chatgpt_version()
    test_performance() 