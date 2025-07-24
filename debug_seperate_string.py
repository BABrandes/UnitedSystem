#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.utils.units.utils import seperate_string
import re

def debug_process_normal_string(string: str, position: str):
    print(f"DEBUG: process_normal_string called with string='{string}', position='{position}'")
    
    if not string.strip():
        print("DEBUG: Empty string, returning []")
        return []
        
    # Split and retain separators
    parts: list[str] = re.split(r'([/*])', string)
    print(f"DEBUG: re.split result: {parts}")

    # Zip into (separator, part) tuples
    # The first separator is implicit, so handle the first part specially
    if parts[0] in ("*", "/"):
        # Starts with a separator
        seperators_and_parts: list[tuple[str, str]] = list(zip(parts[::2], parts[1::2]))
    else:
        # Starts with content
        seperators_and_parts: list[tuple[str, str]] = [("*", parts[0])] + list(zip(parts[1::2], parts[2::2]))
    
    print(f"DEBUG: seperators_and_parts before filtering: {seperators_and_parts}")

    if position == "denominator":
        for i in range(len(seperators_and_parts)):
            if seperators_and_parts[i][0] == "/":
                seperators_and_parts[i] = ("*", seperators_and_parts[i][1])
            else:
                seperators_and_parts[i] = ("/", seperators_and_parts[i][1])
    
    print(f"DEBUG: seperators_and_parts after processing: {seperators_and_parts}")
    return seperators_and_parts

def debug_seperate_string():
    print("=== DEBUGGING SEPERATE_STRING ===")
    
    # Test the failing case
    test_string = "kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)"
    print(f"Test string: {test_string}")
    
    # Test seperate_string
    print("\n--- seperate_string output ---")
    parts = seperate_string(test_string, "nominator")
    print(f"seperate_string result: {parts}")
    
    # Print each part in detail
    print("\n--- Detailed parts ---")
    for i, (separator, part) in enumerate(parts):
        print(f"Part {i}: separator='{separator}', part='{part}'")
    
    # Test a simpler case for comparison
    print("\n--- Simple case for comparison ---")
    simple_string = "kg_thermal*m_elec^2/s_geo^2"
    simple_parts = seperate_string(simple_string, "nominator")
    print(f"Simple string: {simple_string}")
    print(f"Simple seperate_string result: {simple_parts}")
    
    # Test with parentheses to see if operator reversal works
    print("\n--- Test with parentheses ---")
    paren_string = "kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)"
    paren_parts = seperate_string(paren_string, "nominator")
    print(f"Parentheses string: {paren_string}")
    print(f"Parentheses seperate_string result: {paren_parts}")
    
    # Test denominator position
    print("\n--- Test denominator position ---")
    denom_string = "kg_thermal*m_elec^2/s_geo^2"
    denom_parts = seperate_string(denom_string, "denominator")
    print(f"Denominator string: {denom_string}")
    print(f"Denominator seperate_string result: {denom_parts}")
    
    # Test very simple cases
    print("\n--- Very simple tests ---")
    test1 = "kg"
    result1 = seperate_string(test1, "nominator")
    print(f"'{test1}' -> {result1}")
    
    test2 = "kg*m"
    result2 = seperate_string(test2, "nominator")
    print(f"'{test2}' -> {result2}")
    
    test3 = "kg/m"
    result3 = seperate_string(test3, "nominator")
    print(f"'{test3}' -> {result3}")
    
    # Test the specific case that's causing the empty part
    print("\n--- Debug specific case ---")
    debug_string = "kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)"
    print(f"Debug string: '{debug_string}'")
    
    # Manually trace what should happen
    print("\n--- Manual trace ---")
    # First, the normal_string should be: "kg_thermal*m_elec^2/s_geo^2/"
    normal_string = "kg_thermal*m_elec^2/s_geo^2/"
    print(f"Normal string: '{normal_string}'")
    debug_process_normal_string(normal_string, "nominator")
    
    # Show what it SHOULD produce
    print("\n--- What it SHOULD produce ---")
    print("For: kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)")
    print("Expected: [('*', 'kg_thermal'), ('*', 'm_elec^2'), ('/', 's_geo^2'), ('/', 'kg_thermal'), ('/', 'm_elec^2'), ('*', 's_geo^2')]")
    print("This would mean:")
    print("  Nominator: kg_thermal*m_elec^2/s_geo^2")
    print("  Denominator: kg_thermal*m_elec^2/s_geo^2")
    print("  Result: 1 (dimensionless)")

if __name__ == "__main__":
    debug_seperate_string() 