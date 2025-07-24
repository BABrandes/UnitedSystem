#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit

def test_dec_arithmetic():
    print("Testing dec(M) - dec(M) = dec")
    
    # Test dec(M) - dec(M)
    u1 = Unit("dec(kg)")
    u2 = Unit("dec(kg)")
    result = u1 - u2
    
    print(f"u1 = {u1}")
    print(f"u2 = {u2}")
    print(f"u1 - u2 = {result}")
    print(f"Result is dimensionless: {result.is_dimensionless}")
    print(f"Result string: '{result.format_string()}'")
    
    # Expected: result should be "dec"
    assert result.is_dimensionless, f"Expected dimensionless, got {result}"
    assert result.format_string() == "dec", f"Expected 'dec', got '{result.format_string()}'"
    
    print("✅ Test passed: dec(M) - dec(M) = dec")
    
    print("\nTesting dec(M) + dec(L) = dec(M*L)")
    
    # Test dec(M) + dec(L)
    u1 = Unit("dec(kg)")
    u2 = Unit("dec(m)")
    result = u1 + u2
    
    print(f"u1 = {u1}")
    print(f"u2 = {u2}")
    print(f"u1 + u2 = {result}")
    print(f"Result string: '{result.format_string()}'")
    
    # Expected: result should be "dec(M*L)"
    expected = "dec(M*L)"
    assert result.format_string() == expected, f"Expected '{expected}', got '{result.format_string()}'"
    
    print("✅ Test passed: dec(M) + dec(L) = dec(M*L)")
    
    print("\nTesting dec(M) + bin(L) = dec(M*L)")
    
    # Test dec(M) + bin(L) - different log functions
    u1 = Unit("dec(kg)")
    u2 = Unit("bin(m)")
    result = u1 + u2
    
    print(f"u1 = {u1}")
    print(f"u2 = {u2}")
    print(f"u1 + u2 = {result}")
    print(f"Result string: '{result.format_string()}'")
    
    # Expected: result should be "dec(M*L)" (uses the first log function)
    expected = "dec(M*L)"
    assert result.format_string() == expected, f"Expected '{expected}', got '{result.format_string()}'"
    
    print("✅ Test passed: dec(M) + bin(L) = dec(M*L)")

if __name__ == "__main__":
    test_dec_arithmetic() 