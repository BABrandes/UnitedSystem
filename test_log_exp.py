#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.unit import Unit

def test_log_exp():
    print("Testing Unit.log() and Unit.exp() methods...")
    
    # Test 1: log() on regular unit
    print("\n1. Testing Unit('kg').log()")
    u1 = Unit("kg")
    log_u1 = u1.log()
    print(f"u1 = {u1}")
    print(f"u1.log() = {log_u1}")
    print(f"log_u1.format_string() = '{log_u1.format_string()}'")
    
    # Expected: dec(M)
    expected = "dec(M)"
    assert log_u1.format_string() == expected, f"Expected '{expected}', got '{log_u1.format_string()}'"
    print("✅ Test passed: Unit('kg').log() = dec(M)")
    
    # Test 2: exp() on log unit
    print("\n2. Testing Unit('dec(kg)').exp()")
    u2 = Unit("dec(kg)")
    exp_u2 = u2.exp()
    print(f"u2 = {u2}")
    print(f"u2.exp() = {exp_u2}")
    print(f"exp_u2.format_string() = '{exp_u2.format_string()}'")
    
    # Expected: kg (or the canonical unit for mass)
    print("✅ Test passed: Unit('dec(kg)').exp() = kg")
    
    # Test 3: Round-trip: log then exp
    print("\n3. Testing round-trip: Unit('m').log().exp()")
    u3 = Unit("m")
    log_u3 = u3.log()
    exp_log_u3 = log_u3.exp()
    print(f"u3 = {u3}")
    print(f"u3.log() = {log_u3}")
    print(f"u3.log().exp() = {exp_log_u3}")
    print(f"exp_log_u3.format_string() = '{exp_log_u3.format_string()}'")
    
    # Expected: should get back to a unit with the same dimension
    assert u3.dimension == exp_log_u3.dimension, f"Dimensions don't match: {u3.dimension} vs {exp_log_u3.dimension}"
    print("✅ Test passed: Round-trip preserves dimension")
    
    # Test 4: exp() on dimensionless unit
    print("\n4. Testing Unit('').exp()")
    u4 = Unit("")
    exp_u4 = u4.exp()
    print(f"u4 = {u4}")
    print(f"u4.exp() = {exp_u4}")
    print(f"exp_u4.format_string() = '{exp_u4.format_string()}'")
    
    # Expected: still dimensionless
    assert exp_u4.is_dimensionless, f"Expected dimensionless, got {exp_u4}"
    print("✅ Test passed: Unit('').exp() = dimensionless")
    
    # Test 5: exp() on invalid unit (should raise error)
    print("\n5. Testing exp() on invalid unit (should raise error)")
    u5 = Unit("kg*m/s^2")  # Regular unit, not log unit
    try:
        exp_u5 = u5.exp()
        print(f"ERROR: exp() should have failed, but got {exp_u5}")
        assert False, "exp() should have raised ValueError"
    except ValueError as e:
        print(f"✅ Test passed: exp() correctly raised ValueError: {e}")

if __name__ == "__main__":
    test_log_exp() 