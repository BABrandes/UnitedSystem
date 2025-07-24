#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.dimension import Dimension

def test_complex_dimension_parsing():
    print("Testing complex dimension parsing...")
    
    # Test 1: Basic mixed regular and logarithmic
    print("\n1. Testing Dimension('N/mol*DEC(mm^2_geo)')")
    try:
        d1 = Dimension("N/mol*DEC(mm^2_geo)")
        print(f"  Result: {d1}")
        print(f"  Format string: {d1.format_string()}")
        print(f"  Is dimensionless: {d1.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 2: Complex logarithmic with multiple subscripts
    print("\n2. Testing Dimension('DEC(kg_thermal*m_geo/s^2_elec)')")
    try:
        d2 = Dimension("DEC(kg_thermal*m_geo/s^2_elec)")
        print(f"  Result: {d2}")
        print(f"  Format string: {d2.format_string()}")
        print(f"  Is dimensionless: {d2.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 3: Multiple logarithmic dimensions
    print("\n3. Testing Dimension('DEC(M)*DEC(L_geo/T_thermal)')")
    try:
        d3 = Dimension("DEC(M)*DEC(L_geo/T_thermal)")
        print(f"  Result: {d3}")
        print(f"  Format string: {d3.format_string()}")
        print(f"  Is dimensionless: {d3.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 4: Nested logarithmic expressions
    print("\n4. Testing Dimension('DEC(DEC(L_geo*T/M_thermal))')")
    try:
        d4 = Dimension("DEC(DEC(L_geo*T/M_thermal))")
        print(f"  Result: {d4}")
        print(f"  Format string: {d4.format_string()}")
        print(f"  Is dimensionless: {d4.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 5: Complex mixed expression
    print("\n5. Testing Dimension('kg*m/s^2*DEC(A_elec*V_thermal)')")
    try:
        d5 = Dimension("kg*m/s^2*DEC(A_elec*V_thermal)")
        print(f"  Result: {d5}")
        print(f"  Format string: {d5.format_string()}")
        print(f"  Is dimensionless: {d5.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 6: Simple logarithmic with subscript
    print("\n6. Testing Dimension('DEC(m_geo)')")
    try:
        d6 = Dimension("DEC(m_geo)")
        print(f"  Result: {d6}")
        print(f"  Format string: {d6.format_string()}")
        print(f"  Is dimensionless: {d6.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 7: Regular dimensions with subscripts
    print("\n7. Testing Dimension('L_elec*T_thermal/M_geo')")
    try:
        d7 = Dimension("L_elec*T_thermal/M_geo")
        print(f"  Result: {d7}")
        print(f"  Format string: {d7.format_string()}")
        print(f"  Is dimensionless: {d7.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 8: Complex expression with fractions
    print("\n8. Testing Dimension('(N/mol)/(DEC(mm^2_geo)*L_elec)')")
    try:
        d8 = Dimension("(N/mol)/(DEC(mm^2_geo)*L_elec)")
        print(f"  Result: {d8}")
        print(f"  Format string: {d8.format_string()}")
        print(f"  Is dimensionless: {d8.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 9: Multiple subscripts in logarithmic
    print("\n9. Testing Dimension('DEC(L_geo^2*T_thermal/M_elec)')")
    try:
        d9 = Dimension("DEC(L_geo^2*T_thermal/M_elec)")
        print(f"  Result: {d9}")
        print(f"  Format string: {d9.format_string()}")
        print(f"  Is dimensionless: {d9.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Test 10: Edge case - empty logarithmic
    print("\n10. Testing Dimension('DEC()')")
    try:
        d10 = Dimension("DEC()")
        print(f"  Result: {d10}")
        print(f"  Format string: {d10.format_string()}")
        print(f"  Is dimensionless: {d10.is_dimensionless}")
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    test_complex_dimension_parsing() 