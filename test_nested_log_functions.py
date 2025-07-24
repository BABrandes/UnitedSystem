#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.dimension import Dimension

def test_nested_log_functions():
    print("Testing nested log functions...")
    
    # Test cases for nested log functions
    test_cases = [
        # Simple nested cases
        "DEC(dec(L_geo))",
        "dec(DEC(L_geo))",
        "bin(dec(L_geo))",
        "dec(bin(L_geo))",
        "nat(DEC(L_geo))",
        "DEC(nat(L_geo))",
        
        # More complex nested cases
        "DEC(dec(L_geo*T/M_thermal))",
        "dec(DEC(L_geo*T/M_thermal))",
        "bin(dec(L_elec*T_thermal/M_geo))",
        
        # Triple nested
        "DEC(dec(DEC(L_geo)))",
        "dec(DEC(dec(L_geo)))",
        
        # Mixed with regular dimensions
        "N/mol*DEC(dec(mm^2_geo))",
        "kg*m/s^2*dec(DEC(A_elec*V_thermal))"
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {case}")
        try:
            d = Dimension(case)
            print(f"   SUCCESS: {d}")
            print(f"   Format string: {d.format_string()}")
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_nested_log_functions() 