#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.dimension import Dimension
from united_system.utils.units.unit_symbol import LOG_UNIT_SYMBOLS

def test_all_log_functions():
    print("Testing all available log functions from LOG_UNIT_SYMBOLS...")
    
    # First, let's see what log functions are available
    print("\nAvailable log functions:")
    for log_symbol_enum in LOG_UNIT_SYMBOLS:
        print(f"  {log_symbol_enum.name}: {log_symbol_enum.value.symbols}")
    
    # Test cases for each log function
    test_cases = [
        # Base 10 log functions
        "dec(m_geo)",
        "decade(m_geo)", 
        "mag(m_geo)",
        "magnitude(m_geo)",
        
        # Base 2 log functions
        "bin(m_geo)",
        
        # Base e log functions
        "nat(m_geo)",
        
        # Complex cases
        "dec(kg_thermal*m_geo/s^2_elec)",
        "bin(L_elec*T_thermal/M_geo)",
        "nat(DEC(L_geo))",
        
        # Mixed cases
        "N/mol*dec(mm^2_geo)",
        "kg*m/s^2*bin(A_elec*V_thermal)",
        "(N/mol)/(nat(mm^2_geo)*L_elec)"
    ]
    
    print(f"\nTesting {len(test_cases)} cases:")
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing: {case}")
        try:
            d = Dimension(case)
            print(f"   SUCCESS: {d}")
            print(f"   Format string: {d.format_string()}")
        except Exception as e:
            print(f"   ERROR: {e}")

if __name__ == "__main__":
    test_all_log_functions() 