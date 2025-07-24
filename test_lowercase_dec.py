#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.dimension import Dimension

def test_lowercase_dec():
    print("Testing lowercase dec() function parsing...")
    
    # Test the specific case you mentioned
    test_cases = [
        "(N/mol)/(dec(mm^2_geo)*L_elec)",
        "dec(m_geo)",
        "DEC(dec(L_geo))",
        "dec(DEC(L_geo))",
        "dec(kg_thermal*m_geo/s^2_elec)",
        "N/mol*dec(mm^2_geo)",
        "kg*m/s^2*dec(A_elec*V_thermal)"
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
    test_lowercase_dec() 