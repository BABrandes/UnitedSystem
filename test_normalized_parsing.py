#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.dimension import Dimension

def test_normalized_parsing():
    print("Testing parsing of normalized strings...")
    
    test_cases = [
        "DEC(L_geo)",
        "DEC(DEC(L_geo))",
        "DEC(L_geo*T/M_thermal)",
        "DEC(DEC(L_geo*T/M_thermal))"
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case}")
        try:
            d = Dimension(case)
            print(f"  SUCCESS: {d}")
            print(f"  Format string: {d.format_string()}")
        except Exception as e:
            print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_normalized_parsing() 