#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.utils.units.unit_symbol import LOG_UNIT_SYMBOLS

def test_log_detection():
    print("Testing log function detection...")
    
    test_parts = [
        "DEC(L_geo)",
        "dec(L_geo)",
        "bin(L_geo)",
        "nat(L_geo)",
        "DEC(dec(L_geo))",
        "dec(DEC(L_geo))",
        "L_geo",
        "M*T^2"
    ]
    
    for part in test_parts:
        print(f"\nTesting part: '{part}'")
        is_log_function = False
        for log_symbol_enum in LOG_UNIT_SYMBOLS:
            for log_symbol in log_symbol_enum.value.symbols:
                if part.startswith(log_symbol + "("):
                    is_log_function = True
                    print(f"  Matched log symbol: '{log_symbol}'")
                    break
            if is_log_function:
                break
        print(f"  is_log_function: {is_log_function}")

if __name__ == "__main__":
    test_log_detection() 