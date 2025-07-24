#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.utils.units.unit_symbol import LOG_UNIT_SYMBOLS

def test_normalization():
    print("Testing log function normalization...")
    
    test_cases = [
        "dec(L_geo)",
        "DEC(L_geo)",
        "bin(L_geo)",
        "nat(L_geo)",
        "dec(DEC(L_geo))",
        "DEC(dec(L_geo))",
        "bin(dec(L_geo))",
        "dec(bin(L_geo))"
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case}")
        normalized = case
        for log_symbol_enum in LOG_UNIT_SYMBOLS:
            for log_symbol in log_symbol_enum.value.symbols:
                if log_symbol != "DEC":  # Don't replace DEC with DEC
                    normalized = normalized.replace(f"{log_symbol}(", "DEC(")
        print(f"  Normalized: {normalized}")

if __name__ == "__main__":
    test_normalization() 