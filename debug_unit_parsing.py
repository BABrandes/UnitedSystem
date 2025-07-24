#!/usr/bin/env python3

from united_system.unit import Unit
from united_system.utils.units.utils import seperate_string

def debug_unit_parsing():
    print("=== DEBUGGING UNIT PARSING ===")
    
    # Test the dimensionless cases that should reduce to empty string
    dimensionless_cases = [
        "kg_thermal/kg_thermal",
        "m_elec/m_elec", 
        "kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)"
    ]
    
    for i, test_string in enumerate(dimensionless_cases):
        print(f"\n--- Test case {i+1}: {test_string} ---")
        try:
            unit = Unit(test_string)
            print(f"Unit created: {unit}")
            print(f"Unit format_string: {repr(unit.format_string())}")
            
            reduced = Unit.reduce_unit(unit)
            print(f"Reduced unit: {reduced}")
            print(f"Reduced format_string: {repr(reduced.format_string())}")
            print(f"Reduced str(): {repr(str(reduced))}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_unit_parsing() 