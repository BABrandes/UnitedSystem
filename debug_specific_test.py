#!/usr/bin/env python3

from united_system.unit import Unit
from united_system.utils.units.utils import seperate_string

def test_specific_case(input_unit, expected_unit, case_number):
    print(f"Testing case {case_number}: {input_unit}")
    try:
        unit = Unit(input_unit)
        reduced = Unit.reduce_unit(unit)
        result = str(reduced)
        if result == expected_unit:
            print(f"  ✓ PASS: got '{result}', expected '{expected_unit}'")
        else:
            print(f"  ✗ FAIL: got '{result}', expected '{expected_unit}'")
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        return False
    return True

def debug_seperate_string():
    print("=== DEBUGGING SEPERATE_STRING FOR PROBLEMATIC CASES ===")
    
    problematic_cases = [
        "1/s_geo*1/s_geo",
        "1/s_geo*1/s_thermal"
    ]
    
    for i, test_string in enumerate(problematic_cases):
        print(f"\n--- Case {i+1}: {test_string} ---")
        try:
            parts = seperate_string(test_string, "nominator")
            print(f"seperate_string result: {parts}")
        except Exception as e:
            print(f"seperate_string error: {e}")

def main():
    # First debug the problematic cases
    debug_seperate_string()
    
    print("\n" + "="*50)
    print("RUNNING ALL TEST CASES")
    print("="*50)
    
    test_cases = [
        ("kg_thermal*kg_thermal", "kg_thermal^2", 1),
        ("kg_thermal*kg_geo", "kg_thermal*kg_geo", 2),
        ("m_elec*m_elec*m_elec", "m_elec^3", 3),
        ("m_elec*m_geo*m_thermal", "m_elec*m_geo*m_thermal", 4),
        ("kg_thermal*m^2/s^2", "kg_thermal*m^2/s^2", 5),
        ("kg_thermal*m^2/s^2*kg_thermal*m^2/s^2", "kg_thermal^2*m^4/s^4", 6),
        ("kg_thermal*m^2/s^2*kg_geo*m^2/s^2", "kg_thermal*m^4*kg_geo/s^4", 7),
        ("kg_thermal*m_elec^2/s^2", "kg_thermal*m_elec^2/s^2", 8),
        ("kg_thermal*m_elec^2/s^2*kg_thermal*m_elec^2/s^2", "kg_thermal^2*m_elec^4/s^4", 9),
        ("kg_thermal*m_elec^2/s^2*kg_geo*m_elec^2/s^2", "kg_thermal*m_elec^4*kg_geo/s^4", 10),
        ("kg_thermal*m/s^2*kg_thermal*m^2/s^2", "kg_thermal^2*m^3/s^4", 11),
        ("kg_thermal*m/s^2*kg_geo*m^2/s^2", "kg_thermal*m^3*kg_geo/s^4", 12),
        ("A_elec*V_elec", "W_elec", 13),
        ("A_elec*V_elec*A_elec*V_elec", "W_elec^2", 14),
        ("A_elec*V_elec*A_thermal*V_thermal", "W_elec*W_thermal", 15),
        ("kg_thermal*m/s^2/m^2", "kg_thermal/m/s^2", 16),
        ("kg_thermal*m/s^2/m^2*kg_thermal*m/s^2/m^2", "kg_thermal^2/m^2/s^4", 17),
        ("1/s_geo", "1/s_geo", 18),
        ("1/s_geo*1/s_geo", "1/s_geo^2", 19),
        ("1/s_geo*1/s_thermal", "1/s_geo/s_thermal", 20),
        ("kg_thermal*m_elec^2/s_geo^2", "kg_thermal*m_elec^2/s_geo^2", 21),
        ("kg_thermal*m_elec^2/s_geo^2*kg_thermal*m_elec^2/s_geo^2", "kg_thermal^2*m_elec^4/s_geo^4", 22),
        ("K_thermal", "K_thermal", 23),
        ("K_thermal*K_thermal", "K_thermal^2", 24),
        ("mol_reactant", "mol_reactant", 25),
        ("mol_reactant*mol_product", "mol_reactant*mol_product", 26),
        ("A_primary*A_secondary", "A_primary*A_secondary", 27),
        ("A_primary*A_primary", "A_primary^2", 28),
        ("V_high*V_low", "V_high*V_low", 29),
        ("V_high*V_high", "V_high^2", 30),
        ("kg_thermal*m_elec^2/s_geo^2*A_primary*V_high", "kg_thermal*m_elec^2*A_primary*V_high/s_geo^2", 31),
        ("kg_thermal*m_elec^2/s_geo^2*kg_thermal*m_elec^2/s_geo^2*A_primary*V_high", "kg_thermal^2*m_elec^4*A_primary*V_high/s_geo^4", 32),
        ("kg_geo*m_geo^2/s_geo^2*mol/m^3", "J_geo*mol/m^3", 33),
        ("kg_thermal*m_thermal^2/s_thermal^2*K/mol", "J_thermal*K/mol", 34),
        ("kg_elec*m_elec^2/s_elec^2*A*V", "J_elec*W", 35),
        ("kg_thermal*m/s", "kg_thermal*m/s", 36),
        ("kg_thermal*m/s*kg_thermal*m/s", "kg_thermal^2*m^2/s^2", 37),
        ("kg_thermal/kg_thermal", "", 38),
        ("m_elec/m_elec", "", 39),
        ("kg_thermal*m_elec^2/s_geo^2/(kg_thermal*m_elec^2/s_geo^2)", "", 40),
    ]
    
    failed_cases = []
    for input_unit, expected_unit, case_number in test_cases:
        if not test_specific_case(input_unit, expected_unit, case_number):
            failed_cases.append(case_number)
    
    if failed_cases:
        print(f"\nFailed cases: {failed_cases}")
    else:
        print("\nAll cases passed!")

if __name__ == "__main__":
    main() 