#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.utils.units.utils import seperate_string

def reconstruct_expression(parts, position="nominator"):
    """Reconstruct an expression from seperate_string parts to validate correctness."""
    if not parts:
        return ""
    
    result = ""
    for i, (separator, part) in enumerate(parts):
        if i == 0:
            # First part: remove the leading separator and add the part
            if separator == "*":
                result = part
            elif separator == "/":
                result = f"1/{part}"
        else:
            # Subsequent parts: add the separator and part
            result += f"{separator}{part}"
    
    return result

def validate_seperate_string_correctness():
    print("=== VALIDATING SEPERATE_STRING CORRECTNESS ===\n")
    
    # Test cases with expected results
    test_cases = [
        # Simple cases
        ("A*B", "A*B", "Simple multiplication"),
        ("A/B", "A/B", "Simple division"),
        ("A*B*C", "A*B*C", "Multiple multiplication"),
        ("A/B/C", "A/B/C", "Multiple division"),
        
        # Mixed cases
        ("A*B/C*D", "A*B/C*D", "Mixed operators"),
        ("A/B*C/D", "A/B*C/D", "Alternating operators"),
        
        # Parentheses cases
        ("(A*B)", "A*B", "Simple parentheses"),
        ("(A/B)", "A/B", "Division in parentheses"),
        ("(A*B)/(C*D)", "A*B/C*D", "Parentheses with division"),
        ("((A*B)/(C*D))", "A*B/C*D", "Nested parentheses"),
        
        # Log function cases
        ("dec(L)", "dec(L)", "Simple log function"),
        ("dec(L_geo)", "dec(L_geo)", "Log function with subscript"),
        ("dec(L*M/T)", "dec(L*M/T)", "Log function with expression"),
        ("dec(dec(L))", "dec(dec(L))", "Nested log functions"),
        
        # Complex real-world cases
        ("kg_thermal*m_elec^2/s_geo^2", "kg_thermal*m_elec^2/s_geo^2", "Real unit expression"),
        ("(kg_thermal*m_elec^2/s_geo^2)/(dec(mm^2_geo)*L_elec)", "kg_thermal*m_elec^2/s_geo^2/dec(mm^2_geo)/L_elec", "Complex real expression"),
        
        # Edge cases
        ("1/A", "1/A", "Division starting with 1"),
        ("A^2", "A^2", "Exponent notation"),
        ("A_geo", "A_geo", "Subscript notation"),
        ("A_geo^2", "A_geo^2", "Subscript with exponent"),
        
        # Complex nested cases
        ("((A*B)/(C*D))/((E*F)/(G*H))", "A*B/C*D/E*F/G*H", "Complex nested division"),
        ("dec(dec(L_geo*M_elec/T_thermal))/(dec(dec(A_primary*V_high)))", "dec(dec(L_geo*M_elec/T_thermal))/dec(dec(A_primary*V_high))", "Complex nested logs"),
        
        # Stress test cases
        ("A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z", "A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z", "Long mixed expression"),
        ("dec(dec(dec(dec(dec(L_geo*M_elec/T_thermal)))))", "dec(dec(dec(dec(dec(L_geo*M_elec/T_thermal)))))", "Deep nested logs"),
    ]
    
    results = []
    
    for i, (input_str, expected_output, description) in enumerate(test_cases):
        print(f"--- Test {i+1}: {description} ---")
        print(f"Input: {input_str}")
        print(f"Expected: {expected_output}")
        
        try:
            # Test nominator position
            nominator_parts = seperate_string(input_str, "nominator")
            nominator_reconstructed = reconstruct_expression(nominator_parts, "nominator")
            
            # Test denominator position
            denominator_parts = seperate_string(input_str, "denominator")
            denominator_reconstructed = reconstruct_expression(denominator_parts, "denominator")
            
            print(f"Nominator parts: {nominator_parts}")
            print(f"Nominator reconstructed: {nominator_reconstructed}")
            print(f"Denominator parts: {denominator_parts}")
            print(f"Denominator reconstructed: {denominator_reconstructed}")
            
            # Check correctness
            nominator_correct = nominator_reconstructed == expected_output
            denominator_correct = denominator_reconstructed != expected_output  # Denominator should be different
            
            if nominator_correct:
                print("✅ Nominator: CORRECT")
            else:
                print(f"❌ Nominator: INCORRECT - Expected '{expected_output}', got '{nominator_reconstructed}'")
            
            if denominator_correct:
                print("✅ Denominator: CORRECT (different from nominator)")
            else:
                print(f"❌ Denominator: INCORRECT - Should be different from nominator")
            
            # Additional validation checks
            issues = []
            
            # Check for malformed parts
            for sep, part in nominator_parts + denominator_parts:
                if not part.strip():
                    issues.append(f"Empty part: '{part}'")
                if part.startswith(')') or part.endswith('('):
                    issues.append(f"Malformed part: '{part}'")
                if part.count('(') != part.count(')'):
                    issues.append(f"Unbalanced parentheses: '{part}'")
            
            if issues:
                print(f"⚠️  Issues found:")
                for issue in issues:
                    print(f"   {issue}")
            
            results.append({
                'test': i+1,
                'description': description,
                'input': input_str,
                'expected': expected_output,
                'nominator_correct': nominator_correct,
                'denominator_correct': denominator_correct,
                'nominator_reconstructed': nominator_reconstructed,
                'denominator_reconstructed': denominator_reconstructed,
                'issues': issues
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'test': i+1,
                'description': description,
                'input': input_str,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=== CORRECTNESS SUMMARY ===")
    successful_tests = [r for r in results if 'error' not in r]
    total_tests = len(results)
    
    if successful_tests:
        nominator_correct = sum(1 for r in successful_tests if r['nominator_correct'])
        denominator_correct = sum(1 for r in successful_tests if r['denominator_correct'])
        tests_with_issues = sum(1 for r in successful_tests if r.get('issues'))
        
        print(f"Total tests: {total_tests}")
        print(f"Successful: {len(successful_tests)}")
        print(f"Failed: {total_tests - len(successful_tests)}")
        print(f"Nominator correct: {nominator_correct}/{len(successful_tests)} ({nominator_correct/len(successful_tests)*100:.1f}%)")
        print(f"Denominator correct: {denominator_correct}/{len(successful_tests)} ({denominator_correct/len(successful_tests)*100:.1f}%)")
        print(f"Tests with issues: {tests_with_issues}")
        
        # Show incorrect results
        incorrect_nominators = [r for r in successful_tests if not r['nominator_correct']]
        if incorrect_nominators:
            print(f"\n❌ INCORRECT NOMINATOR RESULTS:")
            for r in incorrect_nominators[:5]:  # Show first 5
                print(f"  Test {r['test']}: Expected '{r['expected']}', got '{r['nominator_reconstructed']}'")
            if len(incorrect_nominators) > 5:
                print(f"  ... and {len(incorrect_nominators) - 5} more")
        
        incorrect_denominators = [r for r in successful_tests if not r['denominator_correct']]
        if incorrect_denominators:
            print(f"\n❌ INCORRECT DENOMINATOR RESULTS:")
            for r in incorrect_denominators[:5]:  # Show first 5
                print(f"  Test {r['test']}: Should be different from '{r['expected']}', got '{r['denominator_reconstructed']}'")
            if len(incorrect_denominators) > 5:
                print(f"  ... and {len(incorrect_denominators) - 5} more")
    
    # Show any errors
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\n❌ ERRORS ENCOUNTERED:")
        for r in error_results:
            print(f"  Test {r['test']} ({r['description']}): {r['error']}")

if __name__ == "__main__":
    validate_seperate_string_correctness() 