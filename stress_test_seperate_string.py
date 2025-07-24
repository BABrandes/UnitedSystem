#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.utils.units.utils import seperate_string

def stress_test_seperate_string():
    print("=== STRESS TESTING SEPERATE_STRING FUNCTION ===\n")
    
    # Test cases that push the function to its limits
    extreme_test_cases = [
        # 1. Deeply nested parentheses
        ("((((A*B)/(C*D))/(E*F))/(G*H))/(I*J)", "Deep nesting"),
        
        # 2. Mixed log functions with deep nesting
        ("dec(dec(dec(dec(dec(L)))))*M/T^2", "Deep log nesting"),
        
        # 3. Complex mixed expressions
        ("(A*B/(C*D))/(E*F/(G*H))/(I*J/(K*L))", "Complex mixed division"),
        
        # 4. Nested log functions with subscripts
        ("dec(dec(L_geo*M_elec/T_thermal))/(dec(dec(A_primary*V_high)))", "Nested logs with subscripts"),
        
        # 5. Extremely complex expression
        ("(kg_thermal*m_elec^2/s_geo^2)/(dec(mm^2_geo)*L_elec)/(dec(dec(A_primary*V_high)))", "Extreme complexity"),
        
        # 6. Multiple levels of grouping with operators
        ("((A*B)/(C*D))*((E*F)/(G*H))/((I*J)/(K*L))", "Multiple grouping levels"),
        
        # 7. Log functions inside deeply nested parentheses
        ("(((dec(L_geo)*M_elec)/(dec(T_thermal)))/(dec(dec(A_primary))))/(V_high)", "Logs in deep nesting"),
        
        # 8. Alternating multiplication and division with nesting
        ("A*B/(C*D)*E/F/(G*H)*I/J/(K*L)", "Alternating operators"),
        
        # 9. Complex expression with all features
        ("(kg_thermal*m_elec^2/s_geo^2*A_primary*V_high)/(dec(mm^2_geo)*L_elec)/(dec(dec(A_primary*V_high)))", "All features combined"),
        
        # 10. Edge case: Empty parentheses
        ("()", "Empty parentheses"),
        
        # 11. Edge case: Single character in parentheses
        ("(A)", "Single character"),
        
        # 12. Edge case: Multiple empty parentheses
        ("()()()", "Multiple empty"),
        
        # 13. Edge case: Nested empty parentheses
        ("((()))", "Nested empty"),
        
        # 14. Very long expression
        ("A*B*C*D*E*F*G*H*I*J*K*L*M*N*O*P*Q*R*S*T*U*V*W*X*Y*Z", "Very long expression"),
        
        # 15. Complex division chain
        ("A/B/C/D/E/F/G/H/I/J/K/L/M/N/O/P/Q/R/S/T/U/V/W/X/Y/Z", "Division chain"),
        
        # 16. Mixed long expression
        ("A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z", "Mixed long expression"),
        
        # 17. Log functions with complex arguments
        ("dec(A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z)", "Log with complex args"),
        
        # 18. Multiple log functions with nesting
        ("dec(dec(dec(A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z)))", "Multiple nested logs"),
        
        # 19. Complex expression with all operators and functions
        ("(A*B/(C*D))/(E*F/(G*H))/(I*J/(K*L))/(M*N/(O*P))/(Q*R/(S*T))/(U*V/(W*X))/(Y*Z)", "All operators"),
        
        # 20. Ultimate stress test
        ("((((((((((A*B)/(C*D))/(E*F))/(G*H))/(I*J))/(K*L))/(M*N))/(O*P))/(Q*R))/(S*T))/(U*V)/(W*X)/(Y*Z)", "Ultimate nesting"),
    ]
    
    results = []
    
    for i, (test_string, description) in enumerate(extreme_test_cases):
        print(f"--- Test {i+1}: {description} ---")
        print(f"Input: {test_string}")
        
        try:
            # Test nominator position
            nominator_parts = seperate_string(test_string, "nominator")
            print(f"Nominator result: {nominator_parts}")
            
            # Test denominator position
            denominator_parts = seperate_string(test_string, "denominator")
            print(f"Denominator result: {denominator_parts}")
            
            # Check for potential issues
            issues = []
            
            # Check for empty parts
            for sep, part in nominator_parts + denominator_parts:
                if not part.strip():
                    issues.append(f"Empty part found: '{part}'")
                if part.startswith(')') or part.endswith('('):
                    issues.append(f"Malformed part: '{part}'")
                if part.count('(') != part.count(')'):
                    issues.append(f"Unbalanced parentheses in part: '{part}'")
            
            if issues:
                print(f"⚠️  ISSUES FOUND:")
                for issue in issues:
                    print(f"   {issue}")
            else:
                print("✅ No obvious issues detected")
            
            results.append({
                'test': i+1,
                'description': description,
                'input': test_string,
                'nominator_parts': len(nominator_parts),
                'denominator_parts': len(denominator_parts),
                'issues': issues
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'test': i+1,
                'description': description,
                'input': test_string,
                'error': str(e)
            })
        
        print()
    
    # Summary
    print("=== SUMMARY ===")
    successful_tests = sum(1 for r in results if 'error' not in r)
    total_tests = len(results)
    tests_with_issues = sum(1 for r in results if 'error' not in r and r.get('issues'))
    
    print(f"Total tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"With issues: {tests_with_issues}")
    
    # Show the most complex successful parsing
    complex_results = [r for r in results if 'error' not in r and 'nominator_parts' in r]
    if complex_results:
        most_complex = max(complex_results, key=lambda x: x['nominator_parts'] + x['denominator_parts'])
        print(f"\nMost complex successful parsing:")
        print(f"  Test: {most_complex['description']}")
        print(f"  Input: {most_complex['input']}")
        print(f"  Total parts: {most_complex['nominator_parts'] + most_complex['denominator_parts']}")
    
    # Show any errors
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nErrors encountered:")
        for r in error_results:
            print(f"  Test {r['test']} ({r['description']}): {r['error']}")

if __name__ == "__main__":
    stress_test_seperate_string() 