#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.utils.units.utils import seperate_string
import time

def extreme_stress_test():
    print("=== EXTREME STRESS TESTING SEPERATE_STRING FUNCTION ===\n")
    
    # Ultra-extreme test cases
    ultra_extreme_cases = [
        # 1. Maximum nesting depth (20 levels)
        ("(" * 20 + "A*B" + ")" * 20, "20-level nesting"),
        
        # 2. Maximum nesting with complex content
        ("(" * 15 + "dec(dec(dec(L_geo*M_elec/T_thermal)))" + ")" * 15, "15-level nesting with logs"),
        
        # 3. Alternating nesting with operators
        ("(" * 10 + "A*B/(C*D)*E/F/(G*H)*I/J/(K*L)" + ")" * 10, "10-level alternating nesting"),
        
        # 4. Mixed nesting with all features
        ("(" * 8 + "kg_thermal*m_elec^2/s_geo^2*A_primary*V_high/(dec(mm^2_geo)*L_elec)/(dec(dec(A_primary*V_high)))" + ")" * 8, "8-level mixed features"),
        
        # 5. Recursive log functions with maximum nesting
        ("dec(" * 10 + "L_geo*M_elec/T_thermal" + ")" * 10, "10-level recursive logs"),
        
        # 6. Complex expression with maximum operators
        ("A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z*A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z", "Maximum operators"),
        
        # 7. Nested parentheses with complex content
        ("((((((((((A*B/(C*D))/(E*F/(G*H)))/(I*J/(K*L)))/(M*N/(O*P)))/(Q*R/(S*T)))/(U*V/(W*X)))/(Y*Z))/(A*B))/(C*D))/(E*F))/(G*H)", "Complex nested division"),
        
        # 8. Log functions with maximum complexity
        ("dec(dec(dec(dec(dec(dec(dec(dec(dec(dec(A*B/C*D/E*F/G*H*I/J*K/L*M/N*O/P*Q/R*S/T*U/V*W/X*Y*Z))))))))))", "Maximum log complexity"),
        
        # 9. Mixed extreme case
        ("(" * 5 + "dec(" * 5 + "kg_thermal*m_elec^2/s_geo^2*A_primary*V_high/(dec(mm^2_geo)*L_elec)/(dec(dec(A_primary*V_high)))" + ")" * 5 + ")" * 5, "Mixed extreme"),
        
        # 10. Ultimate stress test
        ("(" * 12 + "dec(" * 8 + "A*B/(C*D)/(E*F/(G*H))/(I*J/(K*L))/(M*N/(O*P))/(Q*R/(S*T))/(U*V/(W*X))/(Y*Z)" + ")" * 8 + ")" * 12, "Ultimate stress test"),
    ]
    
    results = []
    
    for i, (test_string, description) in enumerate(ultra_extreme_cases):
        print(f"--- Ultra Test {i+1}: {description} ---")
        print(f"Input length: {len(test_string)} characters")
        print(f"Input: {test_string[:100]}{'...' if len(test_string) > 100 else ''}")
        
        start_time = time.time()
        
        try:
            # Test nominator position
            nominator_parts = seperate_string(test_string, "nominator")
            nominator_time = time.time() - start_time
            
            # Test denominator position
            denominator_start = time.time()
            denominator_parts = seperate_string(test_string, "denominator")
            denominator_time = time.time() - denominator_start
            
            total_time = time.time() - start_time
            
            print(f"✅ SUCCESS")
            print(f"  Nominator parts: {len(nominator_parts)}")
            print(f"  Denominator parts: {len(denominator_parts)}")
            print(f"  Total time: {total_time:.4f}s")
            print(f"  Nominator time: {nominator_time:.4f}s")
            print(f"  Denominator time: {denominator_time:.4f}s")
            
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
                for issue in issues[:3]:  # Show only first 3 issues
                    print(f"   {issue}")
                if len(issues) > 3:
                    print(f"   ... and {len(issues) - 3} more issues")
            else:
                print("✅ No obvious issues detected")
            
            results.append({
                'test': i+1,
                'description': description,
                'input_length': len(test_string),
                'nominator_parts': len(nominator_parts),
                'denominator_parts': len(denominator_parts),
                'total_time': total_time,
                'issues': issues
            })
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append({
                'test': i+1,
                'description': description,
                'input_length': len(test_string),
                'error': str(e)
            })
        
        print()
    
    # Performance analysis
    print("=== PERFORMANCE ANALYSIS ===")
    successful_tests = [r for r in results if 'error' not in r]
    
    if successful_tests:
        avg_time = sum(r['total_time'] for r in successful_tests) / len(successful_tests)
        max_time = max(r['total_time'] for r in successful_tests)
        min_time = min(r['total_time'] for r in successful_tests)
        
        print(f"Average processing time: {avg_time:.4f}s")
        print(f"Fastest test: {min_time:.4f}s")
        print(f"Slowest test: {max_time:.4f}s")
        
        # Find the most complex test
        most_complex = max(successful_tests, key=lambda x: x['nominator_parts'] + x['denominator_parts'])
        print(f"\nMost complex test:")
        print(f"  Description: {most_complex['description']}")
        print(f"  Input length: {most_complex['input_length']} characters")
        print(f"  Total parts: {most_complex['nominator_parts'] + most_complex['denominator_parts']}")
        print(f"  Processing time: {most_complex['total_time']:.4f}s")
    
    # Summary
    print("\n=== SUMMARY ===")
    successful_count = len(successful_tests)
    total_count = len(results)
    tests_with_issues = sum(1 for r in successful_tests if r.get('issues'))
    
    print(f"Total ultra tests: {total_count}")
    print(f"Successful: {successful_count}")
    print(f"Failed: {total_count - successful_count}")
    print(f"With issues: {tests_with_issues}")
    
    # Show any errors
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\nErrors encountered:")
        for r in error_results:
            print(f"  Test {r['test']} ({r['description']}): {r['error']}")
    
    # Memory usage estimation
    if successful_tests:
        total_parts = sum(r['nominator_parts'] + r['denominator_parts'] for r in successful_tests)
        print(f"\nTotal parts processed: {total_parts}")
        print(f"Average parts per test: {total_parts / len(successful_tests):.1f}")

if __name__ == "__main__":
    extreme_stress_test() 