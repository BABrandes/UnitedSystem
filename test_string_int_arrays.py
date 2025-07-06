"""
Simple test script for StringArray and IntArray classes.

This demonstrates the enhanced functionality of the wrapper classes.
"""

import sys
sys.path.insert(0, 'src')

from united_system.string_array import StringArray
from united_system.int_array import IntArray
import numpy as np

def test_string_array():
    """Test StringArray functionality."""
    print("=== Testing StringArray ===")
    
    # Construction
    str_arr = StringArray(['apple', 'banana', 'cherry', 'apple'])
    print(f"Created: {str_arr}")
    print(f"Shape: {str_arr.shape}, Size: {str_arr.size}")
    
    # Array access
    print(f"First element: {str_arr[0]}")
    print(f"Last two elements: {str_arr[-2:]}")
    print(f"Length: {len(str_arr)}")
    
    # Containment and counting
    print(f"Contains 'apple': {'apple' in str_arr}")
    print(f"Count of 'apple': {str_arr.count('apple')}")
    
    # Utility methods
    print(f"Unique values: {str_arr.unique()}")
    
    # Concatenation
    str_arr2 = StringArray(['date', 'elderberry'])
    combined = str_arr.concatenate(str_arr2)
    print(f"Concatenated: {combined}")
    
    # Factory methods
    empty = StringArray.empty(3)
    print(f"Empty array shape: {empty.shape}")
    
    filled = StringArray.full((2, 2), 'default')
    print(f"Filled array: {filled}")
    
    # Iteration
    print("Iteration:")
    for i, item in enumerate(str_arr):
        if i < 3:  # Just show first 3
            print(f"  [{i}]: {item}")


def test_int_array():
    """Test IntArray functionality."""
    print("\n=== Testing IntArray ===")
    
    # Construction
    int_arr = IntArray([10, 20, 30, 40, 50])
    print(f"Created: {int_arr}")
    print(f"Shape: {int_arr.shape}, Size: {int_arr.size}")
    
    # Array access
    print(f"Third element: {int_arr[2]}")
    print(f"Middle three: {int_arr[1:4]}")
    
    # Arithmetic operations
    print(f"Array + 5: {int_arr + 5}")
    print(f"Array * 2: {int_arr * 2}")
    print(f"Array ** 2: {int_arr ** 2}")
    print(f"Array - 10: {int_arr - 10}")
    
    # Array-to-array operations
    int_arr2 = IntArray([1, 2, 3, 4, 5])
    print(f"Array1 + Array2: {int_arr + int_arr2}")
    print(f"Array1 * Array2: {int_arr * int_arr2}")
    
    # Reduction operations
    print(f"Sum: {int_arr.sum()}")
    print(f"Mean: {int_arr.mean()}")
    print(f"Min: {int_arr.min()}, Max: {int_arr.max()}")
    print(f"Std: {int_arr.std():.2f}, Var: {int_arr.var():.2f}")
    
    # Comparison operations
    print(f"Elements > 25: {int_arr > 25}")
    print(f"Elements == 30: {int_arr == 30}")
    
    # Utility methods
    int_arr3 = IntArray([30, 10, 50, 20, 40])
    print(f"Unsorted: {int_arr3}")
    print(f"Sorted: {int_arr3.sort()}")
    print(f"Unique: {int_arr3.unique()}")
    
    # Factory methods
    zeros = IntArray.zeros(4)
    print(f"Zeros: {zeros}")
    
    ones = IntArray.ones((2, 3))
    print(f"Ones: {ones}")
    
    arange = IntArray.arange(1, 10, 2)
    print(f"Range 1-10 step 2: {arange}")
    
    # Concatenation
    combined = int_arr.concatenate(int_arr2)
    print(f"Concatenated: {combined}")


def test_edge_cases():
    """Test edge cases and error conditions."""
    print("\n=== Testing Edge Cases ===")
    
    # Empty arrays
    empty_str = StringArray([])
    empty_int = IntArray([])
    print(f"Empty string array shape: {empty_str.shape}")
    print(f"Empty int array shape: {empty_int.shape}")
    
    # Single element arrays
    single_str = StringArray(['only'])
    single_int = IntArray([42])
    print(f"Single string: {single_str[0]}")
    print(f"Single int: {single_int[0]}")
    
    # Multi-dimensional arrays
    matrix_int = IntArray(np.array([[1, 2], [3, 4]]))
    print(f"2D int array: {matrix_int}")
    print(f"Shape: {matrix_int.shape}")
    
    # Invalid operations should be handled gracefully
    try:
        # Division by zero
        int_arr = IntArray([1, 2, 3])
        result = int_arr / 0
        print("Division by zero handled")
    except Exception as e:
        print(f"Division by zero error (expected): {type(e).__name__}")


if __name__ == "__main__":
    test_string_array()
    test_int_array()
    test_edge_cases()
    print("\n✅ All tests completed successfully!") 