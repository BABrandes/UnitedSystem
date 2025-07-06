#!/usr/bin/env python3
"""
Comprehensive test file for BoolArray, FloatArray, TimestampArray, and ComplexArray classes.
Demonstrates all the functionality and operations available for each array type.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.bool_array import BoolArray
from united_system.float_array import FloatArray
from united_system.timestamp_array import TimestampArray
from united_system.complex_array import ComplexArray

def test_bool_array():
    """Test BoolArray functionality."""
    print("=" * 60)
    print("TESTING BOOL ARRAY")
    print("=" * 60)
    
    # Construction
    bool_arr = BoolArray([True, False, True, False, True])
    print(f"1. Construction: {bool_arr}")
    print(f"   Shape: {bool_arr.shape}, Size: {bool_arr.size}")
    
    # Array access
    print(f"2. Array access: arr[0] = {bool_arr[0]}, arr[1:3] = {bool_arr[1:3]}")
    print(f"   Length: {len(bool_arr)}")
    print(f"   Contains True: {True in bool_arr}")
    
    # Logical operations
    other_bool = BoolArray([False, True, False, True, False])
    print(f"3. Logical operations:")
    print(f"   AND: {bool_arr & other_bool}")
    print(f"   OR: {bool_arr | other_bool}")
    print(f"   XOR: {bool_arr ^ other_bool}")
    print(f"   NOT: {~bool_arr}")
    
    # Comparison operations
    print(f"4. Comparison operations:")
    print(f"   Equality: {bool_arr == other_bool}")
    print(f"   Inequality: {bool_arr != other_bool}")
    
    # Reduction methods
    print(f"5. Reduction methods:")
    print(f"   Any: {bool_arr.any()}")
    print(f"   All: {bool_arr.all()}")
    print(f"   Count true: {bool_arr.count_true()}")
    print(f"   Count false: {bool_arr.count_false()}")
    
    # Factory methods
    print(f"6. Factory methods:")
    print(f"   True array: {BoolArray.true(3)}")
    print(f"   False array: {BoolArray.false(3)}")
    print(f"   Random array: {BoolArray.random(5, 0.3)}")
    
    # Utility methods
    print(f"7. Utility methods:")
    print(f"   Unique: {bool_arr.unique()}")
    print(f"   Concatenated: {bool_arr.concatenate(other_bool)}")
    
    # Serialization
    json_data = bool_arr.to_json()
    bool_arr_restored = BoolArray.from_json(json_data)
    print(f"8. Serialization: Original = {bool_arr}, Restored = {bool_arr_restored}")
    
    print()

def test_float_array():
    """Test FloatArray functionality."""
    print("=" * 60)
    print("TESTING FLOAT ARRAY")
    print("=" * 60)
    
    # Construction
    float_arr = FloatArray([1.5, 2.7, 3.14, 4.2, 5.8])
    print(f"1. Construction: {float_arr}")
    print(f"   Shape: {float_arr.shape}, Size: {float_arr.size}")
    
    # Array access
    print(f"2. Array access: arr[0] = {float_arr[0]}, arr[1:3] = {float_arr[1:3]}")
    print(f"   Length: {len(float_arr)}")
    print(f"   Contains 3.14: {3.14 in float_arr}")
    
    # Arithmetic operations
    other_float = FloatArray([0.5, 1.3, 2.86, 1.8, 2.2])
    print(f"3. Arithmetic operations:")
    print(f"   Addition: {float_arr + other_float}")
    print(f"   Subtraction: {float_arr - other_float}")
    print(f"   Multiplication: {float_arr * 2}")
    print(f"   Division: {float_arr / 2}")
    print(f"   Power: {float_arr ** 2}")
    print(f"   Negation: {-float_arr}")
    print(f"   Absolute: {abs(float_arr)}")
    
    # Mathematical functions
    print(f"4. Mathematical functions:")
    print(f"   Square root: {float_arr.sqrt()}")
    print(f"   Sine: {float_arr.sin()}")
    print(f"   Cosine: {float_arr.cos()}")
    print(f"   Exponential: {float_arr.exp()}")
    print(f"   Natural log: {float_arr.log()}")
    print(f"   Floor: {float_arr.floor()}")
    print(f"   Ceiling: {float_arr.ceil()}")
    print(f"   Round: {float_arr.round(1)}")
    
    # Reduction methods
    print(f"5. Reduction methods:")
    print(f"   Sum: {float_arr.sum()}")
    print(f"   Mean: {float_arr.mean()}")
    print(f"   Min: {float_arr.min()}")
    print(f"   Max: {float_arr.max()}")
    print(f"   Std: {float_arr.std()}")
    print(f"   Var: {float_arr.var()}")
    print(f"   Median: {float_arr.median()}")
    
    # Factory methods
    print(f"6. Factory methods:")
    print(f"   Zeros: {FloatArray.zeros(3)}")
    print(f"   Ones: {FloatArray.ones(3)}")
    print(f"   Full: {FloatArray.full(3, 7.5)}")
    print(f"   Linspace: {FloatArray.linspace(0, 10, 5)}")
    print(f"   Arange: {FloatArray.arange(0, 10, 2.5)}")
    
    # Utility methods
    print(f"7. Utility methods:")
    print(f"   Unique: {float_arr.unique()}")
    print(f"   Sorted: {float_arr.sort()}")
    print(f"   Clipped: {float_arr.clip(2.0, 4.0)}")
    print(f"   Is finite: {float_arr.is_finite()}")
    
    # Serialization
    json_data = float_arr.to_json()
    float_arr_restored = FloatArray.from_json(json_data)
    print(f"8. Serialization: Original = {float_arr}, Restored = {float_arr_restored}")
    
    print()

def test_timestamp_array():
    """Test TimestampArray functionality."""
    print("=" * 60)
    print("TESTING TIMESTAMP ARRAY")
    print("=" * 60)
    
    # Construction
    timestamp_arr = TimestampArray(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    print(f"1. Construction: {timestamp_arr}")
    print(f"   Shape: {timestamp_arr.shape}, Size: {timestamp_arr.size}")
    
    # Array access
    print(f"2. Array access: arr[0] = {timestamp_arr[0]}, arr[1:3] = {timestamp_arr[1:3]}")
    print(f"   Length: {len(timestamp_arr)}")
    print(f"   Contains '2023-01-01': {'2023-01-01' in timestamp_arr}")
    
    # Date/time properties
    print(f"3. Date/time properties:")
    print(f"   Year: {timestamp_arr.year}")
    print(f"   Month: {timestamp_arr.month}")
    print(f"   Day: {timestamp_arr.day}")
    print(f"   Weekday: {timestamp_arr.weekday}")
    
    # Arithmetic operations with timedelta
    print(f"4. Arithmetic operations:")
    print(f"   Add 1 day: {timestamp_arr + timedelta(days=1)}")
    print(f"   Subtract 2 days: {timestamp_arr - timedelta(days=2)}")
    
    # Date/time specific operations
    print(f"5. Date/time operations:")
    print(f"   Floor to day: {timestamp_arr.floor('D')}")
    print(f"   Normalize: {timestamp_arr.normalize()}")
    print(f"   String format: {timestamp_arr.strftime('%Y-%m-%d')[:3]}")  # First 3 elements
    
    # Reduction methods
    print(f"6. Reduction methods:")
    print(f"   Min: {timestamp_arr.min()}")
    print(f"   Max: {timestamp_arr.max()}")
    print(f"   Range: {timestamp_arr.range()}")
    
    # Factory methods
    print(f"7. Factory methods:")
    print(f"   Date range: {TimestampArray.date_range('2023-01-01', '2023-01-05')}")
    print(f"   From strings: {TimestampArray.from_strings(['2023-01-01', '2023-01-02'])}")
    
    # Utility methods
    print(f"8. Utility methods:")
    print(f"   Unique: {timestamp_arr.unique()}")
    print(f"   Sorted: {timestamp_arr.sort()}")
    print(f"   Is weekend: {timestamp_arr.is_weekend()}")
    print(f"   Is month start: {timestamp_arr.is_month_start()}")
    
    # Serialization
    json_data = timestamp_arr.to_json()
    timestamp_arr_restored = TimestampArray.from_json(json_data)
    print(f"9. Serialization: Original = {timestamp_arr}, Restored = {timestamp_arr_restored}")
    
    print()

def test_complex_array():
    """Test ComplexArray functionality."""
    print("=" * 60)
    print("TESTING COMPLEX ARRAY")
    print("=" * 60)
    
    # Construction
    complex_arr = ComplexArray([1+2j, 3+4j, 5+6j, 7+8j, 9+10j])
    print(f"1. Construction: {complex_arr}")
    print(f"   Shape: {complex_arr.shape}, Size: {complex_arr.size}")
    
    # Array access
    print(f"2. Array access: arr[0] = {complex_arr[0]}, arr[1:3] = {complex_arr[1:3]}")
    print(f"   Length: {len(complex_arr)}")
    print(f"   Contains 1+2j: {1+2j in complex_arr}")
    
    # Complex properties
    print(f"3. Complex properties:")
    print(f"   Real: {complex_arr.real}")
    print(f"   Imaginary: {complex_arr.imag}")
    
    # Arithmetic operations
    other_complex = ComplexArray([1+1j, 2+2j, 3+3j, 4+4j, 5+5j])
    print(f"4. Arithmetic operations:")
    print(f"   Addition: {complex_arr + other_complex}")
    print(f"   Subtraction: {complex_arr - other_complex}")
    print(f"   Multiplication: {complex_arr * 2}")
    print(f"   Division: {complex_arr / 2}")
    print(f"   Power: {complex_arr ** 2}")
    print(f"   Negation: {-complex_arr}")
    print(f"   Absolute: {abs(complex_arr)}")
    
    # Complex-specific operations
    print(f"5. Complex-specific operations:")
    print(f"   Conjugate: {complex_arr.conjugate()}")
    print(f"   Magnitude: {complex_arr.magnitude()}")
    print(f"   Phase: {complex_arr.phase()}")
    print(f"   Phase degrees: {complex_arr.phase_degrees()}")
    
    # Mathematical functions
    print(f"6. Mathematical functions:")
    print(f"   Square root: {complex_arr.sqrt()}")
    print(f"   Exponential: {complex_arr.exp()}")
    print(f"   Natural log: {complex_arr.log()}")
    print(f"   Sine: {complex_arr.sin()}")
    print(f"   Cosine: {complex_arr.cos()}")
    
    # Reduction methods
    print(f"7. Reduction methods:")
    print(f"   Sum: {complex_arr.sum()}")
    print(f"   Mean: {complex_arr.mean()}")
    print(f"   Min magnitude: {complex_arr.min_magnitude()}")
    print(f"   Max magnitude: {complex_arr.max_magnitude()}")
    
    # Factory methods
    print(f"8. Factory methods:")
    print(f"   Zeros: {ComplexArray.zeros(3)}")
    print(f"   Ones: {ComplexArray.ones(3)}")
    print(f"   From real/imag: {ComplexArray.from_real_imag([1, 2, 3], [4, 5, 6])}")
    print(f"   From polar: {ComplexArray.from_polar([1, 2, 3], [0, np.pi/2, np.pi])}")
    
    # Utility methods
    print(f"9. Utility methods:")
    print(f"   Unique: {complex_arr.unique()}")
    print(f"   Is real: {complex_arr.is_real()}")
    print(f"   Is finite: {complex_arr.is_finite()}")
    print(f"   To polar: {complex_arr.to_polar()}")
    
    # Serialization
    json_data = complex_arr.to_json()
    complex_arr_restored = ComplexArray.from_json(json_data)
    print(f"10. Serialization: Original = {complex_arr}, Restored = {complex_arr_restored}")
    
    print()

def main():
    """Run all tests."""
    print("COMPREHENSIVE TEST OF ARRAY WRAPPER CLASSES")
    print("=" * 60)
    print()
    
    test_bool_array()
    test_float_array()
    test_timestamp_array()
    test_complex_array()
    
    print("=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    
    # Summary
    print("SUMMARY OF ARRAY TYPES:")
    print("- BoolArray: Boolean logic, reduction operations (any, all, count)")
    print("- FloatArray: Full arithmetic, mathematical functions, statistics")
    print("- TimestampArray: Date/time operations, formatting, periods")
    print("- ComplexArray: Complex arithmetic, polar/rectangular conversion")
    print()
    print("Each array type provides:")
    print("- Convenient constructors and factory methods")
    print("- Type-specific operations and properties")
    print("- Comprehensive serialization support")
    print("- Intuitive array access and iteration")
    print("- Proper string representation")

if __name__ == "__main__":
    main() 