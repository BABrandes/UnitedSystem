#!/usr/bin/env python3

"""Test script for mixin-based RealUnitedArray implementation."""

import numpy as np
from src.united_system.arrays.real_united_array.real_united_array import RealUnitedArray
from united_system.units.simple.simple_unit import SimpleUnit

def test_mixin_array():
    print("Testing mixin-based RealUnitedArray implementation...")
    
    # Create a simple unit for testing
    meter = SimpleUnit.parse_string("m")
    
    # Test 1: Create an array from values and unit
    print("\n1. Creating array from values and unit...")
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    distances = RealUnitedArray.from_values_and_unit(values, meter)
    print(f"Distances: {distances}")
    
    # Test 2: Test arithmetic operations
    print("\n2. Testing arithmetic operations...")
    doubled = distances * 2
    print(f"Doubled distances: {doubled}")
    
    # Test 3: Test array reduction operations
    print("\n3. Testing array reduction operations...")
    total_distance = distances.sum()
    mean_distance = distances.mean()
    max_distance = distances.max()
    print(f"Total distance: {total_distance}")
    print(f"Mean distance: {mean_distance}")
    print(f"Max distance: {max_distance}")
    
    # Test 4: Test array access
    print("\n4. Testing array access...")
    print(f"Array length: {len(distances)}")
    print(f"Array shape: {distances.shape}")
    print(f"First element: {distances[0]}")
    print(f"First two elements: {distances[:2]}")
    
    # Test 5: Test unit conversion
    print("\n5. Testing unit conversion...")
    km = SimpleUnit.parse_string("km")
    distances_in_km = distances.to_unit(km)
    print(f"Distances in km: {distances_in_km}")
    
    # Test 6: Test formatting
    print("\n6. Testing formatting...")
    formatted = distances.format(km, decimals=3)
    print(f"Formatted distances: {formatted}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_mixin_array() 