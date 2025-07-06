#!/usr/bin/env python3

"""Test script for RealArray implementation."""

import numpy as np
from united_system.real_united_array import RealUnitedArray
from src.united_system.unit import Unit
from united_system.real_united_scalar import RealUnitedScalar

def test_real_array():
    print("Testing RealArray implementation...")
    
    # Test 1: Constructor variants
    print("\n1. Testing constructor variants...")
    
    # From list with unit
    distances = RealUnitedArray([1.0, 2.0, 3.0, 4.0, 5.0], Unit("m"))
    print(f"From list with unit: {distances}")
    
    # From numpy array with unit  
    velocities = RealUnitedArray(np.array([10.0, 20.0, 30.0]), Unit("m/s"))
    print(f"From numpy array with unit: {velocities}")
    
    # Dimensionless array
    ratios = RealUnitedArray([0.1, 0.2, 0.3, 0.4, 0.5])
    print(f"Dimensionless: {ratios}")
    
    # Test 2: Properties
    print("\n2. Testing properties...")
    print(f"Distances shape: {distances.shape}")
    print(f"Distances size: {distances.size}")
    print(f"Distances dimension: {distances.dimension}")
    print(f"Distances display unit: {distances.display_unit}")
    print(f"Distances values: {distances.values}")
    print(f"Distances canonical values: {distances.canonical_values}")
    
    # Test 3: Array indexing
    print("\n3. Testing array indexing...")
    first_distance = distances[0]
    print(f"First distance: {first_distance} (type: {type(first_distance)})")
    first_two = distances[:2]
    print(f"First two distances: {first_two} (type: {type(first_two)})")
    
    # Test 4: Arithmetic operations
    print("\n4. Testing arithmetic operations...")
    
    # Array + Array
    doubled_distances = distances + distances
    print(f"distances + distances: {doubled_distances}")
    
    # Array + Scalar
    offset_distances = distances + RealUnitedScalar(1.0, Unit("m"))
    print(f"distances + 1 m: {offset_distances}")
    
    # Array + Number
    scaled_distances = distances * 2
    print(f"distances * 2: {scaled_distances}")
    
    # Array arithmetic with different units
    times = RealUnitedArray([1.0, 2.0, 3.0], Unit("s"))
    speeds = distances[:3] / times
    print(f"distances / times = speeds: {speeds}")
    
    # Test 5: Array reduction methods
    print("\n5. Testing array reduction methods...")
    total = distances.sum()
    average = distances.mean()
    maximum = distances.max()
    minimum = distances.min()
    std_dev = distances.std()
    variance = distances.var()
    
    print(f"Sum: {total} (type: {type(total)})")
    print(f"Mean: {average}")
    print(f"Max: {maximum}")
    print(f"Min: {minimum}")
    print(f"Std: {std_dev}")
    print(f"Var: {variance}")
    
    # Test 6: Unit conversion
    print("\n6. Testing unit conversion...")
    distances_in_km = distances.in_unit(Unit("km"))
    print(f"Distances in km: {distances_in_km}")
    
    distances_with_km_unit = distances.with_unit(Unit("km"))
    print(f"Distances with km display unit: {distances_with_km_unit}")
    
    # Test 7: String formatting
    print("\n7. Testing string formatting...")
    formatted = distances.format(Unit("mm"), decimals=1)
    print(f"Formatted in mm: {formatted}")
    
    # Test 8: Factory methods
    print("\n8. Testing factory methods...")
    zeros_array = RealUnitedArray.zeros(5, Unit("m"))
    print(f"Zeros array: {zeros_array}")
    
    ones_array = RealUnitedArray.ones((2, 3), Unit("kg"))
    print(f"Ones array: {ones_array}")
    
    dimensionless_array = RealUnitedArray.dimensionless([1, 2, 3, 4])
    print(f"Dimensionless array: {dimensionless_array}")
    
    # Test 9: Compatibility
    print("\n9. Testing compatibility...")
    more_distances = RealUnitedArray([6.0, 7.0], Unit("m"))
    print(f"Compatible with more distances: {distances.compatible_with(more_distances)}")
    print(f"Compatible with velocities: {distances.compatible_with(velocities)}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    test_real_array() 