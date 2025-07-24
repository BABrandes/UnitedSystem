#!/usr/bin/env python3

from united_system.unit import Unit
from united_system.named_quantity import NamedQuantity, CONSIDER_FOR_REDUCTION_QUANTITIES

def debug_reduction():
    print("=== DEBUGGING UNIT REDUCTION ===")
    
    # Test the failing case
    test_unit = Unit("A_elec*V_elec*A_elec*V_elec")
    print(f"Original unit: {test_unit}")
    print(f"Unit elements: {test_unit.unit_elements}")
    
    # Print each element
    for subscript, elements in test_unit.unit_elements.items():
        print(f"  Subscript '{subscript}':")
        for element in elements:
            print(f"    {element.unit_symbol.value.symbols[0]} (exponent: {element.exponent})")
    
    print("\n--- Checking dimensions ---")
    # Check what the test unit's dimension is
    test_dimension = test_unit.dimension
    print(f"Test unit dimension: {test_dimension}")
    
    # Check what POWER (W) dimension is
    power_dimension = NamedQuantity.POWER.dimension
    print(f"POWER (W) dimension: {power_dimension}")
    
    # Check if they match
    print(f"Dimensions match: {test_dimension == power_dimension}")
    
    # Check what's in CONSIDER_FOR_REDUCTION_QUANTITIES
    print(f"\n--- CONSIDER_FOR_REDUCTION_QUANTITIES ---")
    for quantity in CONSIDER_FOR_REDUCTION_QUANTITIES:
        print(f"  {quantity.name}: {quantity.value[0]} -> {quantity.dimension}")
    
    print("\n--- Testing individual A*V reduction ---")
    # Test just A*V to see if it reduces to W
    simple_unit = Unit("A*V")
    print(f"Simple unit: {simple_unit}")
    print(f"Simple unit dimension: {simple_unit.dimension}")
    print(f"Matches POWER dimension: {simple_unit.dimension == power_dimension}")
    
    reduced_simple = Unit.reduce_unit(simple_unit)
    print(f"Reduced simple unit: {reduced_simple}")
    
    print("\n--- Testing A^2*V^2 reduction ---")
    # Test A^2*V^2 to see if it reduces to W^2
    squared_unit = Unit("A^2*V^2")
    print(f"Squared unit: {squared_unit}")
    print(f"Squared unit dimension: {squared_unit.dimension}")
    
    # Check if A^2*V^2 matches W^2 dimension
    power_squared = Unit("W^2")
    print(f"W^2 dimension: {power_squared.dimension}")
    print(f"A^2*V^2 matches W^2 dimension: {squared_unit.dimension == power_squared.dimension}")
    
    reduced_squared = Unit.reduce_unit(squared_unit)
    print(f"Reduced squared unit: {reduced_squared}")
    
    print("\n--- Reducing unit ---")
    reduced = Unit.reduce_unit(test_unit)
    print(f"Reduced unit: {reduced}")
    print(f"Reduced unit elements: {reduced.unit_elements}")
    
    # Print each element of reduced unit
    for subscript, elements in reduced.unit_elements.items():
        print(f"  Subscript '{subscript}':")
        for element in elements:
            print(f"    {element.unit_symbol.value.symbols[0]} (exponent: {element.exponent})")
    
    print(f"\nExpected: W_elec^2")
    print(f"Got: {reduced}")
    print(f"Match: {str(reduced) == 'W_elec^2'}")

if __name__ == "__main__":
    debug_reduction() 