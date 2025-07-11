#!/usr/bin/env python3
"""
Test script for the new row operations methods in UnitedDataframe.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from united_system.united_dataframe import UnitedDataframe
from united_system.dataframe.column_type import ColumnType
from united_system.unit import Unit
from united_system.scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from united_system.dataframe.internal_dataframe_name_formatter import ColumnInformation, SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
import pandas as pd

def test_row_operations():
    """Test the new row operations methods."""
    print("Testing new row operations methods...")
    
    # Use Unit.parse_string to get SI units
    meter = Unit.parse_string('m')
    second = Unit.parse_string('s')
    kilogram = Unit.parse_string('kg')
    
    # Build column_information dict
    column_information = {
        "distance": ColumnInformation.create(meter, ColumnType.REAL_NUMBER_64, meter),
        "time": ColumnInformation.create(second, ColumnType.REAL_NUMBER_64, second),
        "mass": ColumnInformation.create(kilogram, ColumnType.REAL_NUMBER_64, kilogram)
    }
    # Create empty DataFrame with correct columns
    empty_df = pd.DataFrame({
        SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER.create_internal_dataframe_column_name(key, info): []
        for key, info in column_information.items()
    })
    # Create UnitedDataframe
    df = UnitedDataframe.from_dataframe_and_column_information_list(
        empty_df,
        column_information,
        SIMPLE_INTERNAL_DATAFRAME_NAME_FORMATTER
    )
    
    print(f"Initial dataframe shape: {df.shape}")
    print(f"Initial columns: {df.column_keys}")
    
    # Test 1: Add empty rows
    print("\n--- Test 1: Add empty rows ---")
    df.rows_add_empty(3)
    print(f"After adding 3 empty rows: {df.shape}")
    
    # Test 2: Add rows with values
    print("\n--- Test 2: Add rows with values ---")
    values = {
        "distance": [RealUnitedScalar(10.0, meter), RealUnitedScalar(20.0, meter)],
        "time": [RealUnitedScalar(5.0, second), RealUnitedScalar(10.0, second)],
        "mass": [RealUnitedScalar(1.0, kilogram), RealUnitedScalar(2.0, kilogram)]
    }
    df.rows_add_with_values(values)
    print(f"After adding 2 rows with values: {df.shape}")
    
    # Test 3: Insert empty rows
    print("\n--- Test 3: Insert empty rows ---")
    df.rows_insert_empty(1, 2)
    print(f"After inserting 2 empty rows at index 1: {df.shape}")
    
    # Test 4: Insert rows with values
    print("\n--- Test 4: Insert rows with values ---")
    insert_values = {
        "distance": [RealUnitedScalar(15.0, meter)],
        "time": [RealUnitedScalar(7.5, second)],
        "mass": [RealUnitedScalar(1.5, kilogram)]
    }
    df.rows_insert_with_values(2, insert_values)
    print(f"After inserting 1 row with values at index 2: {df.shape}")
    
    # Test 5: Set values for existing rows
    print("\n--- Test 5: Set values for existing rows ---")
    set_values = {
        "distance": [RealUnitedScalar(25.0, meter), RealUnitedScalar(30.0, meter)],
        "time": [RealUnitedScalar(12.5, second), RealUnitedScalar(15.0, second)],
        "mass": [RealUnitedScalar(2.5, kilogram), RealUnitedScalar(3.0, kilogram)]
    }
    df.rows_set_with_values(slice(0, 2), set_values)
    print(f"After setting values for rows 0-1: {df.shape}")
    
    # Test 6: Clear some rows
    print("\n--- Test 6: Clear some rows ---")
    df.rows_clear(3, 2)
    print(f"After clearing 2 rows starting at index 3: {df.shape}")
    
    # Test 7: Remove some rows
    print("\n--- Test 7: Remove some rows ---")
    df.rows_remove(1, 3)
    print(f"After removing rows 1-2: {df.shape}")
    
    # Test 8: Clear all rows
    print("\n--- Test 8: Clear all rows ---")
    df.rows_clear_all()
    print(f"After clearing all rows: {df.shape}")
    print(f"Dataframe empty: {df.empty}")
    
    # Test 9: Add some rows back and test remove all
    print("\n--- Test 9: Add rows back and remove all ---")
    df.rows_add_with_values(values)
    print(f"After adding rows back: {df.shape}")
    df.rows_remove_all()
    print(f"After removing all rows: {df.shape}")
    print(f"Dataframe empty: {df.empty}")
    
    print("\n✅ All row operations tests completed successfully!")

if __name__ == "__main__":
    test_row_operations() 