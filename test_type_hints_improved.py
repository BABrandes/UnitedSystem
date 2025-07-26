#!/usr/bin/env python3

"""Improved test showing better type hints for cell_get_value."""

import sys
sys.path.insert(0, 'src')

from typing import Literal, Union
from united_system.united_dataframe import UnitedDataframe
from united_system.real_united_scalar import RealUnitedScalar
from united_system.unit import Unit
from united_system.column_type import ColumnType
from united_system.utils.dataframe.mixins.constructor_mixin import SimpleInternalDataFrameNameFormatter

# Define specific column key types for better type hints
LengthKey = Literal["Length"]
NameKey = Literal["Name"]
CountKey = Literal["Count"]

# Union of all possible column keys
ColumnKeys = Union[LengthKey, NameKey, CountKey]

def test_improved_type_hints():
    """Test improved type hints for cell_get_value."""
    print("🧪 Testing improved type hints for cell_get_value...")
    
    # Create a simple dataframe
    arrays = {
        "Length": [1.0, 2.0, 3.0],
        "Name": ["A", "B", "C"],
        "Count": [10, 20, 30]
    }
    column_types = {
        "Length": ColumnType.REAL_NUMBER_64,
        "Name": ColumnType.STRING,
        "Count": ColumnType.INTEGER_64
    }
    column_units = {
        "Length": Unit("m"),
        "Name": None,
        "Count": None
    }
    
    # Create dataframe with specific type annotation
    df: UnitedDataframe[ColumnKeys] = UnitedDataframe[ColumnKeys].create_dataframe_from_data(
        arrays=arrays,
        column_types=column_types,
        column_units_or_dimensions=column_units,
        internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
    )
    
    print("✅ Created test dataframe with specific type annotation")
    
    # Now the type checker knows about the specific column keys!
    
    # Test 1: Length column (should return RealUnitedScalar)
    length_value: RealUnitedScalar = df.cell_get_value(1, "Length", RealUnitedScalar)
    print(f"✅ Length with type hint: {type(length_value).__name__} = {length_value}")
    print(f"   Value: {length_value.value()} {length_value.unit}")
    
    # Test 2: Name column (should return str)
    name_value: str = df.cell_get_value(0, "Name", str)
    print(f"✅ Name with type hint: {type(name_value).__name__} = {name_value}")
    
    # Test 3: Count column (should return int)
    count_value: int = df.cell_get_value(2, "Count", int)
    print(f"✅ Count with type hint: {type(count_value).__name__} = {count_value}")
    
    # Test 4: Without type hints (returns SCALAR_TYPE)
    value1 = df.cell_get_value(1, "Length")
    print(f"✅ Without type hint: {type(value1).__name__} = {value1}")
    
    print("🎉 All improved type hint tests passed!")

def test_alternative_approach():
    """Test alternative approach using type annotations."""
    print("\n🧪 Testing alternative approach...")
    
    # Alternative: Use type annotation on the dataframe variable
    arrays = {
        "Length": [1.0, 2.0, 3.0],
        "Name": ["A", "B", "C"]
    }
    column_types = {
        "Length": ColumnType.REAL_NUMBER_64,
        "Name": ColumnType.STRING
    }
    column_units = {
        "Length": Unit("m"),
        "Name": None
    }
    
    # Type annotation helps the type checker
    df: UnitedDataframe[ColumnKeys] = UnitedDataframe[ColumnKeys].create_dataframe_from_data(
        arrays=arrays,
        column_types=column_types,
        column_units_or_dimensions=column_units,
        internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
    )
    
    # Even without explicit type annotation, this should work
    length_value: RealUnitedScalar = df.cell_get_value(1, "Length", RealUnitedScalar)
    print(f"✅ Alternative approach: {type(length_value).__name__} = {length_value}")
    
    print("🎉 Alternative approach test passed!")

if __name__ == "__main__":
    test_improved_type_hints()
    test_alternative_approach() 