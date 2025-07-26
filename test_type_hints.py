#!/usr/bin/env python3

"""Test file to verify type hints work correctly for cell_get_value."""

import sys
sys.path.insert(0, 'src')

from united_system.united_dataframe import UnitedDataframe
from united_system.real_united_scalar import RealUnitedScalar
from united_system.unit import Unit
from united_system.column_type import ColumnType
from united_system.utils.dataframe.mixins.constructor_mixin import SimpleInternalDataFrameNameFormatter

def test_type_hints():
    """Test that type hints work correctly for cell_get_value."""
    print("🧪 Testing type hints for cell_get_value...")
    
    # Create a simple dataframe
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
    
    df = UnitedDataframe.create_dataframe_from_data(
        arrays=arrays,
        column_types=column_types,
        column_units_or_dimensions=column_units,
        internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
    )
    
    print("✅ Created test dataframe")
    
    # Test 1: Without type hint (should return SCALAR_TYPE)
    value1 = df.cell_get_value(1, "Length")
    print(f"✅ Without type hint: {type(value1).__name__} = {value1}")
    
    # Test 2: With type hint (should return RealUnitedScalar)
    value2: RealUnitedScalar = df.cell_get_value(1, "Length", RealUnitedScalar)
    print(f"✅ With type hint: {type(value2).__name__} = {value2}")
    
    # Test 3: Verify the type hint works
    assert isinstance(value2, RealUnitedScalar), f"Expected RealUnitedScalar, got {type(value2)}"
    print(f"✅ Type assertion passed: {value2.value()} {value2.unit}")
    
    # Test 4: Test string column
    name_value: str = df.cell_get_value(0, "Name", str)
    print(f"✅ String with type hint: {type(name_value).__name__} = {name_value}")
    
    print("🎉 All type hint tests passed!")

if __name__ == "__main__":
    test_type_hints() 