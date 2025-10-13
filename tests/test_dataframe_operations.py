#!/usr/bin/env python3
"""
Test suite for UnitedDataframe operations.

This test suite systematically tests UnitedDataframe CRUD operations including:
- Column operations (add, remove, rename, set values)
- Row operations (add, insert, remove, set values, clear)
- Cell operations (get, set individual cells)
- Data retrieval (as arrays, numpy arrays, pandas series)
- Metadata access (units, column types, column information)
"""

import numpy as np
import pandas as pd
from pandas import Timestamp
from typing import Sequence, Optional, Mapping

from united_system import VALUE_TYPE, Dimension, SCALAR_TYPE, UnitedDataframe, DataframeColumnType, Unit, StringArray, RealUnitedScalar, RealUnitedArray, IntArray, ComplexArray

from .test_dataframe import TestColumnKey

class TestUnitedDataframeOperations:
    """Test class for UnitedDataframe operations."""

    def test_column_operations_comprehensive(self):
        """Test all column operations: add, remove, rename, set values."""
        print("\n🔧 Testing comprehensive column operations...")
        
        # Test 1: Create dataframe with initial data
        print("\n📊 Testing dataframe creation with data...")
        
        # Create dataframe with initial data using the constructor approach
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("sample_id"): (DataframeColumnType.STRING, None, ["A", "B", "C"]),
            TestColumnKey("temperature"): (DataframeColumnType.REAL_NUMBER_64, Unit("°C"), [20.0, 25.0, 30.0]),
            TestColumnKey("count"): (DataframeColumnType.INTEGER_64, None, [1, 2, 3])
        }
        
        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
        
        assert len(df.colkeys) == 3
        assert len(df) == 3
        print("✅ Created dataframe with 3 columns and 3 rows successfully")
        
        # Test 2: Column access and metadata
        print("\n📋 Testing column metadata access...")
        
        column_keys = df.colkeys
        assert len(column_keys) == 3
        print(f"✅ Column keys: {[str(key) for key in column_keys]}")
        
        # Test units
        temp_unit = df.units[TestColumnKey("temperature")]
        assert str(temp_unit) == "°C"
        print(f"✅ Temperature unit: {temp_unit}")
        
        # Test column types
        temp_type = df.coltypes[TestColumnKey("temperature")]
        assert temp_type == DataframeColumnType.REAL_NUMBER_64
        print(f"✅ Temperature type: {temp_type}")
        
        # Test columns with/without units
        with_units = df.column_get_with_units()
        without_units = df.column_get_without_units()
        assert len(with_units) == 1
        assert len(without_units) == 2
        print(f"✅ Columns with units: {[str(key) for key in with_units]}")
        print(f"✅ Columns without units: {[str(key) for key in without_units]}")
        
        # Test 3: Create a new dataframe with an additional column (instead of adding to existing)
        print("\n➕ Testing dataframe creation with additional column...")
        columns_with_bool: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("sample_id"): (DataframeColumnType.STRING, None, ["A", "B", "C"]),
            TestColumnKey("temperature"): (DataframeColumnType.REAL_NUMBER_64, Unit("°C"), [20.0, 25.0, 30.0]),
            TestColumnKey("count"): (DataframeColumnType.INTEGER_64, None, [1, 2, 3]),
            TestColumnKey("active"): (DataframeColumnType.BOOL, None, [True, False, True])
        }
        
        df_with_bool: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns_with_bool)
        
        assert len(df_with_bool.colkeys) == 4
        assert TestColumnKey("active") in df_with_bool.colkeys
        print("✅ Created dataframe with BOOL column successfully")
        
        # Use this dataframe for the rest of the tests
        df = df_with_bool
        
        # Test 4: Column rename
        print("\n🏷️ Testing column_rename...")
        df.column_rename(TestColumnKey("sample_id"), TestColumnKey("id"))
        column_keys = df.colkeys
        assert TestColumnKey("id") in column_keys
        assert TestColumnKey("sample_id") not in column_keys
        print("✅ Column renamed successfully")
        
        # Test 5: Column set values
        print("\n🔄 Testing column_set_values...")
        new_string_data = StringArray(np.array(["X", "Y", "Z"]))
        df.column_set_values(TestColumnKey("id"), new_string_data)
        
        # Verify the change
        retrieved_data = df.column_get_as_array(TestColumnKey("id"))
        assert isinstance(retrieved_data, StringArray)
        assert retrieved_data.canonical_np_array.tolist() == ["X", "Y", "Z"]
        print("✅ Column values updated successfully")
        
        # Test 6: Column removal
        print("\n🗑️ Testing column_remove...")
        df.column_remove(TestColumnKey("count"))
        assert len(df.colkeys) == 3  # Was 4, now 3 after removal
        assert TestColumnKey("count") not in df.colkeys
        print("✅ Column removed successfully")
        
        # Test 7: Error cases
        print("\n❌ Testing column operation error cases...")
        
        try:
            # Try to add existing column
            duplicate_data = StringArray(np.array(["D", "E", "F"]))
            df.column_add(TestColumnKey("id"), duplicate_data, DataframeColumnType.STRING, None)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✅ Correctly caught duplicate column error: {e}")
        
        try:
            # Try to remove non-existent column
            df.column_remove(TestColumnKey("nonexistent"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✅ Correctly caught missing column error: {e}")
        
        print("🎉 Column operations test completed successfully!")

    def test_row_operations_comprehensive(self):
        """Test all row operations: add, insert, remove, set values, clear."""
        print("\n🔧 Testing comprehensive row operations...")
        
        # Create initial dataframe with data
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("id"): (DataframeColumnType.STRING, None, ["A", "B"]),
            TestColumnKey("value"): (DataframeColumnType.REAL_NUMBER_64, Unit("m"), [10.0, 20.0])
        }
        
        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
        
        assert len(df) == 2
        print("✅ Initial dataframe: 2 rows, 2 columns")
        
        # Test 1: Add empty rows
        print("\n➕ Testing row_add_empty...")
        df.row_add_empty(2)
        assert len(df) == 4
        print("✅ Added 2 empty rows successfully")
        
        # Test 2: Add rows with values
        print("\n📝 Testing row_add_values...")
        new_values_1: Mapping[TestColumnKey, VALUE_TYPE|SCALAR_TYPE] = {
            TestColumnKey("id"): "E",
            TestColumnKey("value"): RealUnitedScalar(50.0, Unit("m"))
        }
        new_values_2: Mapping[TestColumnKey, VALUE_TYPE|SCALAR_TYPE] = {
            TestColumnKey("id"): "G",
            TestColumnKey("value"): RealUnitedScalar(70.0, Unit("m"))
        }
        df.row_add_items(new_values_1)
        df.row_add_items(new_values_2)
        assert len(df) == 6
        print("✅ Added 2 rows with values successfully")
        
        # Test 3: Insert empty rows
        print("\n⬇️ Testing row_insert_empty...")
        df.row_insert_empty(1)  # Insert 1 empty row at index 1
        assert len(df) == 7
        print("✅ Inserted 1 empty row at index 1 successfully")
        
        # Test 4: Insert rows with values
        print("\n📋 Testing row_insert_values...")
        insert_values: Mapping[TestColumnKey, VALUE_TYPE|SCALAR_TYPE] = {
            TestColumnKey("id"): "X",
            TestColumnKey("value"): RealUnitedScalar(35.0, Unit("m"))
        }
        df.row_insert_items(2, insert_values)
        assert len(df) == 8
        print("✅ Inserted 1 row with values at index 2 successfully")
        
        # Test 5: Set row values (replace)
        print("\n🔄 Testing row_set_values...")
        replace_values: Mapping[TestColumnKey, VALUE_TYPE|SCALAR_TYPE] = {
            TestColumnKey("id"): "Y",
            TestColumnKey("value"): RealUnitedScalar(99.0, Unit("m"))
        }
        df.row_set_items(0, replace_values)
        
        # Verify the change
        first_row = df.row_get_as_dict(0)
        id_value: SCALAR_TYPE = first_row[TestColumnKey("id")]
        assert id_value == "Y"
        print("✅ Row values replaced successfully")
        
        # Test 6: Clear specific rows
        print("\n🧹 Testing row_clear...")
        initial_rows = len(df)
        df.row_clear(1, 2)  # Clear 2 rows starting at index 1
        
        # Rows should still exist but be filled with NA
        assert len(df) == initial_rows
        print("✅ Cleared 2 rows successfully (set to NA)")
        
        # Test 7: Remove rows
        print("\n🗑️ Testing row_remove...")
        df.row_remove(1, 3)  # Remove rows 1-2 (stop exclusive)
        assert len(df) == initial_rows - 2
        print("✅ Removed 2 rows successfully")
        
        # Test 8: Clear all rows
        print("\n🧽 Testing row_clear_all...")
        df.row_clear_all()
        # Rows should still exist but be filled with NA
        current_rows = len(df)
        assert current_rows > 0
        print(f"✅ Cleared all {current_rows} rows successfully (set to NA)")
        
        # Test 9: Remove all rows
        print("\n🗑️ Testing row_remove_all...")
        df.row_remove_all()
        assert len(df) == 0
        print("✅ Removed all rows successfully")
        
        print("🎉 Row operations test completed successfully!")

    def test_row_find_index_by_item(self):
        """Test row_find_by_item across types, units, not-found, invalid, and read-only."""
        print("\n🔎 Testing row_find_by_item...")

        now1: Timestamp = Timestamp("2024-01-01") # type: ignore
        now2: Timestamp = Timestamp("2024-01-02") # type: ignore

        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("strings"): (DataframeColumnType.STRING, None, ["A", "B", "A", "C"]),
            TestColumnKey("reals"): (DataframeColumnType.REAL_NUMBER_64, Unit("m"), [1.0, 2.0, 3.0, 1.0]),
            TestColumnKey("integers"): (DataframeColumnType.INTEGER_64, None, [10, 20, 10, 30]),
            TestColumnKey("active"): (DataframeColumnType.BOOL, None, [True, False, True, False]),
            TestColumnKey("timestamp"): (DataframeColumnType.TIMESTAMP, None, [now1, now2, now1, now2]),
        }

        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)

        # String matches (duplicates)
        print("\n🧵 String matches...")
        string_idx = df.row_find_index_by_item(TestColumnKey("strings"), "A")
        assert string_idx == [0, 2]
        print(f"✅ Found 'A' at rows: {string_idx}")

        # Real with unit: united scalar in same unit
        print("\n📏 Real with unit (same unit)...")
        real_idx_same = df.row_find_index_by_item(TestColumnKey("reals"), RealUnitedScalar(1.0, Unit("m")))
        assert real_idx_same == [0, 3]
        print(f"✅ Found 1.0 m at rows: {real_idx_same}")

        # Real with unit: united scalar with convertible unit
        print("\n📐 Real with unit (convertible unit)...")
        real_idx_conv = df.row_find_index_by_item(TestColumnKey("reals"), RealUnitedScalar(0.001, Unit("km")))
        assert real_idx_conv == [0, 3]
        print(f"✅ Found 0.001 km (==1.0 m) at rows: {real_idx_conv}")

        # Real primitive float (no unit scalar) should still match
        print("\n🔢 Real primitive matches...")
        real_idx_prim = df.row_find_index_by_item(TestColumnKey("reals"), 1.0)
        assert real_idx_prim == [0, 3]
        print(f"✅ Found primitive 1.0 at rows: {real_idx_prim}")

        # Integer matches
        print("\n🔢 Integer matches...")
        int_idx = df.row_find_index_by_item(TestColumnKey("integers"), 10)
        assert int_idx == [0, 2]
        print(f"✅ Found 10 at rows: {int_idx}")

        # Boolean matches
        print("\n✅ Boolean matches...")
        bool_idx = df.row_find_index_by_item(TestColumnKey("active"), True)
        assert bool_idx == [0, 2]
        print(f"✅ Found True at rows: {bool_idx}")

        # Timestamp matches
        print("\n⏱️ Timestamp matches...")
        ts_idx = df.row_find_index_by_item(TestColumnKey("timestamp"), now1)
        assert ts_idx == [0, 2]
        print(f"✅ Found {now1} at rows: {ts_idx}")

        # Not found
        print("\n❓ Not found case...")
        none_idx = df.row_find_index_by_item(TestColumnKey("strings"), "Z")
        assert none_idx == []
        print("✅ No matches returned correctly")

        # Invalid value type should raise
        print("\n❌ Invalid type handling...")
        try:
            df.row_find_index_by_item(TestColumnKey("strings"), {"bad": 1})  # type: ignore[arg-type]
            assert False, "Should have raised ValueError for invalid value type"
        except ValueError as e:
            assert "Invalid value type" in str(e)
            print("✅ Correctly raised ValueError for invalid value type")

        # Read-only mode should still allow searching
        print("\n🔒 Read-only mode search...")
        ro_df = df.copy()
        ro_df.set_read_only(True)
        ro_idx = ro_df.row_find_index_by_item(TestColumnKey("strings"), "A")
        assert ro_idx == [0, 2]
        print("✅ Search works in read-only mode")

        print("🎉 row_find_by_item test completed successfully!")

    def test_cell_operations_comprehensive(self):
        """Test individual cell operations: get and set cell values."""
        print("\n🔧 Testing comprehensive cell operations...")
        
        # Create dataframe with mixed data types
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("text"): (DataframeColumnType.STRING, None, ["Hello", "World", "Test"]),
            TestColumnKey("length"): (DataframeColumnType.REAL_NUMBER_64, Unit("m"), [1.5, 2.5, 3.5]),
            TestColumnKey("count"): (DataframeColumnType.INTEGER_64, None, [10, 20, 30]),
            TestColumnKey("active"): (DataframeColumnType.BOOL, None, [True, False, True])
        }
        
        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
        
        print("✅ Created test dataframe: 3 rows, 4 columns")
        
        # Test 1: Get cell values
        print("\n📖 Testing cell_get_value...")
        
        text_value = df.cell_get_value(0, TestColumnKey("text"))
        assert text_value == "Hello"
        print(f"✅ Text cell [0]: {text_value}")
        
        length_value = df.cell_get_scalar(1, TestColumnKey("length"))
        assert length_value == RealUnitedScalar(2.5, Unit("m"))
        print(f"✅ Length cell [1]: {length_value}")
        
        count_value = df.cell_get_value(2, TestColumnKey("count"))
        assert count_value == 30
        print(f"✅ Count cell [2]: {count_value}")
        
        bool_value = df.cell_get_value(0, TestColumnKey("active"))
        assert bool_value == True
        print(f"✅ Boolean cell [0]: {bool_value}")
        
        # Test 2: Set cell values
        print("\n✏️ Testing cell_set_value...")
        
        # Set string value
        new_text = "Modified"  # Use raw string directly
        df.cell_set_value(0, TestColumnKey("text"), new_text)
        
        # Verify change
        updated_text = df.cell_get_value(0, TestColumnKey("text"))
        assert updated_text == "Modified"
        print("✅ Updated text cell successfully")
        
        # Set numeric value with unit
        new_length: SCALAR_TYPE = RealUnitedScalar(4.5, Unit("m"))
        df.cell_set_scalar(1, TestColumnKey("length"), new_length)
        
        # Verify change
        updated_length = df.cell_get_scalar(1, TestColumnKey("length"))
        assert updated_length == RealUnitedScalar(4.5, Unit("m"))
        print("✅ Updated length cell successfully")
        
        # Set integer value
        new_count = 99  # Use raw integer directly
        df.cell_set_value(2, TestColumnKey("count"), new_count)
        
        # Verify change
        updated_count = df.cell_get_value(2, TestColumnKey("count"))
        assert updated_count == 99
        print("✅ Updated count cell successfully")
        
        # Set boolean value
        new_bool = False  # Use raw boolean directly
        df.cell_set_value(0, TestColumnKey("active"), new_bool)
        
        # Verify change
        updated_bool = df.cell_get_value(0, TestColumnKey("active"))
        assert updated_bool == False
        print("✅ Updated boolean cell successfully")
        
        # Test 3: Error cases
        print("\n❌ Testing cell operation error cases...")
        
        try:
            # Try to access out of bounds row
            df.cell_get_value(10, TestColumnKey("text"))
            assert False, "Should have raised IndexError"
        except (IndexError, ValueError) as e:
            print(f"✅ Correctly caught out of bounds error: {type(e).__name__}")
        
        try:
            # Try to access non-existent column
            df.cell_get_value(0, TestColumnKey("nonexistent"))
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✅ Correctly caught missing column error: {e}")
        
        print("🎉 Cell operations test completed successfully!")

    def test_data_retrieval_comprehensive(self):
        """Test data retrieval in different formats: arrays, numpy arrays, pandas series."""
        print("\n🔧 Testing comprehensive data retrieval...")
        
        # Create test dataframe
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("strings"): (DataframeColumnType.STRING, None, ["A", "B", "C", "D"]),
            TestColumnKey("reals"): (DataframeColumnType.REAL_NUMBER_64, Unit("kg"), [1.1, 2.2, 3.3, 4.4]),
            TestColumnKey("integers"): (DataframeColumnType.INTEGER_64, None, [10, 20, 30, 40]),
            TestColumnKey("complex_vals"): (DataframeColumnType.COMPLEX_128, None, [1+2j, 3+4j, 5+6j, 7+8j])
        }
        
        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
        
        print("✅ Created test dataframe: 4 rows, 4 columns with mixed types")
        
        # Test 1: Get as UnitedSystem arrays
        print("\n📊 Testing column_get_as_array...")
        
        string_array = df.column_get_as_array(TestColumnKey("strings"))
        assert isinstance(string_array, StringArray)
        assert string_array.canonical_np_array.tolist() == ["A", "B", "C", "D"]
        print(f"✅ String array: {string_array.canonical_np_array.tolist()}")
        
        real_array = df.column_get_as_array(TestColumnKey("reals"))
        assert isinstance(real_array, RealUnitedArray)
        assert real_array.canonical_np_array.tolist() == [1.1, 2.2, 3.3, 4.4]
        assert str(real_array.unit) == "kg"
        print(f"✅ Real array: {real_array.canonical_np_array.tolist()} {real_array.unit}")
        
        int_array = df.column_get_as_array(TestColumnKey("integers"))
        assert isinstance(int_array, IntArray)
        assert int_array.canonical_np_array.tolist() == [10, 20, 30, 40]
        print(f"✅ Integer array: {int_array.canonical_np_array.tolist()}")
        
        complex_array = df.column_get_as_array(TestColumnKey("complex_vals"))
        assert isinstance(complex_array, ComplexArray)
        expected_complex = [1+2j, 3+4j, 5+6j, 7+8j]
        assert complex_array.canonical_np_array.tolist() == expected_complex
        print(f"✅ Complex array: {complex_array.canonical_np_array.tolist()}")
        
        # Test 2: Get as NumPy arrays
        print("\n🔢 Testing column_get_as_numpy_array...")
        
        string_numpy = df.column_get_as_numpy_array(TestColumnKey("strings"))
        assert isinstance(string_numpy, np.ndarray)
        assert list(string_numpy) == ["A", "B", "C", "D"]
        print(f"✅ String NumPy array: {list(string_numpy)}")
        
        real_numpy = df.column_get_as_numpy_array(TestColumnKey("reals"))
        assert isinstance(real_numpy, np.ndarray)
        assert np.allclose(real_numpy, [1.1, 2.2, 3.3, 4.4])
        print(f"✅ Real NumPy array: {list(real_numpy)}")
        
        int_numpy = df.column_get_as_numpy_array(TestColumnKey("integers"))
        assert isinstance(int_numpy, np.ndarray)
        assert list(int_numpy) == [10, 20, 30, 40]
        print(f"✅ Integer NumPy array: {list(int_numpy)}")
        
        # Test 3: Get as Pandas Series
        print("\n🐼 Testing column_get_as_pd_series...")
        
        string_series: pd.Series[str] = df.column_get_as_pd_series(TestColumnKey("strings")) # type: ignore
        assert isinstance(string_series, pd.Series)
        assert list(string_series) == ["A", "B", "C", "D"]
        print(f"✅ String Pandas Series: {list(string_series)}")
        
        real_series: pd.Series[float] = df.column_get_as_pd_series(TestColumnKey("reals")) # type: ignore
        assert isinstance(real_series, pd.Series)
        assert np.allclose(real_series, [1.1, 2.2, 3.3, 4.4])
        print(f"✅ Real Pandas Series: {list(real_series)}")
        
        # Test 4: Sliced data retrieval
        print("\n✂️ Testing sliced data retrieval...")
        
        # Get slice of data as array
        sliced_reals = df.column_get_as_array(TestColumnKey("reals"), expected_column_type=None, slice=slice(1, 3))
        assert isinstance(sliced_reals, RealUnitedArray)
        assert sliced_reals.canonical_np_array.tolist() == [2.2, 3.3]
        print(f"✅ Sliced real array [1:3]: {sliced_reals.canonical_np_array.tolist()}")
        
        # Get slice as numpy array
        sliced_numpy = df.column_get_as_numpy_array(TestColumnKey("integers"), slice=slice(0, 2))
        assert list(sliced_numpy) == [10, 20]
        print(f"✅ Sliced integer NumPy array [0:2]: {list(sliced_numpy)}")
        
        # Get slice as pandas series
        sliced_series: pd.Series[str] = df.column_get_as_pd_series(TestColumnKey("strings"), slice=slice(2, 4)) # type: ignore
        assert list(sliced_series) == ["C", "D"]
        print(f"✅ Sliced string Pandas Series [2:4]: {list(sliced_series)}")
        
        # Test 5: Type-specific retrieval
        print("\n🎯 Testing type-specific retrieval...")
        
        # Specify expected type
        typed_real_array = df.column_get_as_array(TestColumnKey("reals"), expected_column_type=RealUnitedArray)
        assert isinstance(typed_real_array, RealUnitedArray)
        print("✅ Type-specific retrieval works")
        
        try:
            # Try wrong type
            df.column_get_as_array(TestColumnKey("strings"), expected_column_type=RealUnitedArray)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            print(f"✅ Correctly caught type mismatch error: {e}")
        
        print("🎉 Data retrieval test completed successfully!")

    def test_metadata_access_comprehensive(self):
        """Test comprehensive metadata access: units, types, column information."""
        print("\n🔧 Testing comprehensive metadata access...")
        
        # Create test dataframe with diverse metadata
        now: Timestamp = Timestamp.now()
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("mass"): (DataframeColumnType.REAL_NUMBER_64, Unit("kg"), [1.0, 2.0]),
            TestColumnKey("length"): (DataframeColumnType.REAL_NUMBER_32, Unit("cm"), [10.0, 20.0]),
            TestColumnKey("voltage"): (DataframeColumnType.COMPLEX_NUMBER_128, Unit("V"), [1+0j, 2+0j]),
            TestColumnKey("name"): (DataframeColumnType.STRING, None, ["A", "B"]),
            TestColumnKey("count"): (DataframeColumnType.INTEGER_32, None, [5, 10]),
            TestColumnKey("active"): (DataframeColumnType.BOOL, None, [True, False]),
            TestColumnKey("timestamp"): (DataframeColumnType.TIMESTAMP, None, [now, now])
        }
        
        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
        
        print("✅ Created test dataframe: 2 rows, 7 columns with diverse metadata")
        
        # Test 1: Basic metadata
        print("\n📋 Testing basic metadata access...")
        
        assert len(df) == 2
        assert len(df.colkeys) == 7
        print(f"✅ Dataframe dimensions: {len(df)} rows, {len(df.colkeys)} columns")
        
        column_keys = df.colkeys
        assert len(column_keys) == 7
        expected_keys = ["mass", "length", "voltage", "name", "count", "active", "timestamp"]
        assert all(TestColumnKey(key) in column_keys for key in expected_keys)
        print(f"✅ Column keys: {[str(key) for key in column_keys]}")
        
        # Test 2: Units access
        print("\n📏 Testing units access...")
        
        mass_unit = df.units[TestColumnKey("mass")]
        assert str(mass_unit) == "kg"
        print(f"✅ Mass unit: {mass_unit}")
        
        length_unit = df.units[TestColumnKey("length")]
        assert str(length_unit) == "cm"
        print(f"✅ Length unit: {length_unit}")
        
        voltage_unit = df.units[TestColumnKey("voltage")]
        assert str(voltage_unit) == "V"
        print(f"✅ Voltage unit: {voltage_unit}")
        
        name_unit = df.units[TestColumnKey("name")]
        assert name_unit is None or str(name_unit) == "None"
        print(f"✅ Name unit: {name_unit}")
        
        # Test 3: Column types access
        print("\n🔤 Testing column types access...")
        
        mass_type = df.coltypes[TestColumnKey("mass")]
        assert mass_type == DataframeColumnType.REAL_NUMBER_64
        print(f"✅ Mass type: {mass_type}")
        
        length_type = df.coltypes[TestColumnKey("length")]
        assert length_type == DataframeColumnType.REAL_NUMBER_32
        print(f"✅ Length type: {length_type}")
        
        voltage_type = df.coltypes[TestColumnKey("voltage")]
        assert voltage_type == DataframeColumnType.COMPLEX_NUMBER_128
        print(f"✅ Voltage type: {voltage_type}")
        
        name_type = df.coltypes[TestColumnKey("name")]
        assert name_type == DataframeColumnType.STRING
        print(f"✅ Name type: {name_type}")
        
        count_type = df.coltypes[TestColumnKey("count")]
        assert count_type == DataframeColumnType.INTEGER_32
        print(f"✅ Count type: {count_type}")
        
        active_type = df.coltypes[TestColumnKey("active")]
        assert active_type == DataframeColumnType.BOOL
        print(f"✅ Active type: {active_type}")
        
        timestamp_type = df.coltypes[TestColumnKey("timestamp")]
        assert timestamp_type == DataframeColumnType.TIMESTAMP
        print(f"✅ Timestamp type: {timestamp_type}")
        
        # Test 4: Columns categorized by units
        print("\n🎯 Testing unit categorization...")
        
        with_units = df.column_get_with_units()
        without_units = df.column_get_without_units()
        
        expected_with_units = [TestColumnKey("mass"), TestColumnKey("length"), TestColumnKey("voltage")]
        expected_without_units = [TestColumnKey("name"), TestColumnKey("count"), TestColumnKey("active"), TestColumnKey("timestamp")]
        
        assert len(with_units) == 3
        assert len(without_units) == 4
        
        for key in expected_with_units:
            assert key in with_units
        for key in expected_without_units:
            assert key in without_units
        
        print(f"✅ Columns with units: {[str(key) for key in with_units]}")
        print(f"✅ Columns without units: {[str(key) for key in without_units]}")
        
        # Test 5: Dimensions access
        print("\n📐 Testing dimensions access...")
        
        mass_dimension = df.dim_get_dimension(TestColumnKey("mass"))
        assert mass_dimension is not None
        print(f"✅ Mass dimension: {mass_dimension}")
        
        length_dimension = df.dim_get_dimension(TestColumnKey("length"))
        assert length_dimension is not None
        print(f"✅ Length dimension: {length_dimension}")
        
        # Check if name column has dimension first
        if df.dim_has_dimension(TestColumnKey("name")):
            name_dimension = df.dim_get_dimension(TestColumnKey("name"))
        else:
            name_dimension = None
        assert name_dimension is None
        print(f"✅ Name dimension: {name_dimension}")
        
        # Test 6: Iterator access
        print("\n🔄 Testing iterator access...")
        
        # Test unit iterator
        units_list = list(df.iter_units())
        assert len(units_list) == 7
        print(f"✅ Units iterator: {len(units_list)} units")
        
        # Test dimensions iterator
        dimensions_list = list(df.iter_dimensions())
        assert len(dimensions_list) == 7
        print(f"✅ Dimensions iterator: {len(dimensions_list)} dimensions")
        
        # Test column iterator
        columns_list = list(df.iter_columns())
        assert len(columns_list) == 7
        print(f"✅ Columns iterator: {len(columns_list)} column accessors")
        
        # Test row iterator
        rows_list = list(df.iter_rows())
        assert len(rows_list) == 2
        print(f"✅ Rows iterator: {len(rows_list)} row accessors")
        
        print("🎉 Metadata access test completed successfully!")

    def test_read_only_mode_comprehensive(self):
        """Test operations in read-only mode to ensure proper error handling."""
        print("\n🔧 Testing read-only mode comprehensive...")
        
        # Create test dataframe
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {TestColumnKey("test"): (DataframeColumnType.STRING, None, ["A", "B"])}
        
        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
        
        # Convert to read-only
        read_only_df = df.copy()
        read_only_df.set_read_only(True)
        
        print("✅ Created read-only dataframe copy")
        
        # Test that read operations still work
        print("\n📖 Testing read operations on read-only dataframe...")
        
        assert len(read_only_df) == 2
        assert len(read_only_df.colkeys) == 1
        
        data = read_only_df.column_get_as_array(TestColumnKey("test"))
        assert data.canonical_np_array.tolist() == ["A", "B"]
        
        cell_value: VALUE_TYPE = read_only_df.cell_get_value(0, TestColumnKey("test"))
        assert cell_value == "A"
        
        print("✅ Read operations work correctly in read-only mode")
        
        # Test that write operations fail
        print("\n❌ Testing write operations on read-only dataframe...")
        
        operations_to_test = [
            ("column_add", lambda: read_only_df.column_add(TestColumnKey("new"), StringArray(np.array(["X", "Y"])), DataframeColumnType.STRING, None)),
            ("column_remove", lambda: read_only_df.column_remove(TestColumnKey("test"))),
            ("column_set_values", lambda: read_only_df.column_set_values(TestColumnKey("test"), StringArray(np.array(["X", "Y"])))),
            ("row_add_empty", lambda: read_only_df.row_add_empty(1)),
            ("row_remove", lambda: read_only_df.row_remove(0, 1)),
            ("cell_set_value", lambda: read_only_df.cell_set_value(0, TestColumnKey("test"), "X")),
        ]
        
        for op_name, op_func in operations_to_test:
            try:
                op_func()
                assert False, f"{op_name} should have raised ValueError in read-only mode"
            except ValueError as e:
                assert "read-only" in str(e).lower()
                print(f"✅ {op_name} correctly blocked in read-only mode")
        
        print("🎉 Read-only mode test completed successfully!")

    def test_dataframe_get_of_unique_rows(self):
        """Test deduplicating rows using dataframe_get_of_unique_rows."""
        print("\n🔁 Testing dataframe_get_of_unique_rows...")

        # Create dataframe with 4 columns and 8 rows with diverse duplicates
        columns: Mapping[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
            TestColumnKey("category"): (DataframeColumnType.STRING, None, ["A", "A", "B", "B", "C", "C", "A", "B"]),
            TestColumnKey("value"): (DataframeColumnType.INTEGER_64, None, [10, 10, 20, 20, 30, 30, 10, 20]),
            TestColumnKey("weight"): (DataframeColumnType.REAL_NUMBER_64, Unit("kg"), [1.5, 1.5, 2.5, 2.5, 3.5, 3.5, 1.5, 2.5]),
            TestColumnKey("active"): (DataframeColumnType.BOOL, None, [True, True, False, False, True, True, True, False]),
        }

        df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)

        # Case 1: Unique by all columns (default)
        print("\n✅ Unique by all columns (default)...")
        unique_all = df.dataframe_get_of_unique_rows()
        assert len(unique_all) == 3  # (A,10,1.5,True), (B,20,2.5,False), (C,30,3.5,True) - all others are duplicates
        assert unique_all.row_get_as_dict(0)[TestColumnKey("category")] == "A"
        assert unique_all.row_get_as_dict(0)[TestColumnKey("value")] == 10
        # Check that weight is a float value (the unit is preserved in the dataframe structure)
        weight_value = unique_all.row_get_as_dict(0)[TestColumnKey("weight")]
        assert weight_value == 1.5  # The value is stored as float, unit is in column metadata
        assert unique_all.row_get_as_dict(0)[TestColumnKey("active")] == True
        assert unique_all.row_get_as_dict(1)[TestColumnKey("category")] == "B"
        assert unique_all.row_get_as_dict(1)[TestColumnKey("value")] == 20
        assert unique_all.row_get_as_dict(1)[TestColumnKey("weight")] == 2.5
        assert unique_all.row_get_as_dict(1)[TestColumnKey("active")] == False
        assert unique_all.row_get_as_dict(2)[TestColumnKey("category")] == "C"
        assert unique_all.row_get_as_dict(2)[TestColumnKey("value")] == 30
        assert unique_all.row_get_as_dict(2)[TestColumnKey("weight")] == 3.5
        assert unique_all.row_get_as_dict(2)[TestColumnKey("active")] == True
        print("✅ Default unique keeps first occurrences across all columns")

        # Case 2: Unique by subset of columns (category only)
        print("\n✅ Unique by subset of columns (category only)...")
        unique_category = df.dataframe_get_of_unique_rows(TestColumnKey("category"))
        assert len(unique_category) == 3  # First A, first B, first C
        assert unique_category.row_get_as_dict(0)[TestColumnKey("category")] == "A"
        assert unique_category.row_get_as_dict(1)[TestColumnKey("category")] == "B"
        assert unique_category.row_get_as_dict(2)[TestColumnKey("category")] == "C"
        # Should keep first occurrences, thus corresponding values are the first seen per category
        assert unique_category.row_get_as_dict(0)[TestColumnKey("value")] == 10
        assert unique_category.row_get_as_dict(1)[TestColumnKey("value")] == 20
        assert unique_category.row_get_as_dict(2)[TestColumnKey("value")] == 30
        print("✅ Subset unique works and preserves row order (first occurrences)")

        # Case 3: Unique by multiple columns (category + value)
        print("\n✅ Unique by multiple columns (category + value)...")
        unique_multi = df.dataframe_get_of_unique_rows(TestColumnKey("category"), TestColumnKey("value"))
        assert len(unique_multi) == 3  # (A,10), (B,20), (C,30)
        assert unique_multi.row_get_as_dict(0)[TestColumnKey("category")] == "A"
        assert unique_multi.row_get_as_dict(0)[TestColumnKey("value")] == 10
        assert unique_multi.row_get_as_dict(1)[TestColumnKey("category")] == "B"
        assert unique_multi.row_get_as_dict(1)[TestColumnKey("value")] == 20
        assert unique_multi.row_get_as_dict(2)[TestColumnKey("category")] == "C"
        assert unique_multi.row_get_as_dict(2)[TestColumnKey("value")] == 30
        print("✅ Multi-column unique works correctly")

        # Case 4: Unique by numeric column with units
        print("\n✅ Unique by numeric column with units (weight)...")
        unique_weight = df.dataframe_get_of_unique_rows(TestColumnKey("weight"))
        assert len(unique_weight) == 3  # 1.5kg, 2.5kg, 3.5kg
        assert unique_weight.row_get_as_dict(0)[TestColumnKey("weight")] == 1.5
        assert unique_weight.row_get_as_dict(1)[TestColumnKey("weight")] == 2.5
        assert unique_weight.row_get_as_dict(2)[TestColumnKey("weight")] == 3.5
        print("✅ Unit-aware unique works correctly")

        # Case 5: Invalid column key should raise
        print("\n❌ Testing invalid column key handling...")
        try:
            _ = df.dataframe_get_of_unique_rows(TestColumnKey("nonexistent"))
            assert False, "Should have raised ValueError for nonexistent column key"
        except ValueError as e:
            assert "does not exist" in str(e)
            print(f"✅ Correctly raised error for invalid key: {e}")

        print("🎉 dataframe_get_of_unique_rows test completed successfully!")
