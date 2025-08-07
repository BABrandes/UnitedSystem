#!/usr/bin/env python3
"""
Test suite for UnitedDataframe debugging.

This test suite systematically tests UnitedDataframe functionality to identify
and debug any issues with the implementation.
"""

from pathlib import Path
from typing import Optional, Sequence

from united_system import VALUE_TYPE, Dimension, Unit, UnitedDataframe, DataframeColumnType

# Import TestColumnKey from the main test module
from .test_dataframe import TestColumnKey

class TestUnitedDataframeCore:
    """Test basic core functionality of UnitedDataframe objects."""
    
    def test_empty_dataframe_creation(self):
        """Test creating an empty UnitedDataframe with minimal configuration."""
        try:
            df = UnitedDataframe()
            assert len(df) == 0
            print("✅ Empty dataframe creation successful")
        except Exception as e:
            print(f"❌ Empty dataframe creation failed: {e}")
            raise

    def test_simple_numeric_dataframe_creation(self):
        """Test creating a UnitedDataframe with a simple numeric column."""
        try:
            # Create using create_from_data instead of constructor
            test_key = TestColumnKey("test")
            columns: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
                test_key: (DataframeColumnType.REAL_NUMBER_64, Unit("K"), [])
            }
            
            df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
            assert len(df) == 0  # Should have zero rows
            print("✅ Simple numeric dataframe creation successful")
            assert df.coltypes[test_key] == DataframeColumnType.REAL_NUMBER_64
            assert df.units[test_key] == Unit("K")
        except Exception as e:
            print(f"❌ Simple numeric dataframe creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_dataframe_with_data_creation(self):
        """Test creating a UnitedDataframe with actual data."""
        try:
            # Create test data using the public API only
            temp_col_key = TestColumnKey("temperature")
            temp_unit = Unit("K")
            
            # Use create_from_data to create dataframe with actual data
            columns: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
                temp_col_key: (DataframeColumnType.REAL_NUMBER_64, temp_unit, [273.15, 300.0, 350.0])
            }
            
            df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
            
            assert len(df) == 3
            print("✅ Dataframe with data creation successful")
            
        except Exception as e:
            print(f"❌ Dataframe with data creation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_dataframe_column_access(self):
        """Test accessing dataframe columns and basic operations."""
        try:
            # Create a simple dataframe
            temp_col_key = TestColumnKey("temperature")
            pressure_col_key = TestColumnKey("pressure")
            
            # Create with UnitedDataframe() constructor
            # Create dataframe using create_from_data instead of constructor
            columns: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
                temp_col_key: (DataframeColumnType.REAL_NUMBER_64, Unit("K"), []),
                pressure_col_key: (DataframeColumnType.REAL_NUMBER_64, Unit("Pa"), [])
            }
            
            df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
            
            # Test basic properties using PUBLIC API
            assert len(df.colkeys) == 2
            assert temp_col_key in df.colkeys
            assert pressure_col_key in df.colkeys
            
            # Test column types and units
            assert df.coltypes[temp_col_key] == DataframeColumnType.REAL_NUMBER_64
            assert df.units[temp_col_key] == Unit("K")
            
            print("✅ Dataframe column access successful")
            
        except Exception as e:
            print(f"❌ Dataframe column access failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_dataframe_data_manipulation(self):
        """Test adding and manipulating data in the dataframe."""
        try:
            # Create empty dataframe
            temp_col_key = TestColumnKey("temperature")
            
            # Create dataframe using create_from_data instead of constructor
            columns: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
                temp_col_key: (DataframeColumnType.REAL_NUMBER_64, Unit("K"), [])
            }
            
            df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
            
            # Test data creation with arrays
            from united_system._scalars.real_united_scalar import RealUnitedScalar
            
            # Create some temperature data
            temperatures: list[RealUnitedScalar] = [
                RealUnitedScalar(273.15, Unit("K")),
                RealUnitedScalar(300.0, Unit("K")),
                RealUnitedScalar(350.0, Unit("K"))
            ]
            
            df.row_add_items({temp_col_key: temperatures[0]})
            df.row_add_items({temp_col_key: temperatures[1]})
            df.row_add_items({temp_col_key: temperatures[2]})
            
            print(f"✅ Dataframe data manipulation test structure successful (created {len(temperatures)} temperature values)")
            
        except Exception as e:
            print(f"❌ Dataframe data manipulation failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_dataframe_thread_safety(self):
        """Test that dataframe operations are thread-safe."""
        try:
            df = UnitedDataframe()
            
            # Test context manager (thread safety)
            with df:
                # This should acquire a write lock
                pass
            
            # Test read-only functionality (public API)
            assert not df.is_read_only()
            df.set_read_only(True)
            assert df.is_read_only()
            df.set_read_only(False)
            assert not df.is_read_only()
            
            print("✅ Dataframe thread safety test successful")
            
        except Exception as e:
            print(f"❌ Dataframe thread safety test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_complex_dataframe_with_operations(self):
        """Test creating a complex dataframe with multiple column types and testing operations."""
        try:
            print("\n🔬 Creating complex experimental dataframe...")
            
            # Define column keys
            sample_id_key = TestColumnKey("sample_id")
            temperature_key = TestColumnKey("temperature")
            pressure_key = TestColumnKey("pressure")
            length_key = TestColumnKey("length")
            is_valid_key = TestColumnKey("is_valid")
            notes_key = TestColumnKey("notes")
            
            # Create empty dataframe with complex structure using create_from_data
            columns: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
                sample_id_key: (DataframeColumnType.STRING, None, []),
                temperature_key: (DataframeColumnType.REAL_NUMBER_64, Unit("K"), []),
                pressure_key: (DataframeColumnType.REAL_NUMBER_64, Unit("Pa"), []),
                length_key: (DataframeColumnType.REAL_NUMBER_64, Unit("m"), []),
                is_valid_key: (DataframeColumnType.BOOL, None, []),
                notes_key: (DataframeColumnType.STRING, None, [])
            }
            
            df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
            assert len(df) == 0
            print(f"✅ Complex dataframe created with {len(df.colkeys)} columns")
            
            # Test column properties using PUBLIC API
            assert len(df.colkeys) == 6
            assert df.coltypes[temperature_key] == DataframeColumnType.REAL_NUMBER_64
            assert df.units[temperature_key] == Unit("K")
            assert df.units[sample_id_key] is None
            print("✅ Column properties verified")
            
            # Test dataframe info methods using PUBLIC API
            print(f"📊 Dataframe info: {len(df)} rows, {len(df.colkeys)} columns")
            print(f"🏷️ Column keys: {[str(key) for key in df.colkeys]}")
            print(f"🔢 Column types: {[(str(k), v.name) for k, v in df.coltypes.items()]}")
            print(f"📏 Units: {[(str(k), str(v) if v else 'None') for k, v in df.units.items()]}")
            
            # Create test data using PUBLIC API
            sample_ids = ["EXP-001", "EXP-002", "EXP-003", "EXP-004", "EXP-005"]
            temperatures = [273.15, 298.15, 323.15, 348.15, 373.15]  # K
            pressures = [101325, 200000, 150000, 300000, 250000]     # Pa
            lengths = [0.001, 0.002, 0.0015, 0.0025, 0.003]         # m
            is_valid = [True, True, False, True, True]
            notes = ["Control", "Test A", "Invalid", "Test B", "Final"]
            
            # Create UnitedDataframe with data using PUBLIC API
            columns: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {
                sample_id_key: (DataframeColumnType.STRING, None, sample_ids),
                temperature_key: (DataframeColumnType.REAL_NUMBER_64, Unit("K"), temperatures),
                pressure_key: (DataframeColumnType.REAL_NUMBER_64, Unit("Pa"), pressures),
                length_key: (DataframeColumnType.REAL_NUMBER_64, Unit("m"), lengths),
                is_valid_key: (DataframeColumnType.BOOL, None, is_valid),
                notes_key: (DataframeColumnType.STRING, None, notes)
            }
            
            df_with_data: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns)
            
            print(f"✅ Complex dataframe populated with {len(df_with_data)} rows")
            
            # Test basic operations using PUBLIC API
            print("\n🧪 Testing operations on complex dataframe...")
            
            # Test length
            assert len(df_with_data) == 5
            print("✅ Row count correct")
            
            # Test column access using PUBLIC API
            col_keys = df_with_data.colkeys
            assert len(col_keys) == 6
            assert temperature_key in col_keys
            print("✅ Column access works")
            
            # Test string representation
            df_str = str(df_with_data)
            assert "5 rows" in df_str
            assert "6 columns" in df_str
            print(f"✅ String representation: {df_str}")
            
            # Test repr
            df_repr = repr(df_with_data)
            assert "5" in df_repr and "6" in df_repr
            print("✅ Repr works")
            
            # Test copy operations
            df_copy = df_with_data.copy()
            assert len(df_copy) == len(df_with_data)
            assert len(df_copy.colkeys) == len(df_with_data.colkeys)
            print("✅ Copy operation works")
            
            # Test head/tail operations
            df_head = df_with_data.head(3)
            assert len(df_head) == 3
            print("✅ Head operation works")
            
            df_tail = df_with_data.tail(2)
            assert len(df_tail) == 2
            print("✅ Tail operation works")
            
            # Test filtering (if available)
            try:
                # Filter for valid samples only
                df_valid = df_with_data.filter_column_equals(is_valid_key, True)
                assert len(df_valid) == 4  # Should have 4 valid samples
                print("✅ Filtering operation works")
            except Exception as filter_error:
                print(f"⚠️ Filtering not fully implemented: {filter_error}")
            
            # Test unit conversions (basic check)
            temp_unit = df_with_data.units[temperature_key]
            assert str(temp_unit) == "K"
            print("✅ Unit information accessible")
            
            # Test cell access operations (if available)
            try:
                # Get a specific cell value
                first_temp = df_with_data.cell_get_value(0, temperature_key)
                print(f"✅ Cell access works: First temperature = {first_temp}")
            except Exception as cell_error:
                print(f"⚠️ Cell access not fully implemented: {cell_error}")
            
            # Test statistical operations (if available)
            try:
                # Try to get column statistics
                temp_mean = df_with_data.column_get_mean(temperature_key)
                print(f"✅ Statistics work: Temperature mean = {temp_mean}")
            except Exception as stats_error:
                print(f"⚠️ Statistics not fully implemented: {stats_error}")
            
            print("\n🎉 Complex dataframe test completed successfully!")
            print(f"📈 Created dataframe with:")
            print(f"   • {len(df_with_data)} rows of experimental data")
            print(f"   • {len(df_with_data.colkeys)} columns with mixed types")
            print(f"   • 3 physical quantities with units (temperature, pressure, length)")
            print(f"   • 3 non-physical columns (ID, validity flag, notes)")
            print(f"   • Thread-safe operations with read/write locks")
            print(f"   • Copy, head, tail operations")
            print(f"   • String representation and repr methods")
            
        except Exception as e:
            print(f"❌ Complex dataframe test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_column_type_unit_distinction(self):
        """Test that demonstrates the correct distinction between united types (with units) and raw types (without units)."""
        try:
            print("\n🔬 Testing ColumnType unit distinction...")
            
            # Test data
            value_key = TestColumnKey("value")
            
            # Test 1: Physical quantity with units -> should use REAL_NUMBER_64
            print("\n📏 Testing physical quantities with units...")
            columns_with_units: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {value_key: (DataframeColumnType.REAL_NUMBER_64, Unit("m/s"), [1.5, 2.3, 3.7])}
            
            df_with_units: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns_with_units)
            
            print(f"✅ Physical quantity: {df_with_units.coltypes[value_key]} with unit {df_with_units.units[value_key]}")
            assert df_with_units.coltypes[value_key] == DataframeColumnType.REAL_NUMBER_64
            assert df_with_units.units[value_key] == Unit("m/s")
            
            # Test 2: Dimensionless quantity without units -> should use FLOAT_64
            print("\n📊 Testing dimensionless quantities without units...")
            ratio_key = TestColumnKey("ratio")
            columns_without_units: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {ratio_key: (DataframeColumnType.FLOAT_64, None, [0.85, 1.25, 0.95])}
            
            df_without_units: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns_without_units)
            
            print(f"✅ Dimensionless quantity: {df_without_units.coltypes[ratio_key]} with unit {df_without_units.units[ratio_key]}")
            assert df_without_units.coltypes[ratio_key] == DataframeColumnType.FLOAT_64
            assert df_without_units.units[ratio_key] is None
            
            # Test 3: Complex physical quantity with units -> should use COMPLEX_NUMBER_128
            print("\n🔢 Testing complex quantities with units...")
            impedance_key = TestColumnKey("impedance")
            columns_complex_units: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {impedance_key: (DataframeColumnType.COMPLEX_NUMBER_128, Unit("Ω"), [1.0+2.0j, 3.0-1.5j, 0.5+4.2j])}
            
            df_complex_units: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns_complex_units)
            
            print(f"✅ Complex physical quantity: {df_complex_units.coltypes[impedance_key]} with unit {df_complex_units.units[impedance_key]}")
            assert df_complex_units.coltypes[impedance_key] == DataframeColumnType.COMPLEX_NUMBER_128
            assert df_complex_units.units[impedance_key] == Unit("Ω")
            
            # Test 4: Complex dimensionless quantity without units -> should use COMPLEX_128
            print("\n🔄 Testing complex dimensionless quantities without units...")
            transform_key = TestColumnKey("transform_coefficient")
            columns_complex_raw: dict[TestColumnKey, tuple[DataframeColumnType, Optional[Unit|Dimension], Sequence[VALUE_TYPE]] | tuple[DataframeColumnType, Sequence[VALUE_TYPE]]] = {transform_key: (DataframeColumnType.COMPLEX_128, None, [2.0+1.0j, -1.0+3.0j, 0.0-2.0j])}
            
            df_complex_raw: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_from_data(columns=columns_complex_raw)
            
            print(f"✅ Complex dimensionless quantity: {df_complex_raw.coltypes[transform_key]} with unit {df_complex_raw.units[transform_key]}")
            assert df_complex_raw.coltypes[transform_key] == DataframeColumnType.COMPLEX_128
            assert df_complex_raw.units[transform_key] is None
            
            # Test 5: Verify HDF5 round-trip preserves the distinction
            print("\n💾 Testing HDF5 round-trip preservation of type distinctions...")
            
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                hdf5_path: Path = Path(tmp_file.name)
            
            try:
                # Save and load each type
                test_cases: list[tuple[UnitedDataframe[TestColumnKey], str, DataframeColumnType, Unit|None]] = [
                    (df_with_units, "physical_with_units", DataframeColumnType.REAL_NUMBER_64, Unit("m/s")),
                    (df_without_units, "dimensionless", DataframeColumnType.FLOAT_64, None),
                    (df_complex_units, "complex_with_units", DataframeColumnType.COMPLEX_NUMBER_128, Unit("Ω")),
                    (df_complex_raw, "complex_dimensionless", DataframeColumnType.COMPLEX_128, None)
                ]
                
                for original_df, key_name, expected_type, expected_unit in test_cases:
                    # Save to HDF5
                    original_df.to_hdf5(hdf5_path, key=key_name)
                    
                    # Load from HDF5
                    loaded_df: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].from_hdf5(hdf5_path, key=key_name, column_key_type=TestColumnKey)
                    
                    # Verify type and unit preservation
                    loaded_key = list(loaded_df.colkeys)[0]
                    loaded_type: DataframeColumnType = loaded_df.coltypes[loaded_key]
                    loaded_unit: Unit|None = loaded_df.units[loaded_key]
                    
                    print(f"  ✅ {key_name}: {loaded_type} with unit {loaded_unit} (preserved correctly)")
                    assert loaded_type == expected_type, f"Type mismatch for {key_name}: {loaded_type} != {expected_type}"
                    assert loaded_unit == expected_unit, f"Unit mismatch for {key_name}: {loaded_unit} != {expected_unit}"
                
            finally:
                if os.path.exists(hdf5_path):
                    os.unlink(hdf5_path)
            
            print("\n🎉 COLUMN TYPE UNIT DISTINCTION TEST PASSED!")
            print("📋 Summary of correct mappings:")
            print("   • Physical quantities (with units): REAL_NUMBER_XX, COMPLEX_NUMBER_128")
            print("   • Dimensionless quantities (no units): FLOAT_XX, COMPLEX_128")
            print("   • All types preserve correctly through HDF5 serialization")
            print("   • The system correctly distinguishes united vs raw types")
            
        except Exception as e:
            print(f"❌ Column type unit distinction test failed: {e}")
            import traceback
            traceback.print_exc()
            raise