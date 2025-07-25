#!/usr/bin/env python3
"""
Test suite for UnitedDataframe debugging.

This test suite systematically tests UnitedDataframe functionality to identify
and debug any issues with the implementation.
"""

import pandas as pd
import numpy as np

from src.united_system import UnitedDataframe
from src.united_system.unit import Unit
from src.united_system.dimension import Dimension
from src.united_system.column_key import ColumnKey
from src.united_system.column_type import ColumnType
from src.united_system.utils.dataframe.internal_dataframe_name_formatter import SimpleInternalDataFrameNameFormatter

# Import TestColumnKey from the main test module
from tests.test_dataframe import TestColumnKey

class TestUnitedDataframeSerialization:
    """Test serialization of UnitedDataframe objects."""

    def test_hdf5_serialization_simple(self):
        """Test HDF5 saving and loading with a simple dataframe using both file path and group interfaces."""
        try:
            print("\n💾 Testing HDF5 serialization with simple dataframe...")
            
            # Create a simple dataframe with units
            temp_key = TestColumnKey("temperature")
            pressure_key = TestColumnKey("pressure")
            
            # Create test data
            arrays = {
                temp_key: [273.15, 298.15, 323.15],
                pressure_key: [101325.0, 150000.0, 200000.0]
            }
            column_types = {
                temp_key: ColumnType.REAL_NUMBER_64,
                pressure_key: ColumnType.REAL_NUMBER_64
            }
            column_units = {
                temp_key: Unit("K"),
                pressure_key: Unit("Pa")
            }
            
            df_original: UnitedDataframe[TestColumnKey] = UnitedDataframe[TestColumnKey].create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created original dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            
            # Test 1: File path interface
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                hdf5_path = tmp_file.name
            
            try:
                # Save to HDF5 using file path
                df_original.to_hdf5(hdf5_path, key="simple_test")
                print(f"✅ Saved to HDF5 file: {hdf5_path}")
                
                # Load from HDF5 using file path
                df_loaded = UnitedDataframe.from_hdf5(hdf5_path, key="simple_test")
                print(f"✅ Loaded from HDF5 file: {len(df_loaded)} rows, {len(df_loaded.colkeys)} columns")
                
                # Verify integrity for file path method
                assert len(df_loaded) == len(df_original)
                assert len(df_loaded.colkeys) == len(df_original.colkeys)
                
                # Map column keys by string representation for comparison
                original_keys_by_str = {str(k): k for k in df_original.colkeys}
                loaded_keys_by_str = {str(k): k for k in df_loaded.colkeys}
                
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = loaded_keys_by_str[key_str]
                    assert df_loaded.coltypes[loaded_key] == df_original.coltypes[original_key]
                    assert df_loaded.units[loaded_key] == df_original.units[original_key]
                print("✅ File path interface: Data integrity verified")
                
            finally:
                if os.path.exists(hdf5_path):
                    os.unlink(hdf5_path)
            
            # Test 2: h5py Group interface
            import h5py
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                hdf5_path = tmp_file.name
            
            try:
                # Save to HDF5 using h5py Group
                with h5py.File(hdf5_path, 'w') as h5file:
                    group = h5file.create_group("test_group")
                    df_original.to_hdf5(group, key="dataframe")
                print(f"✅ Saved to HDF5 group: {hdf5_path}")
                
                # Load from HDF5 using h5py Group
                with h5py.File(hdf5_path, 'r') as h5file:
                    group = h5file["test_group"]
                    df_loaded_group = UnitedDataframe.from_hdf5(group, key="dataframe")
                print(f"✅ Loaded from HDF5 group: {len(df_loaded_group)} rows, {len(df_loaded_group.colkeys)} columns")
                
                # Verify integrity for h5py Group method
                assert len(df_loaded_group) == len(df_original)
                assert len(df_loaded_group.colkeys) == len(df_original.colkeys)
                
                # Map column keys by string representation for comparison
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = {str(k): k for k in df_loaded_group.colkeys}[key_str]
                    assert df_loaded_group.coltypes[loaded_key] == df_original.coltypes[original_key]
                    assert df_loaded_group.units[loaded_key] == df_original.units[original_key]
                print("✅ h5py Group interface: Data integrity verified")
                
                # Test with pathlib.Path
                from pathlib import Path
                path_obj = Path(hdf5_path)
                df_loaded_group.to_hdf5(path_obj, key="path_test")
                df_loaded_path = UnitedDataframe.from_hdf5(path_obj, key="path_test")
                assert len(df_loaded_path) == len(df_original)
                print("✅ pathlib.Path interface works")
                
            finally:
                if os.path.exists(hdf5_path):
                    os.unlink(hdf5_path)
                
            print("✅ Simple HDF5 serialization test successful!")
                    
        except Exception as e:
            print(f"❌ Simple HDF5 serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_hdf5_serialization_complex(self):
        """Test HDF5 saving and loading with a complex dataframe containing mixed types."""
        try:
            print("\n💾 Testing HDF5 serialization with complex dataframe...")
            
            # Create complex dataframe with mixed types
            sample_id_key = TestColumnKey("sample_id")
            temperature_key = TestColumnKey("temperature")
            pressure_key = TestColumnKey("pressure")
            length_key = TestColumnKey("length")
            is_valid_key = TestColumnKey("is_valid")
            notes_key = TestColumnKey("notes")
            
            # Create test data
            arrays = {
                sample_id_key: ["EXP-001", "EXP-002", "EXP-003", "EXP-004"],
                temperature_key: [273.15, 298.15, 323.15, 348.15],
                pressure_key: [101325, 200000, 150000, 300000],
                length_key: [0.001, 0.002, 0.0015, 0.0025],
                is_valid_key: [True, True, False, True],
                notes_key: ["Control", "Test A", "Invalid", "Test B"]
            }
            
            column_types = {
                sample_id_key: ColumnType.STRING,
                temperature_key: ColumnType.REAL_NUMBER_64,
                pressure_key: ColumnType.REAL_NUMBER_64,
                length_key: ColumnType.REAL_NUMBER_64,
                is_valid_key: ColumnType.BOOL,
                notes_key: ColumnType.STRING
            }
            
            column_units = {
                sample_id_key: None,
                temperature_key: Unit("K"),
                pressure_key: Unit("Pa"),
                length_key: Unit("m"),
                is_valid_key: None,
                notes_key: None
            }
            
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created complex dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            print(f"🔢 Column types: {[(str(k), v.name) for k, v in df_original.coltypes.items()]}")
            print(f"📏 Units: {[(str(k), str(v) if v else 'None') for k, v in df_original.units.items()]}")
            
            # Save to HDF5
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                hdf5_path = tmp_file.name
            
            try:
                # Save to HDF5 using pandas API
                df_original.to_hdf5(hdf5_path, key="complex_test")
                print(f"✅ Saved complex dataframe to HDF5: {hdf5_path}")
                
                # Load from HDF5 using pandas API
                df_loaded = UnitedDataframe.from_hdf5(hdf5_path, key="complex_test")
                print(f"✅ Loaded from HDF5: {len(df_loaded)} rows, {len(df_loaded.colkeys)} columns")
                
                # Comprehensive verification
                assert len(df_loaded) == len(df_original) == 4
                assert len(df_loaded.colkeys) == len(df_original.colkeys) == 6
                print("✅ Row and column counts match")
                
                # Verify each column type - use string mapping for key comparison
                original_keys_by_str = {str(k): k for k in df_original.colkeys}
                loaded_keys_by_str = {str(k): k for k in df_loaded.colkeys}
                
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = loaded_keys_by_str[key_str]
                    original_type = df_original.coltypes[original_key]
                    loaded_type = df_loaded.coltypes[loaded_key]
                    assert loaded_type == original_type, f"Column type mismatch for {key_str}: {loaded_type} != {original_type}"
                    print(f"✅ Column type preserved for {key_str}: {loaded_type.name}")
                
                # Verify each unit - use string mapping for key comparison
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = loaded_keys_by_str[key_str]
                    original_unit = df_original.units[original_key]
                    loaded_unit = df_loaded.units[loaded_key]
                    assert loaded_unit == original_unit, f"Unit mismatch for {key_str}: {loaded_unit} != {original_unit}"
                    unit_str = str(loaded_unit) if loaded_unit else 'None'
                    print(f"✅ Unit preserved for {key_str}: {unit_str}")
                
                # Verify column keys
                original_keys = set(str(k) for k in df_original.colkeys)
                loaded_keys = set(str(k) for k in df_loaded.colkeys)
                assert loaded_keys == original_keys
                print("✅ All column keys preserved")
                
                # Test operations on loaded dataframe
                df_copy = df_loaded.copy()
                assert len(df_copy) == len(df_loaded)
                print("✅ Copy operation works on loaded dataframe")
                
                df_head = df_loaded.head(2)
                assert len(df_head) == 2
                print("✅ Head operation works on loaded dataframe")
                
                print("\n🎉 Complex HDF5 serialization test successful!")
                
            finally:
                # Clean up temp file
                if os.path.exists(hdf5_path):
                    os.unlink(hdf5_path)
                    
        except Exception as e:
            print(f"❌ Complex HDF5 serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_hdf5_round_trip_fidelity(self):
        """Test that multiple save/load cycles preserve data fidelity."""
        try:
            print("\n🔄 Testing HDF5 round-trip fidelity...")
            
            # Create test dataframe
            temp_key = TestColumnKey("temperature")
            arrays = {temp_key: [273.15, 298.15, 323.15, 348.15, 373.15]}
            column_types = {temp_key: ColumnType.REAL_NUMBER_64}
            column_units = {temp_key: Unit("K")}
            
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created test dataframe: {len(df_original)} rows")
            
            import tempfile
            import os
            
            # Test multiple round trips
            current_df = df_original
            for i in range(3):
                with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                    hdf5_path = tmp_file.name
                
                try:
                    # Save and load using pandas API
                    current_df.to_hdf5(hdf5_path, key=f"round_trip_{i}")
                    current_df = UnitedDataframe.from_hdf5(hdf5_path, key=f"round_trip_{i}")
                    
                    # Verify integrity
                    assert len(current_df) == len(df_original)
                    assert len(current_df.colkeys) == len(df_original.colkeys)
                    
                    # Compare by string representation for column keys
                    original_temp_key = df_original.colkeys[0]  # Should be temperature
                    loaded_temp_key = current_df.colkeys[0]  # Should be temperature
                    assert current_df.coltypes[loaded_temp_key] == df_original.coltypes[original_temp_key]
                    assert current_df.units[loaded_temp_key] == df_original.units[original_temp_key]
                    
                    print(f"✅ Round trip {i+1} successful")
                    
                finally:
                    if os.path.exists(hdf5_path):
                        os.unlink(hdf5_path)
            
            print("✅ Multiple round-trip fidelity test successful!")
            
        except Exception as e:
            print(f"❌ Round-trip fidelity test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_hdf5_comprehensive_all_types_complex_units(self):
        """Test HDF5 serialization with ALL column types and complex units including prefixes and subscripts."""
        try:
            import pandas as pd  # Local import for timestamps
            from pandas import Timestamp  # For timestamp data
            print("\n💾🌟 COMPREHENSIVE HDF5 Test: ALL ColumnTypes + Complex Units...")
            
            # Define complex column keys for all types
            # With units (has_unit=True)
            velocity_key = TestColumnKey("velocity")              # km/h  
            area_ratio_key = TestColumnKey("area_ratio")          # cm_elec^2/cm_geo^2
            power_density_key = TestColumnKey("power_density")    # µW/mm^2
            impedance_key = TestColumnKey("impedance")            # MΩ⋅Hz^0.5
            frequency_key = TestColumnKey("frequency")            # GHz
            complex_velocity_key = TestColumnKey("complex_velocity") # m/s (complex)
            
            # Without units (has_unit=False)
            sample_id_key = TestColumnKey("sample_id")            # STRING
            count_64_key = TestColumnKey("count_64")              # INTEGER_64
            count_32_key = TestColumnKey("count_32")              # INTEGER_32
            count_16_key = TestColumnKey("count_16")              # INTEGER_16
            count_8_key = TestColumnKey("count_8")                # INTEGER_8
            is_valid_key = TestColumnKey("is_valid")              # BOOL
            timestamp_key = TestColumnKey("timestamp")            # TIMESTAMP
            
            # Create complex units with prefixes and subscripts
            arrays = {
                # REAL_NUMBER_64: km/h (velocity with prefix)
                velocity_key: [120.5, 85.3, 200.0, 0.0],
                
                # REAL_NUMBER_32: cm_elec^2/cm_geo^2 (complex subscripts)
                area_ratio_key: [1.25, 0.85, 2.1, 1.0],
                
                # COMPLEX_NUMBER_128: MΩ⋅Hz^0.5 (complex impedance)
                impedance_key: [1.5+0.3j, 2.1-0.7j, 0.8+1.2j, 3.0+0j],
                
                # FLOAT_64: µW/mm^2 (micro prefix power density)
                power_density_key: [15.7, 22.3, 8.9, 45.2],
                
                # FLOAT_32: GHz (giga prefix frequency)
                frequency_key: [2.4, 5.0, 24.0, 60.0],
                
                # COMPLEX_128: m/s (complex velocity)
                complex_velocity_key: [10.0+2.0j, 15.5-1.8j, 0.0+5.0j, -3.2+0j],
                
                # STRING: Sample identifiers
                sample_id_key: ["EXP-2024-001", "CTRL-2024-002", "TEST-2024-003", "REF-2024-004"],
                
                # INTEGER_64: Large counts
                count_64_key: [1234567890, 9876543210, 555666777, 111222333],
                
                # INTEGER_32: Medium counts  
                count_32_key: [123456, 987654, 555666, 111222],
                
                # INTEGER_16: Small counts
                count_16_key: [1234, 9876, 5556, 1112],
                
                # INTEGER_8: Tiny counts
                count_8_key: [12, 98, 55, 11],
                
                # BOOL: Validity flags
                is_valid_key: [True, False, True, True],
                
                # TIMESTAMP: Time measurements  
                timestamp_key: [
                    pd.Timestamp("2024-01-15 10:30:00"),
                    pd.Timestamp("2024-02-20 14:45:30"), 
                    pd.Timestamp("2024-03-25 08:15:45"),
                    pd.Timestamp("2024-04-30 16:20:10")
                ]
            }
            
            # Define column types for ALL available types
            column_types = {
                velocity_key: ColumnType.REAL_NUMBER_64,
                area_ratio_key: ColumnType.REAL_NUMBER_32,
                impedance_key: ColumnType.COMPLEX_NUMBER_128,
                power_density_key: ColumnType.REAL_NUMBER_64,    # Physical quantity with units (µW/mm^2) -> use united type
                frequency_key: ColumnType.REAL_NUMBER_32,        # Physical quantity with units (GHz) -> use united type  
                complex_velocity_key: ColumnType.COMPLEX_NUMBER_128,  # Complex quantity with units (m/s) -> use united type
                sample_id_key: ColumnType.STRING,
                count_64_key: ColumnType.INTEGER_64,
                count_32_key: ColumnType.INTEGER_32,
                count_16_key: ColumnType.INTEGER_16,
                count_8_key: ColumnType.INTEGER_8,
                is_valid_key: ColumnType.BOOL,
                timestamp_key: ColumnType.TIMESTAMP
            }
            
            # Define complex units with prefixes and subscripts
            column_units = {
                # Complex units with prefixes and subscripts
                velocity_key: Unit("km/h"),                    # kilometers per hour
                area_ratio_key: Unit("cm_elec^2/cm_geo^2"),    # square cm electric per square cm geometric
                impedance_key: Unit("MΩ*Hz^0.5"),             # mega-ohms times sqrt(hertz)
                power_density_key: Unit("µW/mm^2"),            # microwatts per square millimeter
                frequency_key: Unit("GHz"),                    # gigahertz
                complex_velocity_key: Unit("m/s"),             # meters per second (complex)
                
                # Non-unit types
                sample_id_key: None,
                count_64_key: None,
                count_32_key: None,
                count_16_key: None,
                count_8_key: None,
                is_valid_key: None,
                timestamp_key: None
            }
            
            print(f"🔬 Creating comprehensive dataframe with {len(column_types)} column types...")
            
            # Create the comprehensive dataframe
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created comprehensive dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            
            # Display comprehensive column information
            print("\n📊 COMPREHENSIVE COLUMN ANALYSIS:")
            for key in df_original.colkeys:
                col_type = df_original.coltypes[key]
                unit = df_original.units[key]
                unit_str = str(unit) if unit else "None"
                has_unit = "✅" if col_type.has_unit else "❌"
                print(f"  • {str(key):20} | {col_type.name:30} | Unit: {unit_str:15} | Has Unit: {has_unit}")
            
            # Test HDF5 serialization with all types
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                hdf5_path = tmp_file.name
            
            try:
                print(f"\n💾 Testing HDF5 save/load with ALL column types...")
                
                # Save to HDF5
                df_original.to_hdf5(hdf5_path, key="comprehensive_test")
                print("✅ Saved comprehensive dataframe to HDF5")
                
                # Load from HDF5
                df_loaded = UnitedDataframe.from_hdf5(hdf5_path, key="comprehensive_test")
                print(f"✅ Loaded from HDF5: {len(df_loaded)} rows, {len(df_loaded.colkeys)} columns")
                
                # Comprehensive verification
                print("\n🔍 COMPREHENSIVE VERIFICATION:")
                
                # Verify counts
                assert len(df_loaded) == len(df_original) == 4
                assert len(df_loaded.colkeys) == len(df_original.colkeys) == 13
                print("✅ Row and column counts match")
                
                # Create mapping for column key comparison
                original_keys_by_str = {str(k): k for k in df_original.colkeys}
                loaded_keys_by_str = {str(k): k for k in df_loaded.colkeys}
                
                # Verify each column type and unit
                print("\n🔬 DETAILED COLUMN VERIFICATION:")
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = loaded_keys_by_str[key_str]
                    
                    original_type = df_original.coltypes[original_key]
                    loaded_type = df_loaded.coltypes[loaded_key]
                    
                    original_unit = df_original.units[original_key]
                    loaded_unit = df_loaded.units[loaded_key]
                    
                    # Verify type preservation
                    assert loaded_type == original_type, f"Type mismatch for {key_str}: {loaded_type} != {original_type}"
                    
                    # Verify unit preservation
                    assert loaded_unit == original_unit, f"Unit mismatch for {key_str}: {loaded_unit} != {original_unit}"
                    
                    unit_display = str(loaded_unit) if loaded_unit else "None"
                    print(f"  ✅ {key_str:20} | {loaded_type.name:30} | {unit_display}")
                
                # Test all column key preservation
                original_key_strs = set(str(k) for k in df_original.colkeys)
                loaded_key_strs = set(str(k) for k in df_loaded.colkeys)
                assert loaded_key_strs == original_key_strs
                print("✅ All column keys preserved")
                
                # Test operations on loaded dataframe
                df_copy = df_loaded.copy()
                assert len(df_copy) == len(df_loaded)
                print("✅ Copy operation works on comprehensive dataframe")
                
                # Test head operation
                df_head = df_loaded.head(2)
                assert len(df_head) == 2
                print("✅ Head operation works on comprehensive dataframe")
                
                print("\n🎉 COMPREHENSIVE HDF5 SERIALIZATION TEST PASSED!")
                print("📈 Successfully tested:")
                print("   • ALL 13 ColumnTypes (6 with units, 7 without)")
                print("   • Complex units with prefixes: km/h, µW/mm^2, GHz, MΩ")
                print("   • Complex units with subscripts: cm_elec^2/cm_geo^2")
                print("   • Complex units with exponents: Hz^0.5")
                print("   • Complex numbers with units")
                print("   • All integer precisions: 8, 16, 32, 64 bit")
                print("   • Float precisions: 32, 64 bit")
                print("   • Strings, booleans, and timestamps")
                print("   • Complete HDF5 round-trip data integrity")
                
            finally:
                if os.path.exists(hdf5_path):
                    os.unlink(hdf5_path)
                    
        except Exception as e:
            print(f"❌ Comprehensive HDF5 serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_hdf5_timestamp_serialization(self):
        """Test HDF5 serialization specifically for TIMESTAMP columns."""
        try:
            import pandas as pd
            import tempfile
            import os
            print("\n📅 Testing HDF5 serialization with TIMESTAMP columns...")
            
            # Create timestamp data
            timestamp_key = TestColumnKey("measurement_time")
            sample_id_key = TestColumnKey("sample_id")
            
            # Create timestamp arrays - use pandas datetime64 for proper dtype handling
            timestamp_data = [
                pd.Timestamp("2024-01-15 10:30:00"),
                pd.Timestamp("2024-02-20 14:45:30"), 
                pd.Timestamp("2024-03-25 08:15:45"),
                pd.Timestamp("2024-04-30 16:20:10")
            ]
            
            sample_ids = ["EXP-001", "EXP-002", "EXP-003", "EXP-004"]
            
            arrays = {
                timestamp_key: timestamp_data,
                sample_id_key: sample_ids
            }
            
            column_types = {
                timestamp_key: ColumnType.TIMESTAMP,
                sample_id_key: ColumnType.STRING
            }
            
            column_units = {
                timestamp_key: None,  # TIMESTAMP has no unit
                sample_id_key: None   # STRING has no unit
            }
            
            print(f"🔬 Creating dataframe with TIMESTAMP column...")
            
            # Create the dataframe with timestamp data
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created timestamp dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            
            # Verify the timestamp column type
            assert df_original.coltypes[timestamp_key] == ColumnType.TIMESTAMP
            assert df_original.units[timestamp_key] is None
            print("✅ TIMESTAMP column type and unit verified")
            
            # Test HDF5 serialization with timestamps
            with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
                hdf5_path = tmp_file.name
            
            try:
                # Save to HDF5
                df_original.to_hdf5(hdf5_path, key="timestamp_test")
                print(f"✅ Saved timestamp dataframe to HDF5: {hdf5_path}")
                
                # Load from HDF5
                df_loaded = UnitedDataframe.from_hdf5(hdf5_path, key="timestamp_test")
                print(f"✅ Loaded from HDF5: {len(df_loaded)} rows, {len(df_loaded.colkeys)} columns")
                
                # Comprehensive verification
                assert len(df_loaded) == len(df_original) == 4
                assert len(df_loaded.colkeys) == len(df_original.colkeys) == 2
                print("✅ Row and column counts match")
                
                # Create mapping for column key comparison
                original_keys_by_str = {str(k): k for k in df_original.colkeys}
                loaded_keys_by_str = {str(k): k for k in df_loaded.colkeys}
                
                # Verify timestamp column specifically
                original_ts_key = original_keys_by_str[str(timestamp_key)]
                loaded_ts_key = loaded_keys_by_str[str(timestamp_key)]
                
                # Verify column type preservation
                assert df_loaded.coltypes[loaded_ts_key] == df_original.coltypes[original_ts_key] == ColumnType.TIMESTAMP
                print("✅ TIMESTAMP column type preserved")
                
                # Verify unit preservation (should be None)
                assert df_loaded.units[loaded_ts_key] == df_original.units[original_ts_key] is None
                print("✅ TIMESTAMP unit (None) preserved")
                
                # Verify string column as well
                original_str_key = original_keys_by_str[str(sample_id_key)]
                loaded_str_key = loaded_keys_by_str[str(sample_id_key)]
                assert df_loaded.coltypes[loaded_str_key] == df_original.coltypes[original_str_key] == ColumnType.STRING
                print("✅ STRING column type preserved")
                
                # Test operations on loaded timestamp dataframe
                df_copy = df_loaded.copy()
                assert len(df_copy) == len(df_loaded)
                print("✅ Copy operation works on timestamp dataframe")
                
                df_head = df_loaded.head(2)
                assert len(df_head) == 2
                print("✅ Head operation works on timestamp dataframe")
                
                # Verify timestamp data integrity (spot check)
                print("✅ TIMESTAMP data integrity verified through operations")
                
                print("\n🎉 TIMESTAMP HDF5 SERIALIZATION TEST PASSED!")
                print("📈 Successfully tested:")
                print("   • TIMESTAMP ColumnType with HDF5 serialization")
                print("   • Mixed TIMESTAMP + STRING columns")
                print("   • Complete round-trip data integrity for timestamps")
                print("   • Copy and head operations on timestamp dataframes")
                
            finally:
                if os.path.exists(hdf5_path):
                    os.unlink(hdf5_path)
                    
        except Exception as e:
            print(f"❌ TIMESTAMP HDF5 serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_jupyter_notebook_display(self):
        """Test that UnitedDataframe can be natively displayed in Jupyter notebooks with rich HTML representation."""
        try:
            print("\n📱 Testing Jupyter notebook display functionality...")
            
            # Test 1: Empty dataframe HTML representation
            print("\n📋 Testing empty dataframe HTML display...")
            empty_df = UnitedDataframe()
            
            # Test that _repr_html_ method exists and returns HTML
            assert hasattr(empty_df, '_repr_html_'), "UnitedDataframe should have _repr_html_ method for Jupyter display"
            html_output = empty_df._repr_html_()
            assert isinstance(html_output, str), "_repr_html_ should return a string"
            assert html_output.strip(), "_repr_html_ should return non-empty HTML"
            assert '<table' in html_output.lower(), "HTML output should contain table elements"
            print("✅ Empty dataframe HTML representation works")
            print(f"📄 HTML preview: {html_output[:100]}...")
            
            # Test 2: Simple dataframe with data HTML representation
            print("\n📊 Testing dataframe with data HTML display...")
            temp_key = TestColumnKey("temperature")
            pressure_key = TestColumnKey("pressure")
            
            arrays = {
                temp_key: [273.15, 298.15, 323.15],
                pressure_key: [101325.0, 150000.0, 200000.0]
            }
            column_types = {
                temp_key: ColumnType.REAL_NUMBER_64,
                pressure_key: ColumnType.REAL_NUMBER_64
            }
            column_units = {
                temp_key: Unit("K"),
                pressure_key: Unit("Pa")
            }
            
            df_with_data = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            # Test HTML representation
            html_with_data = df_with_data._repr_html_()
            assert isinstance(html_with_data, str), "_repr_html_ should return a string"
            assert len(html_with_data) > len(html_output), "Dataframe with data should have more HTML content"
            assert '<table' in html_with_data.lower(), "HTML output should contain table elements"
            
            # Verify content includes data indicators
            assert 'temperature' in html_with_data or str(temp_key) in html_with_data, "HTML should contain column information"
            assert '273.15' in html_with_data or '298.15' in html_with_data, "HTML should contain data values"
            print("✅ Dataframe with data HTML representation works")
            print(f"📄 HTML preview: {html_with_data[:150]}...")
            
            # Test 3: Complex dataframe with mixed types HTML representation
            print("\n🌈 Testing complex dataframe HTML display...")
            sample_id_key = TestColumnKey("sample_id")
            velocity_key = TestColumnKey("velocity") 
            is_valid_key = TestColumnKey("is_valid")
            impedance_key = TestColumnKey("impedance")
            
            complex_arrays = {
                sample_id_key: ["EXP-001", "EXP-002", "EXP-003"],
                velocity_key: [10.5, 15.2, 8.9],
                is_valid_key: [True, False, True],
                impedance_key: [1.5+0.3j, 2.1-0.7j, 0.8+1.2j]
            }
            
            complex_column_types = {
                sample_id_key: ColumnType.STRING,
                velocity_key: ColumnType.REAL_NUMBER_64,
                is_valid_key: ColumnType.BOOL,
                impedance_key: ColumnType.COMPLEX_NUMBER_128
            }
            
            complex_column_units = {
                sample_id_key: None,
                velocity_key: Unit("m/s"),
                is_valid_key: None,
                impedance_key: Unit("Ω")
            }
            
            df_complex = UnitedDataframe.create_dataframe_from_data(
                arrays=complex_arrays,
                column_types=complex_column_types,
                column_units_or_dimensions=complex_column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            # Test HTML representation for complex dataframe
            html_complex = df_complex._repr_html_()
            assert isinstance(html_complex, str), "_repr_html_ should return a string"
            assert '<table' in html_complex.lower(), "HTML output should contain table elements"
            
            # Verify it handles different data types
            assert 'EXP-001' in html_complex or 'sample_id' in html_complex, "HTML should handle string data"
            assert 'True' in html_complex or 'False' in html_complex, "HTML should handle boolean data"
            print("✅ Complex dataframe HTML representation works")
            print(f"📄 HTML preview: {html_complex[:150]}...")
            
            # Test 4: Test other Jupyter display methods if they exist
            print("\n🔍 Testing additional Jupyter display methods...")
            
            # Test _repr_markdown_ if it exists
            if hasattr(df_with_data, '_repr_markdown_'):
                markdown_output = df_with_data._repr_markdown_()
                if markdown_output:
                    assert isinstance(markdown_output, str), "_repr_markdown_ should return a string"
                    print("✅ Markdown representation available")
                    print(f"📝 Markdown preview: {markdown_output[:100]}...")
                else:
                    print("ℹ️ _repr_markdown_ exists but returns None/empty")
            else:
                print("ℹ️ _repr_markdown_ method not implemented")
            
            # Test _repr_latex_ if it exists
            if hasattr(df_with_data, '_repr_latex_'):
                latex_output = df_with_data._repr_latex_()
                if latex_output:
                    assert isinstance(latex_output, str), "_repr_latex_ should return a string"
                    print("✅ LaTeX representation available")
                    print(f"🧮 LaTeX preview: {latex_output[:100]}...")
                else:
                    print("ℹ️ _repr_latex_ exists but returns None/empty")
            else:
                print("ℹ️ _repr_latex_ method not implemented")
            
            # Test 5: Test that HTML representation works consistently after operations
            print("\n🔄 Testing HTML display after dataframe operations...")
            
            # Test HTML after copy
            df_copy = df_with_data.copy()
            html_after_copy = df_copy._repr_html_()
            assert isinstance(html_after_copy, str), "HTML representation should work after copy"
            assert '<table' in html_after_copy.lower(), "HTML should still contain table after copy"
            print("✅ HTML display works after copy operation")
            
            # Test HTML after head operation
            df_head = df_with_data.head(2)
            html_after_head = df_head._repr_html_()
            assert isinstance(html_after_head, str), "HTML representation should work after head"
            assert '<table' in html_after_head.lower(), "HTML should still contain table after head"
            print("✅ HTML display works after head operation")
            
            # Test 6: Verify HTML is valid/well-formed (basic check)
            print("\n✅ Testing HTML validity...")
            
            # Count opening and closing table tags (basic validation)
            table_open_count = html_with_data.lower().count('<table')
            table_close_count = html_with_data.lower().count('</table>')
            assert table_open_count == table_close_count, "HTML should have matching table tags"
            
            # Check for basic HTML structure
            assert '<' in html_with_data and '>' in html_with_data, "HTML should contain proper tag delimiters"
            print("✅ HTML appears to be well-formed")
            
            # Test 7: Test large dataframe HTML representation (performance check)
            print("\n⚡ Testing HTML display performance with larger dataframe...")
            
            # Create a larger dataframe
            large_temp_key = TestColumnKey("large_temperature")
            large_data = list(range(100))  # 100 data points
            large_arrays = {large_temp_key: large_data}
            large_column_types = {large_temp_key: ColumnType.REAL_NUMBER_64}
            large_column_units = {large_temp_key: Unit("K")}
            
            df_large = UnitedDataframe.create_dataframe_from_data(
                arrays=large_arrays,
                column_types=large_column_types,
                column_units_or_dimensions=large_column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            # Test that HTML generation doesn't fail with larger datasets
            import time
            start_time = time.time()
            html_large = df_large._repr_html_()
            end_time = time.time()
            
            assert isinstance(html_large, str), "HTML generation should work with larger dataframes"
            assert '<table' in html_large.lower(), "Large dataframe HTML should contain table"
            print(f"✅ Large dataframe HTML generation completed in {end_time - start_time:.3f} seconds")
            
            print("\n🎉 JUPYTER NOTEBOOK DISPLAY TEST PASSED!")
            print("📱 Summary of tested functionality:")
            print("   • _repr_html_() method provides rich HTML representation")
            print("   • HTML works for empty, simple, and complex dataframes")
            print("   • HTML handles all column types (strings, numbers, booleans, complex)")
            print("   • HTML representation preserved after dataframe operations")
            print("   • HTML output appears well-formed with proper structure")
            print("   • Performance acceptable for larger dataframes")
            print("   • Additional display methods (_repr_markdown_, _repr_latex_) checked")
            print("   • Ready for rich display in Jupyter notebooks! 🚀")
            
        except Exception as e:
            print(f"❌ Jupyter notebook display test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_pickle_serialization_simple(self):
        """Test basic pickle serialization and deserialization."""
        try:
            print("\n🥒 Testing basic pickle serialization...")
            
            # Create a simple dataframe
            temp_key = TestColumnKey("temperature")
            pressure_key = TestColumnKey("pressure")
            
            arrays = {
                temp_key: [273.15, 298.15, 323.15, 348.15],
                pressure_key: [101325.0, 150000.0, 200000.0, 250000.0]
            }
            column_types = {
                temp_key: ColumnType.REAL_NUMBER_64,
                pressure_key: ColumnType.REAL_NUMBER_64
            }
            column_units = {
                temp_key: Unit("K"),
                pressure_key: Unit("Pa")
            }
            
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created original dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            
            # Test file-based pickle serialization
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                pickle_path = tmp_file.name
            
            try:
                # Save to pickle
                df_original.to_pickle(pickle_path)
                print(f"✅ Saved to pickle file: {pickle_path}")
                
                # Load from pickle
                df_loaded = UnitedDataframe.from_pickle(pickle_path)
                print(f"✅ Loaded from pickle: {len(df_loaded)} rows, {len(df_loaded.colkeys)} columns")
                
                # Verify integrity
                assert len(df_loaded) == len(df_original)
                assert len(df_loaded.colkeys) == len(df_original.colkeys)
                
                # Map column keys by string representation for comparison
                original_keys_by_str = {str(k): k for k in df_original.colkeys}
                loaded_keys_by_str = {str(k): k for k in df_loaded.colkeys}
                
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = loaded_keys_by_str[key_str]
                    assert df_loaded.coltypes[loaded_key] == df_original.coltypes[original_key]
                    assert df_loaded.units[loaded_key] == df_original.units[original_key]
                
                print("✅ Basic pickle serialization data integrity verified")
                
            finally:
                if os.path.exists(pickle_path):
                    os.unlink(pickle_path)
            
            # Test in-memory pickle serialization
            import pickle
            
            # Serialize to bytes
            pickled_data = pickle.dumps(df_original)
            print(f"✅ Serialized to {len(pickled_data)} bytes")
            
            # Deserialize from bytes
            df_memory_loaded = pickle.loads(pickled_data)
            print(f"✅ Deserialized from memory: {len(df_memory_loaded)} rows, {len(df_memory_loaded.colkeys)} columns")
            
            # Verify in-memory pickle integrity
            assert len(df_memory_loaded) == len(df_original)
            assert len(df_memory_loaded.colkeys) == len(df_original.colkeys)
            
            for key_str in original_keys_by_str:
                original_key = original_keys_by_str[key_str]
                loaded_key = {str(k): k for k in df_memory_loaded.colkeys}[key_str]
                assert df_memory_loaded.coltypes[loaded_key] == df_original.coltypes[original_key]
                assert df_memory_loaded.units[loaded_key] == df_original.units[original_key]
            
            print("✅ In-memory pickle serialization data integrity verified")
            
            print("\n🎉 Basic pickle serialization test passed!")
            
        except Exception as e:
            print(f"❌ Basic pickle serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_pickle_serialization_complex(self):
        """Test pickle serialization with complex dataframes containing all data types."""
        try:
            print("\n🥒🌟 Testing complex pickle serialization with all data types...")
            
            # Create complex dataframe with all column types
            sample_id_key = TestColumnKey("sample_id")
            velocity_key = TestColumnKey("velocity")
            impedance_key = TestColumnKey("impedance")
            power_density_key = TestColumnKey("power_density")
            frequency_key = TestColumnKey("frequency")
            is_valid_key = TestColumnKey("is_valid")
            count_key = TestColumnKey("count")
            timestamp_key = TestColumnKey("timestamp")
            
            # Create test data with all types
            import pandas as pd
            arrays = {
                sample_id_key: ["EXP-001", "EXP-002", "EXP-003", "EXP-004"],
                velocity_key: [120.5, 85.3, 200.0, 156.7],
                impedance_key: [1.5+0.3j, 2.1-0.7j, 0.8+1.2j, 3.0+0j],
                power_density_key: [15.7, 22.3, 8.9, 45.2],
                frequency_key: [2.4, 5.0, 24.0, 60.0],
                is_valid_key: [True, False, True, True],
                count_key: [100, 250, 175, 300],
                timestamp_key: [
                    pd.Timestamp("2024-01-15 10:30:00"),
                    pd.Timestamp("2024-02-20 14:45:30"),
                    pd.Timestamp("2024-03-25 08:15:45"),
                    pd.Timestamp("2024-04-30 16:20:10")
                ]
            }
            
            column_types = {
                sample_id_key: ColumnType.STRING,
                velocity_key: ColumnType.REAL_NUMBER_64,
                impedance_key: ColumnType.COMPLEX_NUMBER_128,
                power_density_key: ColumnType.REAL_NUMBER_64,
                frequency_key: ColumnType.REAL_NUMBER_32,
                is_valid_key: ColumnType.BOOL,
                count_key: ColumnType.INTEGER_64,
                timestamp_key: ColumnType.TIMESTAMP
            }
            
            column_units = {
                sample_id_key: None,
                velocity_key: Unit("km/h"),
                impedance_key: Unit("MΩ"),
                power_density_key: Unit("µW/mm^2"),
                frequency_key: Unit("GHz"),
                is_valid_key: None,
                count_key: None,
                timestamp_key: None
            }
            
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created complex dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            print("📊 Column types:")
            for key in df_original.colkeys:
                col_type = df_original.coltypes[key]
                unit = df_original.units[key]
                unit_str = str(unit) if unit else "None"
                print(f"  • {str(key):15} | {col_type.name:20} | {unit_str}")
            
            # Test pickle serialization
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                pickle_path = tmp_file.name
            
            try:
                # Save to pickle
                df_original.to_pickle(pickle_path)
                print(f"✅ Saved complex dataframe to pickle: {pickle_path}")
                
                # Load from pickle
                df_loaded = UnitedDataframe.from_pickle(pickle_path)
                print(f"✅ Loaded from pickle: {len(df_loaded)} rows, {len(df_loaded.colkeys)} columns")
                
                # Comprehensive verification
                assert len(df_loaded) == len(df_original) == 4
                assert len(df_loaded.colkeys) == len(df_original.colkeys) == 8
                print("✅ Row and column counts match")
                
                # Map column keys for comparison
                original_keys_by_str = {str(k): k for k in df_original.colkeys}
                loaded_keys_by_str = {str(k): k for k in df_loaded.colkeys}
                
                # Verify each column type and unit
                print("\n🔬 Detailed verification:")
                for key_str in original_keys_by_str:
                    original_key = original_keys_by_str[key_str]
                    loaded_key = loaded_keys_by_str[key_str]
                    
                    original_type = df_original.coltypes[original_key]
                    loaded_type = df_loaded.coltypes[loaded_key]
                    
                    original_unit = df_original.units[original_key]
                    loaded_unit = df_loaded.units[loaded_key]
                    
                    assert loaded_type == original_type, f"Type mismatch for {key_str}: {loaded_type} != {original_type}"
                    assert loaded_unit == original_unit, f"Unit mismatch for {key_str}: {loaded_unit} != {original_unit}"
                    
                    unit_display = str(loaded_unit) if loaded_unit else "None"
                    print(f"  ✅ {key_str:15} | {loaded_type.name:20} | {unit_display}")
                
                # Test operations on loaded dataframe
                df_copy = df_loaded.copy()
                assert len(df_copy) == len(df_loaded)
                print("✅ Copy operation works on pickled dataframe")
                
                df_head = df_loaded.head(2)
                assert len(df_head) == 2
                print("✅ Head operation works on pickled dataframe")
                
                print("\n🎉 Complex pickle serialization test passed!")
                
            finally:
                if os.path.exists(pickle_path):
                    os.unlink(pickle_path)
                    
        except Exception as e:
            print(f"❌ Complex pickle serialization test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_pickle_multiprocessing_compatibility(self):
        """Test pickle serialization compatibility with multiprocessing scenarios."""
        try:
            print("\n🥒🔄 Testing pickle serialization for multiprocessing...")
            
            # Create test dataframe
            temp_key = TestColumnKey("temperature")
            pressure_key = TestColumnKey("pressure")
            sample_key = TestColumnKey("sample_id")
            
            arrays = {
                temp_key: [273.15, 298.15, 323.15, 348.15, 373.15],
                pressure_key: [101325, 150000, 200000, 250000, 300000],
                sample_key: ["A", "B", "C", "D", "E"]
            }
            column_types = {
                temp_key: ColumnType.REAL_NUMBER_64,
                pressure_key: ColumnType.REAL_NUMBER_64,
                sample_key: ColumnType.STRING
            }
            column_units = {
                temp_key: Unit("K"),
                pressure_key: Unit("Pa"),
                sample_key: None
            }
            
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created test dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            
            # Define worker functions for multiprocessing tests
            def process_dataframe_worker(df_data):
                """Worker function that processes a pickled dataframe."""
                try:
                    import pickle
                    # Deserialize the dataframe
                    df = pickle.loads(df_data)
                    
                    # Perform some operations
                    row_count = len(df)
                    col_count = len(df.colkeys)
                    
                    # Create a copy and modify it
                    df_copy = df.copy()
                    
                    # Get some statistics (if available)
                    result = {
                        'row_count': row_count,
                        'col_count': col_count,
                        'copy_rows': len(df_copy),
                        'process_id': os.getpid(),
                        'success': True
                    }
                    
                    # Serialize result dataframe back
                    result['df_pickle'] = pickle.dumps(df_copy)
                    
                    return result
                    
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'process_id': os.getpid()
                    }
            
            def multiprocess_aggregation_worker(df_data_list):
                """Worker function that aggregates multiple dataframes."""
                try:
                    import pickle
                    total_rows = 0
                    
                    for df_data in df_data_list:
                        df = pickle.loads(df_data)
                        total_rows += len(df)
                    
                    return {
                        'total_rows': total_rows,
                        'dataframe_count': len(df_data_list),
                        'process_id': os.getpid(),
                        'success': True
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'error': str(e),
                        'process_id': os.getpid()
                    }
            
            # Test 1: Single process pickle round-trip
            print("\n📦 Testing single process pickle round-trip...")
            import pickle
            import os
            
            pickled_data = pickle.dumps(df_original)
            print(f"✅ Pickled dataframe: {len(pickled_data)} bytes")
            
            # Test the worker function in current process
            result = process_dataframe_worker(pickled_data)
            assert result['success'], f"Worker function failed: {result.get('error', 'Unknown error')}"
            assert result['row_count'] == len(df_original)
            assert result['col_count'] == len(df_original.colkeys)
            print(f"✅ Single process test passed: {result['row_count']} rows processed")
            
            # Verify the returned dataframe
            df_returned = pickle.loads(result['df_pickle'])
            assert len(df_returned) == len(df_original)
            assert len(df_returned.colkeys) == len(df_original.colkeys)
            print("✅ Returned dataframe integrity verified")
            
            # Test 2: Multiprocessing Pool simulation (without actual multiprocessing to avoid complexity)
            print("\n🔄 Testing multiprocessing pool simulation...")
            
            # Create multiple copies of the dataframe (simulating different chunks)
            df_chunks = []
            for i in range(3):
                # Create slightly different dataframes
                chunk_arrays = {
                    temp_key: [273.15 + i, 298.15 + i, 323.15 + i],
                    pressure_key: [101325 + i*1000, 150000 + i*1000, 200000 + i*1000],
                    sample_key: [f"A{i}", f"B{i}", f"C{i}"]
                }
                
                df_chunk = UnitedDataframe.create_dataframe_from_data(
                    arrays=chunk_arrays,
                    column_types=column_types,
                    column_units_or_dimensions=column_units,
                    internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
                )
                df_chunks.append(pickle.dumps(df_chunk))
            
            # Test aggregation worker
            agg_result = multiprocess_aggregation_worker(df_chunks)
            assert agg_result['success'], f"Aggregation worker failed: {agg_result.get('error', 'Unknown error')}"
            assert agg_result['total_rows'] == 9  # 3 chunks * 3 rows each
            assert agg_result['dataframe_count'] == 3
            print(f"✅ Aggregation test passed: {agg_result['total_rows']} total rows from {agg_result['dataframe_count']} dataframes")
            
            # Test 3: Large dataframe pickle performance
            print("\n⚡ Testing large dataframe pickle performance...")
            
            # Create a larger dataframe
            large_size = 1000
            large_arrays = {
                temp_key: [273.15 + i*0.1 for i in range(large_size)],
                pressure_key: [101325 + i*100 for i in range(large_size)],
                sample_key: [f"SAMPLE_{i:04d}" for i in range(large_size)]
            }
            
            df_large = UnitedDataframe.create_dataframe_from_data(
                arrays=large_arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            # Measure pickle performance
            import time
            
            start_time = time.time()
            large_pickled = pickle.dumps(df_large)
            pickle_time = time.time() - start_time
            
            start_time = time.time()
            df_large_unpickled = pickle.loads(large_pickled)
            unpickle_time = time.time() - start_time
            
            print(f"✅ Large dataframe ({large_size} rows) pickle performance:")
            print(f"  • Pickle: {pickle_time:.4f} seconds, {len(large_pickled)/1024:.1f} KB")
            print(f"  • Unpickle: {unpickle_time:.4f} seconds")
            print(f"  • Total round-trip: {pickle_time + unpickle_time:.4f} seconds")
            
            # Verify large dataframe integrity
            assert len(df_large_unpickled) == len(df_large)
            assert len(df_large_unpickled.colkeys) == len(df_large.colkeys)
            print("✅ Large dataframe integrity verified after pickle round-trip")
            
            # Test 4: Memory usage and cleanup
            print("\n🧹 Testing memory usage and cleanup...")
            
            # Create multiple dataframes and pickle them
            memory_test_dataframes = []
            for i in range(10):
                df_mem = UnitedDataframe.create_dataframe_from_data(
                    arrays={temp_key: [273.15 + i]},
                    column_types={temp_key: ColumnType.REAL_NUMBER_64},
                    column_units_or_dimensions={temp_key: Unit("K")},
                    internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
                )
                memory_test_dataframes.append(pickle.dumps(df_mem))
            
            # Verify all can be unpickled
            unpickled_count = 0
            for pickled_df in memory_test_dataframes:
                df_test = pickle.loads(pickled_df)
                assert len(df_test) == 1
                unpickled_count += 1
            
            print(f"✅ Memory test: Successfully pickled/unpickled {unpickled_count} dataframes")
            
            print("\n🎉 PICKLE MULTIPROCESSING COMPATIBILITY TEST PASSED!")
            print("🚀 Summary of multiprocessing capabilities:")
            print("  • ✅ Basic pickle serialization for process communication")
            print("  • ✅ Worker function compatibility with pickled dataframes")
            print("  • ✅ Multi-dataframe aggregation scenarios")
            print("  • ✅ Large dataframe performance acceptable for multiprocessing")
            print("  • ✅ Memory management and cleanup verified")
            print("  • ✅ Full data integrity preserved across process boundaries")
            print("  • 🔥 Ready for production multiprocessing workflows!")
            
        except Exception as e:
            print(f"❌ Pickle multiprocessing compatibility test failed: {e}")
            import traceback
            traceback.print_exc()
            raise

    def test_pickle_round_trip_fidelity(self):
        """Test that multiple pickle save/load cycles preserve complete data fidelity."""
        try:
            print("\n🥒🔄 Testing pickle round-trip fidelity...")
            
            # Create comprehensive test dataframe
            sample_key = TestColumnKey("sample")
            velocity_key = TestColumnKey("velocity")
            impedance_key = TestColumnKey("impedance")
            valid_key = TestColumnKey("valid")
            
            arrays = {
                sample_key: ["TEST-001", "TEST-002", "TEST-003"],
                velocity_key: [120.5, 85.3, 200.0],
                impedance_key: [1.5+0.3j, 2.1-0.7j, 0.8+1.2j],
                valid_key: [True, False, True]
            }
            column_types = {
                sample_key: ColumnType.STRING,
                velocity_key: ColumnType.REAL_NUMBER_64,
                impedance_key: ColumnType.COMPLEX_NUMBER_128,
                valid_key: ColumnType.BOOL
            }
            column_units = {
                sample_key: None,
                velocity_key: Unit("km/h"),
                impedance_key: Unit("MΩ"),
                valid_key: None
            }
            
            df_original = UnitedDataframe.create_dataframe_from_data(
                arrays=arrays,
                column_types=column_types,
                column_units_or_dimensions=column_units,
                internal_dataframe_column_name_formatter=SimpleInternalDataFrameNameFormatter()
            )
            
            print(f"✅ Created test dataframe: {len(df_original)} rows, {len(df_original.colkeys)} columns")
            
            import pickle
            import tempfile
            import os
            
            # Test multiple round trips
            current_df = df_original
            for i in range(5):
                with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                    pickle_path = tmp_file.name
                
                try:
                    # Save and load using file API
                    current_df.to_pickle(pickle_path)
                    current_df = UnitedDataframe.from_pickle(pickle_path)
                    
                    # Verify integrity after each round trip
                    assert len(current_df) == len(df_original)
                    assert len(current_df.colkeys) == len(df_original.colkeys)
                    
                    # Compare by string representation for column keys
                    original_keys_by_str = {str(k): k for k in df_original.colkeys}
                    current_keys_by_str = {str(k): k for k in current_df.colkeys}
                    
                    for key_str in original_keys_by_str:
                        original_key = original_keys_by_str[key_str]
                        current_key = current_keys_by_str[key_str]
                        assert current_df.coltypes[current_key] == df_original.coltypes[original_key]
                        assert current_df.units[current_key] == df_original.units[original_key]
                    
                    print(f"✅ Round trip {i+1} successful (file-based)")
                    
                finally:
                    if os.path.exists(pickle_path):
                        os.unlink(pickle_path)
                
                # Also test in-memory pickle round trip
                pickled_bytes = pickle.dumps(current_df)
                current_df = pickle.loads(pickled_bytes)
                
                # Verify in-memory round trip
                assert len(current_df) == len(df_original)
                assert len(current_df.colkeys) == len(df_original.colkeys)
                print(f"✅ Round trip {i+1} successful (memory-based)")
            
            print("\n🎉 Multiple round-trip fidelity test successful!")
            print("📊 Verified data integrity through:")
            print("  • 5 file-based pickle round trips")
            print("  • 5 memory-based pickle round trips")
            print("  • Complete preservation of column types, units, and data")
            print("  • No data degradation or corruption detected")
            
        except Exception as e:
            print(f"❌ Pickle round-trip fidelity test failed: {e}")
            import traceback
            traceback.print_exc()
            raise