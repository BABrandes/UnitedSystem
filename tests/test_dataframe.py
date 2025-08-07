#!/usr/bin/env python3
"""
Test suite for UnitedDataframe debugging.

This test suite systematically tests UnitedDataframe functionality to identify
and debug any issues with the implementation.
"""

from typing import Any

from united_system import DataframeColumnKey

class TestColumnKey(DataframeColumnKey):
    """Test column key."""
    
    def __init__(self, name: str):
        self.name = name

    def to_united_dataframe_string(self) -> str:
        return self.name
    
    @classmethod
    def from_united_dataframe_string(cls, string: str) -> 'TestColumnKey':
        return cls(string)
    
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TestColumnKey):
            return False
        return self.name == other.name
    
    def __hash__(self):
        return hash(self.name)
    
    def __repr__(self):
        return f"TestColumnKey({self.name!r})"

def run_instantiation_tests():
    """Run all instantiation tests and report results."""
    print("🧪 Running UnitedDataframe instantiation tests...")

    from tests.test_dataframe_core import TestUnitedDataframeCore
    from tests.test_dataframe_serialization import TestUnitedDataframeSerialization
    from tests.test_dataframe_operations import TestUnitedDataframeOperations
    
    test_class_core = TestUnitedDataframeCore()
    test_class_serialization = TestUnitedDataframeSerialization()
    test_class_operations = TestUnitedDataframeOperations()
    
    tests = [
        # Core functionality tests
        test_class_core.test_empty_dataframe_creation,
        test_class_core.test_simple_numeric_dataframe_creation,
        test_class_core.test_dataframe_with_data_creation,
        test_class_core.test_dataframe_column_access,
        test_class_core.test_dataframe_data_manipulation,
        test_class_core.test_dataframe_thread_safety,
        test_class_core.test_complex_dataframe_with_operations,
        test_class_core.test_column_type_unit_distinction,
        
        # Operations tests
        test_class_operations.test_column_operations_comprehensive,
        test_class_operations.test_row_operations_comprehensive,
        test_class_operations.test_cell_operations_comprehensive,
        test_class_operations.test_data_retrieval_comprehensive,
        test_class_operations.test_metadata_access_comprehensive,
        test_class_operations.test_read_only_mode_comprehensive,
        
        # Serialization tests
        test_class_serialization.test_hdf5_serialization_simple,
        test_class_serialization.test_hdf5_serialization_complex,
        test_class_serialization.test_hdf5_round_trip_fidelity,
        test_class_serialization.test_hdf5_comprehensive_all_types_complex_units,
        test_class_serialization.test_hdf5_timestamp_serialization,
        test_class_serialization.test_jupyter_notebook_display,
        test_class_serialization.test_pickle_serialization_simple,
        test_class_serialization.test_pickle_serialization_complex,
        test_class_serialization.test_pickle_multiprocessing_compatibility,
        test_class_serialization.test_pickle_round_trip_fidelity,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n🔍 Running {test.__name__}...")
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_instantiation_tests()
    if not success:
        exit(1)
