#!/usr/bin/env python3
"""
Quick Start: UnitedSystem Naming Convention Refactoring

This script provides the IMMEDIATE first steps to start improving
your naming conventions. Run this to see the specific changes needed.
"""

def generate_united_dataframe_additions():
    """Generate the code to add to UnitedDataframe class for Phase 1."""
    
    code = '''
# =============================================================================
# PHASE 1: NEW NAMING CONVENTIONS - ADD TO UnitedDataframe CLASS
# =============================================================================

# Add these methods to src/united_system/united_dataframe/united_dataframe.py

# ----------- IMPROVED PROPERTIES -----------

@property
def column_count(self) -> int:
    """Get the number of columns in the dataframe.""" 
    with self._rlock:
        return len(self._column_keys)

@property
def row_count(self) -> int:
    """Get the number of rows in the dataframe."""
    with self._rlock:
        return len(self._internal_canonical_dataframe)

# ----------- COLUMN OPERATIONS -----------

def column_sum(self, column_key: CK) -> RealUnitedScalar:
    """Calculate the sum of a numeric column."""
    return self.colfun_sum(column_key)

def column_mean(self, column_key: CK) -> RealUnitedScalar:
    """Calculate the mean of a numeric column."""
    return self.colfun_mean(column_key)

def column_std(self, column_key: CK) -> RealUnitedScalar:
    """Calculate the standard deviation of a numeric column."""
    return self.colfun_std(column_key)

def column_min(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> RealUnitedScalar:
    """Get the minimum value of a numeric column."""
    return self.colfun_min(column_key, case)

def column_max(self, column_key: CK, case: Literal["only_positive", "only_negative", "only_non_negative", "only_non_positive", "all"] = "all") -> RealUnitedScalar:
    """Get the maximum value of a numeric column."""
    return self.colfun_max(column_key, case)

def column_as_numpy(self, column_key: CK, unit: Unit|None = None, precision: Literal[8, 16, 32, 64, 128, 256]|None = None) -> np.ndarray:
    """Get a column as a numpy array in specified units."""
    return self.column_values_as_numpy_array(column_key, unit, precision)

def column_as_pandas(self, column_key: CK, unit: Unit|None = None) -> pd.Series:
    """Get a column as a pandas Series in specified units."""
    return self.column_values_as_pandas_series(column_key, unit)

def column_as_array(self, column_key: CK, display_unit: Unit|None = None) -> ArrayLike:
    """Get a column as a UnitedArray with its display unit."""
    return self.column_values_as_array(column_key, display_unit)

def column_unique_values(self, column_key: CK) -> ArrayLike:
    """Get the unique values of a column."""
    return self.colfun_unique_as_array(column_key)

# ----------- ROW OPERATIONS -----------

def rows_head(self, n: int = 5) -> "UnitedDataframe[CK]":
    """Get the first n rows of the dataframe."""
    return self.rowfun_head(n)

def rows_tail(self, n: int = 5) -> "UnitedDataframe[CK]":
    """Get the last n rows of the dataframe."""
    return self.rowfun_tail(n)

def rows_last(self) -> "UnitedDataframe[CK]":
    """Get the last row of the dataframe."""
    return self.rowfun_last()

def row_get(self, row_index: int) -> dict[CK, UnitedScalar]:
    """Get a row from the dataframe."""
    return self.row(row_index)

def row_add(self, values: list[UnitedScalar]|dict[CK, UnitedScalar]) -> None:
    """Add a row to the dataframe."""
    return self.add_row(values)

def row_remove(self, row: int) -> None:
    """Remove a row from the dataframe."""
    return self.remove_row(row)

# ----------- MASK OPERATIONS -----------

def mask_is_na(self, subset: list[CK] | None = None) -> np.ndarray:
    """Return a boolean mask indicating which values are NA/NaN."""
    return self.maskfun_isna(subset)

def mask_not_na(self, subset: list[CK] | None = None) -> np.ndarray:
    """Return a boolean mask indicating which values are not NA/NaN."""
    return self.maskfun_notna(subset)

def mask_from_condition(self, column_key_and_callable: dict[CK, Callable[[UnitedScalar], bool] | Callable[[str], bool]]) -> np.ndarray:
    """Return a boolean mask of rows that satisfy filter functions."""
    return self.maskfun_get_from_filter(column_key_and_callable)

# ----------- FILTER OPERATIONS -----------

def filter_by_values(self, filter_dict: dict[CK, SCALAR_TYPE]) -> "UnitedDataframe[CK]":
    """Filter the dataframe by a dictionary of column keys and values."""
    return self.filterfun_by_filterdict(filter_dict)

# ----------- CELL OPERATIONS -----------

def cell_get(self, row_index: int, column_key: CK) -> UnitedScalar:
    """Get the value of a specific cell."""
    return self.cell_value_get(row_index, column_key)

def cell_set(self, row_index: int, column_key: CK, value: SCALAR_TYPE) -> None:
    """Set the value of a specific cell."""
    return self.cell_value_set(row_index, column_key, value)

def cell_is_empty(self, row_index: int, column_key: CK) -> bool:
    """Check if a cell is empty."""
    return self.cell_value_is_empty(row_index, column_key)
'''
    
    return code


def generate_type_variable_updates():
    """Generate improved type variable definitions."""
    
    code = '''
# =============================================================================
# IMPROVED TYPE VARIABLES - UPDATE IN RELEVANT FILES
# =============================================================================

# Add to the top of united_dataframe.py and other relevant files

from typing import TypeVar, Generic

# New, descriptive type variables
ColumnKeyType = TypeVar("ColumnKeyType", bound=ColumnKey|str, default=str)
FilterColumnKeyType = TypeVar("FilterColumnKeyType", bound=ColumnKey|str, default=str)

# Backward compatibility aliases
CK = ColumnKeyType
CK_CF = FilterColumnKeyType

# Update class definition
class UnitedDataframe(JSONable, HDF5able, Generic[ColumnKeyType]):
    """A unit-aware DataFrame that maintains type safety and thread safety."""
    # ... rest of class
'''
    
    return code


def create_testing_code():
    """Generate code to test the new naming conventions."""
    
    code = '''
# =============================================================================
# TESTING CODE - VERIFY NEW NAMING WORKS
# =============================================================================

def test_new_naming_conventions():
    """Test that new naming conventions work correctly."""
    
    # This is example test code - adapt to your actual testing framework
    
    # Create a test dataframe
    df = UnitedDataframe.create_empty(
        column_keys=["temperature", "pressure"],
        units=[SimpleUnit.parse("K"), SimpleUnit.parse("Pa")],
        column_types=[ColumnType.REAL_NUMBER_64, ColumnType.REAL_NUMBER_64]
    )
    
    # Test new property names
    assert df.column_count == 2
    assert df.row_count == 0
    
    # Test that old names still work
    assert df.cols == 2  # Should be same as column_count
    assert df.rows == 0  # Should be same as row_count
    
    # Add some test data
    df.add_empty_rows(3)
    df.cell_set(0, "temperature", RealUnitedScalar.create(273.15, SimpleUnit.parse("K")))
    df.cell_set(0, "pressure", RealUnitedScalar.create(101325, SimpleUnit.parse("Pa")))
    
    # Test new method names work
    temp_value = df.cell_get(0, "temperature")
    assert temp_value is not None
    
    # Test row operations
    first_rows = df.rows_head(2)
    assert first_rows.row_count == 2
    
    # Test mask operations  
    na_mask = df.mask_is_na()
    assert na_mask is not None
    
    print("✅ All new naming convention tests passed!")


if __name__ == "__main__":
    test_new_naming_conventions()
'''
    
    return code


def show_quick_start_guide():
    """Display the quick start guide."""
    
    print("🚀 UnitedSystem Naming Convention Quick Start")
    print("=" * 50)
    print()
    
    print("📋 STEP 1: Add New Methods (Phase 1)")
    print("-" * 30)
    print("Copy the following code and add it to your UnitedDataframe class:")
    print()
    print(generate_united_dataframe_additions())
    print()
    
    print("📋 STEP 2: Update Type Variables")
    print("-" * 30)
    print("Update your type variable definitions:")
    print()
    print(generate_type_variable_updates())
    print()
    
    print("📋 STEP 3: Test Everything Works")
    print("-" * 30)
    print("Use this test code to verify the changes:")
    print()
    print(create_testing_code())
    print()
    
    print("🎯 NEXT STEPS:")
    print("1. Implement the code additions above")
    print("2. Run your existing tests to ensure backward compatibility")
    print("3. Test the new method names work correctly")
    print("4. Start using the new names in new code")
    print("5. Plan for Phase 2 (deprecation warnings) in next version")
    print()
    
    print("✅ Benefits you'll see immediately:")
    print("• Cleaner, more intuitive method names")
    print("• Better IDE autocomplete (methods grouped by prefix)")
    print("• More professional API design")
    print("• 100% backward compatibility")
    print("• Easier to discover functionality")


if __name__ == "__main__":
    show_quick_start_guide() 