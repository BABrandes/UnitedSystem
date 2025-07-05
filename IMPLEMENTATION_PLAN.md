# UnitedSystem Naming Convention Implementation Plan

## 🎯 **Executive Summary**

This plan outlines a **systematic, backward-compatible refactoring** of the UnitedSystem codebase to implement consistent, professional naming conventions. The refactoring will be done in **3 phases** to ensure no breaking changes for existing users.

## 📋 **Phase 1: Foundation (Immediate - No Breaking Changes)**

### **1.1 Add New Methods Alongside Old Ones**

#### **UnitedDataframe Class** (`src/united_system/united_dataframe/united_dataframe.py`)

```python
# NEW METHODS (add these alongside existing ones)

# Column operations
def column_sum(self, column_key: CK) -> RealUnitedScalar:
    """Calculate the sum of a column."""
    return self.colfun_sum(column_key)  # delegate to existing implementation

def column_mean(self, column_key: CK) -> RealUnitedScalar:
    """Calculate the mean of a column.""" 
    return self.colfun_mean(column_key)

def column_std(self, column_key: CK) -> RealUnitedScalar:
    """Calculate the standard deviation of a column."""
    return self.colfun_std(column_key)

def column_min(self, column_key: CK) -> RealUnitedScalar:
    """Get the minimum value of a column."""
    return self.colfun_min(column_key)

def column_max(self, column_key: CK) -> RealUnitedScalar:
    """Get the maximum value of a column."""
    return self.colfun_max(column_key)

def column_as_numpy(self, column_key: CK, unit=None) -> np.ndarray:
    """Get column as numpy array in specified units."""
    return self.column_values_as_numpy_array(column_key, unit)

def column_as_pandas(self, column_key: CK, unit=None) -> pd.Series:
    """Get column as pandas Series in specified units."""
    return self.column_values_as_pandas_series(column_key, unit)

def column_as_array(self, column_key: CK) -> UnitedArray:
    """Get column as UnitedArray."""
    return self.column_values_as_array(column_key)

# Row operations  
def rows_head(self, n: int = 5) -> "UnitedDataframe[CK]":
    """Get the first n rows."""
    return self.rowfun_head(n)

def rows_tail(self, n: int = 5) -> "UnitedDataframe[CK]":
    """Get the last n rows."""
    return self.rowfun_tail(n)

def row_get(self, index: int) -> dict[CK, UnitedScalar]:
    """Get a single row."""
    return self.row(index)

# Mask operations
def mask_is_na(self, subset=None) -> np.ndarray:
    """Return mask indicating NA values."""
    return self.maskfun_isna(subset)

def mask_not_na(self, subset=None) -> np.ndarray:
    """Return mask indicating non-NA values."""
    return self.maskfun_notna(subset)

def filter_by_values(self, conditions: dict) -> "UnitedDataframe[CK]":
    """Filter rows by value conditions."""
    return self.filterfun_by_filterdict(conditions)

# Cell operations
def cell_get(self, row: int, column: CK) -> UnitedScalar:
    """Get cell value."""
    return self.cell_value_get(row, column)

def cell_set(self, row: int, column: CK, value: UnitedScalar) -> None:
    """Set cell value."""
    return self.cell_value_set(row, column, value)

def cell_is_empty(self, row: int, column: CK) -> bool:
    """Check if cell is empty."""
    return self.cell_value_is_empty(row, column)

# Properties
@property
def column_count(self) -> int:
    """Get number of columns."""
    return self.cols

@property  
def row_count(self) -> int:
    """Get number of rows."""
    return self.rows
```

### **1.2 Update Type Variables**

```python
# In all relevant files, add cleaner type variables
ColumnKeyType = TypeVar("ColumnKeyType", bound=ColumnKey|str)
FilterColumnKeyType = TypeVar("FilterColumnKeyType", bound=ColumnKey|str)

# Keep old ones for backward compatibility
CK = ColumnKeyType  # alias
CK_CF = FilterColumnKeyType  # alias
```

### **1.3 Update Private Attribute Names (Internal Only)**

```python
# Add shorter aliases for long attribute names (internal use only)
@property
def _data(self) -> pd.DataFrame:
    """Shorter alias for internal dataframe."""
    return self._internal_canonical_dataframe

@property
def _column_info(self) -> dict:
    """Shorter alias for column information."""
    return self._column_information
```

## 📋 **Phase 2: Deprecation Warnings (Version 2.0)**

### **2.1 Add Deprecation Warnings to Old Methods**

```python
import warnings

def colfun_sum(self, column_key: CK) -> RealUnitedScalar:
    """
    Calculate the sum of a column.
    
    .. deprecated:: 2.0.0
        Use `column_sum` instead. This method will be removed in v3.0.0.
    """
    warnings.warn(
        "Method 'colfun_sum' is deprecated and will be removed in v3.0.0. "
        "Use 'column_sum' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.column_sum(column_key)

def cols(self) -> int:
    """
    Get number of columns.
    
    .. deprecated:: 2.0.0
        Use `column_count` instead. This property will be removed in v3.0.0.
    """
    warnings.warn(
        "Property 'cols' is deprecated and will be removed in v3.0.0. "
        "Use 'column_count' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return self.column_count
```

### **2.2 Create Migration Script**

```python
#!/usr/bin/env python3
"""
Migration script to help users update their code to use new naming conventions.
"""

import ast
import re
from pathlib import Path

REPLACEMENTS = {
    # Properties
    '.cols': '.column_count',
    '.rows': '.row_count',
    
    # Column methods
    '.colfun_sum(': '.column_sum(',
    '.colfun_mean(': '.column_mean(',
    '.colfun_std(': '.column_std(',
    '.colfun_min(': '.column_min(',
    '.colfun_max(': '.column_max(',
    '.column_values_as_numpy_array(': '.column_as_numpy(',
    '.column_values_as_pandas_series(': '.column_as_pandas(',
    '.column_values_as_array(': '.column_as_array(',
    
    # Row methods
    '.rowfun_head(': '.rows_head(',
    '.rowfun_tail(': '.rows_tail(',
    '.rowfun_last(': '.rows_last(',
    
    # Mask methods
    '.maskfun_isna(': '.mask_is_na(',
    '.maskfun_notna(': '.mask_not_na(',
    '.filterfun_by_filterdict(': '.filter_by_values(',
    
    # Cell methods
    '.cell_value_get(': '.cell_get(',
    '.cell_value_set(': '.cell_set(',
    '.cell_value_is_empty(': '.cell_is_empty(',
}

def migrate_file(file_path: Path) -> None:
    """Migrate a single Python file to use new naming conventions."""
    content = file_path.read_text()
    original_content = content
    
    for old, new in REPLACEMENTS.items():
        content = content.replace(old, new)
    
    if content != original_content:
        file_path.write_text(content)
        print(f"✅ Migrated: {file_path}")
    else:
        print(f"⚪ No changes: {file_path}")

def migrate_directory(directory: Path) -> None:
    """Migrate all Python files in a directory."""
    for py_file in directory.rglob("*.py"):
        migrate_file(py_file)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python migrate.py <directory_or_file>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    if path.is_file():
        migrate_file(path)
    else:
        migrate_directory(path)
```

### **2.3 Update Documentation**

```markdown
# UnitedSystem v2.0 Migration Guide

## New Naming Conventions

UnitedSystem v2.0 introduces cleaner, more consistent naming conventions:

### Properties
- `df.cols` → `df.column_count` 
- `df.rows` → `df.row_count`

### Column Operations  
- `df.colfun_sum()` → `df.column_sum()`
- `df.colfun_mean()` → `df.column_mean()`
- `df.column_values_as_numpy_array()` → `df.column_as_numpy()`

### Row Operations
- `df.rowfun_head()` → `df.rows_head()`
- `df.rowfun_tail()` → `df.rows_tail()`

### Mask Operations
- `df.maskfun_isna()` → `df.mask_is_na()`
- `df.maskfun_notna()` → `df.mask_not_na()`

### Cell Operations
- `df.cell_value_get()` → `df.cell_get()`
- `df.cell_value_set()` → `df.cell_set()`

## Migration

1. **Automatic Migration**: Use the provided migration script
2. **Manual Migration**: Update method calls as shown above
3. **Gradual Migration**: Old methods still work but show deprecation warnings

## Timeline

- **v2.0**: New methods available, old methods deprecated
- **v3.0**: Old methods removed (breaking change)
```

## 📋 **Phase 3: Cleanup (Version 3.0 - Breaking Changes)**

### **3.1 Remove Deprecated Methods**

```python
# Remove all methods marked as deprecated in Phase 2
# This includes:
# - colfun_* methods
# - rowfun_* methods  
# - maskfun_* methods
# - cell_value_* methods
# - abbreviated properties (cols, rows)
```

### **3.2 Update All Internal Code**

```python
# Replace all internal usage of old method names
# Update all documentation examples
# Update all test cases
# Update all examples in docstrings
```

### **3.3 Simplify Type Variables**

```python
# Remove old type variable aliases
# Standardize on new descriptive names
ColumnKeyType = TypeVar("ColumnKeyType", bound=ColumnKey|str)
FilterColumnKeyType = TypeVar("FilterColumnKeyType", bound=ColumnKey|str)
```

## 🛠️ **Implementation Checklist**

### **Immediate Actions (Phase 1)**
- [ ] Add new method aliases to UnitedDataframe class
- [ ] Add new property aliases  
- [ ] Update type variable definitions
- [ ] Test that all new methods work correctly
- [ ] Ensure 100% backward compatibility

### **Version 2.0 Release (Phase 2)**
- [ ] Add deprecation warnings to all old methods
- [ ] Create and test migration script
- [ ] Update documentation with migration guide
- [ ] Create examples using new API
- [ ] Update README with new method names

### **Version 3.0 Release (Phase 3)**
- [ ] Remove all deprecated methods
- [ ] Update all internal code to use new names
- [ ] Update all tests
- [ ] Update all documentation
- [ ] Verify no breaking changes beyond removed deprecated methods

## 🔍 **Quality Assurance**

### **Testing Strategy**
```python
def test_naming_convention_compatibility():
    """Test that both old and new naming conventions work."""
    df = UnitedDataframe.create_empty()
    
    # Test new names work
    assert hasattr(df, 'column_count')
    assert hasattr(df, 'column_sum')
    assert hasattr(df, 'rows_head')
    
    # Test old names still work (with warnings)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = df.cols  # Should trigger deprecation warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
```

### **Documentation Review**
- [ ] All method signatures documented
- [ ] Migration guide complete
- [ ] Examples updated
- [ ] API reference updated
- [ ] Breaking changes clearly noted

## 📈 **Benefits of This Approach**

1. **Zero Breaking Changes** in Phase 1 & 2
2. **Gradual Migration** path for users
3. **Professional API** design
4. **Better IDE Support** with consistent prefixes
5. **Easier Maintenance** with cleaner code
6. **Future-Proof** naming conventions

## 🚀 **Success Metrics**

- **Backward Compatibility**: 100% of existing code continues to work
- **Adoption Rate**: Track usage of new vs old method names
- **User Feedback**: Positive response to cleaner API
- **Documentation Quality**: Clear migration path and examples
- **Code Quality**: Consistent naming across entire codebase

---

*This implementation plan ensures a smooth transition to professional naming conventions while maintaining the excellent functionality of the UnitedSystem library.* 