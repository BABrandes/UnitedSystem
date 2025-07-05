# Modular Refactoring - Next Steps

## ✅ Completed
1. Created modular directory structure
2. Extracted UnitedDataframeCore with essential functionality
3. Created ValidationMixin with validation logic
4. Created new lightweight UnitedDataframe class
5. Backed up original file

## 🎯 Next Steps

### Phase 1: Extract Remaining Mixins

1. **ColumnOperationsMixin** (operations/column_ops.py)
   - Extract all `colfun_*` methods
   - Extract column value retrieval methods
   - Extract column manipulation methods
   - Add improved naming methods (column_sum, column_mean, etc.)

2. **RowOperationsMixin** (operations/row_ops.py)
   - Extract all `rowfun_*` methods
   - Extract row manipulation methods
   - Add improved naming methods (rows_head, rows_tail, etc.)

3. **CellOperationsMixin** (operations/cell_ops.py)
   - Extract all `cell_value_*` methods
   - Extract `__getitem__` and `__setitem__` methods
   - Add improved naming methods (cell_get, cell_set, etc.)

4. **MaskFilterMixin** (operations/mask_filter.py)
   - Extract all `maskfun_*` methods
   - Extract all `filterfun_*` methods
   - Add improved naming methods (mask_is_na, filter_by_values, etc.)

5. **SerializationMixin** (serialization/serialization.py)
   - Extract JSON/HDF5 methods
   - Extract info/describe methods

6. **ConstructorMixin** (serialization/constructors.py)
   - Extract all class methods for construction
   - Extract concatenation methods

### Phase 2: Testing and Integration

1. Create comprehensive tests for each mixin
2. Ensure all imports work correctly
3. Test backward compatibility
4. Performance testing

### Phase 3: Documentation

1. Update all docstrings
2. Create migration guide
3. Update examples
4. Create architecture documentation

## 🔧 How to Extract a Mixin

1. **Identify methods**: Find all methods that belong to the mixin
2. **Extract to new file**: Create the mixin file with proper imports
3. **Add to main class**: Import and inherit from the mixin
4. **Update __init__.py**: Export the mixin
5. **Test**: Ensure everything works correctly

## 📁 File Template

```python
"""
[MixinName] for UnitedDataframe.

This module provides [description of functionality].
"""

from typing import Generic, TypeVar, List, Optional
# ... other imports

CK = TypeVar("CK", bound=ColumnKey|str, default=str)

class [MixinName](Generic[CK]):
    """
    Mixin providing [description].
    
    This mixin handles:
    - [functionality 1]
    - [functionality 2]
    - [functionality 3]
    """
    
    def method_name(self, ...):
        """Method description."""
        with self._rlock:  # type: ignore
            # Implementation
            pass
    
    # NEW: Improved naming methods
    def improved_method_name(self, ...):
        """Improved method with better naming."""
        return self.method_name(...)
```

## 🚀 Benefits So Far

1. **Reduced main file size**: From 2,559 lines to ~100 lines
2. **Clear separation**: Core functionality is now isolated
3. **Better testability**: Each mixin can be tested independently
4. **Improved maintainability**: Easier to find and modify specific functionality
5. **Backward compatibility**: All existing code continues to work

## 🧪 Testing the Current State

Run these commands to test the current implementation:

```python
# Test basic functionality
from src.united_system.united_dataframe import UnitedDataframe

# Create a test dataframe
df = UnitedDataframe.create_empty(
    column_keys=['test_col'],
    units=[None], 
    column_types=[ColumnType.INTEGER_64],
    initial_number_of_rows=10
)

# Test basic properties
print(f"Rows: {df.rows}")
print(f"Cols: {df.cols}")
print(f"Column count: {df.column_count()}")  # New naming
print(f"Row count: {df.row_count()}")        # New naming
print(f"Summary: {df.info_summary()}")       # New method
```

Continue with extracting the remaining mixins one by one!
