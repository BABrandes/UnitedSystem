# UnitedDataframe Modular Restructure Plan

## 🎯 **Current Problem**
- `united_dataframe.py`: **2,559 lines** - too large for maintainability
- Multiple responsibilities mixed together
- Difficult to navigate and modify
- Hard to test individual functionality

## 🏗️ **Proposed Solution: Mixin Architecture**

Break the monolithic class into focused mixins using multiple inheritance:

```python
class UnitedDataframe(
    UnitedDataframeCore,           # Base functionality
    ColumnOperationsMixin,          # Column operations
    RowOperationsMixin,            # Row operations  
    CellOperationsMixin,           # Cell operations
    MaskFilterMixin,               # Mask/filter operations
    SerializationMixin,            # JSON/HDF5 serialization
    ConstructorMixin,              # Factory methods
    ValidationMixin,               # Compatibility checking
    Generic[CK]
):
    """Unit-aware DataFrame with modular architecture."""
    pass
```

## 📁 **File Structure**

```
src/united_system/united_dataframe/
├── __init__.py                    # Export main class
├── united_dataframe.py           # Main class (reduced to ~100 lines)
├── core/
│   ├── __init__.py
│   ├── base.py                   # UnitedDataframeCore
│   ├── validation.py             # ValidationMixin
│   └── utils.py                  # Shared utilities
├── operations/
│   ├── __init__.py
│   ├── column_ops.py             # ColumnOperationsMixin
│   ├── row_ops.py                # RowOperationsMixin
│   ├── cell_ops.py               # CellOperationsMixin
│   └── mask_filter.py            # MaskFilterMixin
├── serialization/
│   ├── __init__.py
│   ├── serialization.py          # SerializationMixin
│   └── constructors.py           # ConstructorMixin
├── accessors/                    # Keep existing
│   ├── _column_accessor.py
│   ├── _row_accessor.py
│   └── _group.py
└── legacy/                       # Existing files
    ├── column_type.py
    └── utils.py
```

## 🔧 **Detailed Breakdown**

### **1. UnitedDataframeCore (core/base.py)**
**~200 lines** - Essential functionality
```python
class UnitedDataframeCore(JSONable, HDF5able, Generic[CK]):
    """Core functionality and state management."""
    
    # Core attributes
    _internal_canonical_dataframe: pd.DataFrame
    _column_information: dict[CK, ColumnInformation]
    _read_only: bool
    _lock: rwlock.RWLockFairD
    
    # Essential methods
    def __init__(...)
    def __len__(...)
    def copy(...)
    def is_read_only(...)
    def set_read_only(...)
    
    # Property accessors
    @property
    def cols(...)
    @property 
    def rows(...)
    @property
    def shape(...)
    @property
    def empty(...)
    
    # Lock management
    def acquire_read_lock(...)
    def acquire_write_lock(...)
```

### **2. ColumnOperationsMixin (operations/column_ops.py)**
**~400 lines** - All column-related operations
```python
class ColumnOperationsMixin:
    """Mixin for column operations and statistics."""
    
    # Column retrieval
    def column_values_as_numpy_array(...)
    def column_values_as_array(...)
    def column_values_as_pandas_series(...)
    
    # Column statistics  
    def colfun_sum(...)
    def colfun_mean(...)
    def colfun_std(...)
    def colfun_min(...)
    def colfun_max(...)
    def colfun_unique(...)
    
    # Column manipulation
    def add_empty_column(...)
    def remove_column(...)
    def rename_column(...)
    def set_column_values_from_numpy_array(...)
    
    # NEW: Improved naming methods
    def column_sum(...)     # delegates to colfun_sum
    def column_mean(...)    # delegates to colfun_mean
    def column_as_numpy(...)# delegates to column_values_as_numpy_array
```

### **3. RowOperationsMixin (operations/row_ops.py)**
**~300 lines** - All row-related operations
```python
class RowOperationsMixin:
    """Mixin for row operations and manipulation."""
    
    # Row access
    def row(...)
    def iterrows(...)
    def get_iterator_for_row(...)
    
    # Row manipulation
    def add_row(...)
    def insert_row(...)
    def remove_row(...)
    def remove_rows(...)
    def add_empty_rows(...)
    
    # Row functions
    def rowfun_head(...)
    def rowfun_tail(...)
    def rowfun_first(...)
    def rowfun_last(...)
    
    # NEW: Improved naming methods  
    def rows_head(...)      # delegates to rowfun_head
    def rows_tail(...)      # delegates to rowfun_tail
    def row_get(...)        # delegates to row
    def row_add(...)        # delegates to add_row
```

### **4. CellOperationsMixin (operations/cell_ops.py)**
**~200 lines** - Individual cell operations
```python
class CellOperationsMixin:
    """Mixin for individual cell operations."""
    
    # Cell access
    def cell_value_get(...)
    def cell_value_set(...)
    def cell_value_is_empty(...)
    
    # Magic methods for cell access
    def __getitem__(...)
    def __setitem__(...)
    
    # NEW: Improved naming methods
    def cell_get(...)       # delegates to cell_value_get  
    def cell_set(...)       # delegates to cell_value_set
    def cell_is_empty(...)  # delegates to cell_value_is_empty
```

### **5. MaskFilterMixin (operations/mask_filter.py)**
**~300 lines** - Filtering and masking operations
```python
class MaskFilterMixin:
    """Mixin for mask and filter operations."""
    
    # Mask operations
    def maskfun_isna(...)
    def maskfun_notna(...)
    def maskfun_get_from_filter(...)
    def maskfun_apply_mask(...)
    
    # Filter operations
    def filterfun_by_filterdict(...)
    def filterfun_get_by_column_key_types(...)
    
    # Utility operations
    def mask_in_range(...)
    def dropna(...)
    def sample(...)
    
    # NEW: Improved naming methods
    def mask_is_na(...)     # delegates to maskfun_isna
    def mask_not_na(...)    # delegates to maskfun_notna
    def filter_by_values(...)# delegates to filterfun_by_filterdict
```

### **6. SerializationMixin (serialization/serialization.py)**
**~100 lines** - JSON/HDF5 serialization
```python
class SerializationMixin:
    """Mixin for serialization and deserialization."""
    
    # JSON serialization
    def to_json(...)
    @classmethod
    def from_json(...)
    
    # HDF5 serialization  
    def to_hdf5(...)
    @classmethod
    def from_hdf5(...)
    
    # Utility methods
    def info(...)
    def describe(...)
```

### **7. ConstructorMixin (serialization/constructors.py)**
**~200 lines** - Factory methods and constructors  
```python
class ConstructorMixin:
    """Mixin for construction and factory methods."""
    
    @classmethod
    def create_from_pandas_dataframe(...)
    
    @classmethod  
    def create_from_pandas_dataframe_and_column_information(...)
    
    @classmethod
    def create_empty(...)
    
    def create_empty_from_column_information(...)
    def create_from_row_values_and_column_information(...)
    
    # Concatenation utilities
    @classmethod
    def dataframes_can_concatenate(...)
    @classmethod
    def concatenate_dataframes(...)
```

### **8. ValidationMixin (core/validation.py)**
**~150 lines** - Validation and compatibility
```python
class ValidationMixin:
    """Mixin for validation and compatibility checking."""
    
    # Column validation
    def has_column(...)
    def compatible_with_column(...)
    def is_numeric(...)
    def get_numeric_column_keys(...)
    
    # Internal validation
    def _check_scalar_compatibility(...)
    def _get_numpy_dtype_from_precision(...)
    
    # Type checking utilities
    def column_keys_of_type(...)
    def column_information_of_type(...)
```

## 🔧 **Implementation Strategy**

### **Phase 1: Extract Mixins (Week 1)**
1. Create the new directory structure
2. Extract each mixin into its own file
3. Ensure all imports work correctly
4. Add comprehensive tests for each mixin

### **Phase 2: Update Main Class (Week 2)**  
1. Update `united_dataframe.py` to inherit from all mixins
2. Add improved naming methods (column_*, rows_*, etc.)
3. Ensure backward compatibility
4. Update documentation

### **Phase 3: Add New Naming (Week 3)**
1. Implement the improved naming conventions in each mixin
2. Add deprecation warnings to old methods
3. Create migration documentation
4. Update examples and tests

## ✅ **Benefits of This Architecture**

1. **🔍 Maintainability**: Each file focuses on one responsibility
2. **🧪 Testability**: Each mixin can be tested independently  
3. **📖 Readability**: Much easier to find and understand specific functionality
4. **🔄 Reusability**: Mixins could be reused in other dataframe-like classes
5. **🚀 Performance**: No performance impact (same final class)
6. **🔒 Backward Compatibility**: All existing code continues to work
7. **📏 Size**: Each file becomes manageable (100-400 lines)

## 🎯 **File Size Reduction**

| Current | After Refactoring |
|---------|-------------------|
| `united_dataframe.py`: **2,559 lines** | `united_dataframe.py`: **~100 lines** |
| | `core/base.py`: **~200 lines** |
| | `operations/column_ops.py`: **~400 lines** |
| | `operations/row_ops.py`: **~300 lines** |  
| | `operations/cell_ops.py`: **~200 lines** |
| | `operations/mask_filter.py`: **~300 lines** |
| | `serialization/serialization.py`: **~100 lines** |
| | `serialization/constructors.py`: **~200 lines** |
| | `core/validation.py`: **~150 lines** |

## 🧪 **Testing Strategy**

```python
# Each mixin gets its own test file
tests/
├── test_column_operations.py      # Test ColumnOperationsMixin
├── test_row_operations.py         # Test RowOperationsMixin  
├── test_cell_operations.py        # Test CellOperationsMixin
├── test_mask_filter.py            # Test MaskFilterMixin
├── test_serialization.py          # Test SerializationMixin
├── test_constructors.py           # Test ConstructorMixin
├── test_validation.py             # Test ValidationMixin
└── test_integration.py            # Test full UnitedDataframe
```

## 🚀 **Next Steps**

1. **Create the modular structure** using the plan above
2. **Implement improved naming conventions** alongside the refactoring
3. **Add comprehensive tests** for each module
4. **Update documentation** to reflect the new structure
5. **Create migration guide** for developers

This modular architecture will make the codebase much more maintainable while providing an excellent foundation for implementing the improved naming conventions we discussed earlier! 