# UnitedSystem Naming Conventions Guide

## 🎯 **Overall Philosophy**
- **Clarity over brevity**: `column_count` over `cols`
- **Consistency**: Same patterns across all modules
- **Domain-specific**: Reflect the unit-aware nature
- **Pandas-familiar**: Similar to pandas where appropriate

## 📋 **1. Method Naming Patterns**

### **Column Operations**
```python
# OLD (inconsistent)
colfun_sum()          # prefix + abbreviated
column_values_as_array()  # full descriptive

# NEW (consistent)
column_sum()          # clear, concise
column_mean()
column_std()
column_min()
column_max()
column_as_array()
column_as_numpy()
column_as_pandas()
column_unique_values()
```

### **Row Operations**
```python
# OLD
rowfun_head()
rowfun_tail()
rowfun_last()

# NEW
rows_head()
rows_tail()
rows_last()
rows_first()
row_get()             # for single row
row_add()
row_remove()
```

### **Mask/Filter Operations**
```python
# OLD
maskfun_isna()
maskfun_notna()
filterfun_by_filterdict()

# NEW
mask_is_na()
mask_not_na()
mask_from_conditions()
filter_by_values()
filter_by_condition()
```

### **Cell Operations**
```python
# OLD
cell_value_get()
cell_value_set()
cell_value_is_empty()

# NEW
cell_get()
cell_set()
cell_is_empty()
```

## 📋 **2. Property Naming**

### **Current Issues & Fixes**
```python
# OLD
@property
def cols(self) -> int:  # abbreviated

@property  
def rows(self) -> int:  # abbreviated

# NEW
@property
def column_count(self) -> int:  # clear, descriptive

@property
def row_count(self) -> int:    # clear, descriptive

@property
def shape(self) -> tuple[int, int]:  # keep - matches pandas
```

## 📋 **3. Private Attribute Naming**

### **Consistent Patterns**
```python
# Internal data storage
_data                    # instead of _internal_canonical_dataframe
_column_info            # instead of _column_information
_column_name_formatter  # instead of _internal_dataframe_column_name_formatter

# Locks (keep short for frequent use)
_read_lock              # instead of _rlock  
_write_lock             # instead of _wlock
_lock                   # main lock object

# Derived attributes
_column_keys
_unit_quantities
_display_units
_column_types
```

## 📋 **4. Type Variable Naming**

### **More Descriptive Type Variables**
```python
# OLD
CK = TypeVar("CK", bound=ColumnKey|str)
CK_I2 = TypeVar("CK_I2", bound=ColumnKey|str)  # cryptic
CK_CF = TypeVar("CK_CF", bound=ColumnKey|str)  # cryptic

# NEW
ColumnKeyType = TypeVar("ColumnKeyType", bound=ColumnKey|str)
ColumnKeyType2 = TypeVar("ColumnKeyType2", bound=ColumnKey|str) 
FilterColumnKeyType = TypeVar("FilterColumnKeyType", bound=ColumnKey|str)

# OR keep short but meaningful
CKey = TypeVar("CKey", bound=ColumnKey|str)
CKey2 = TypeVar("CKey2", bound=ColumnKey|str)
FilterCKey = TypeVar("FilterCKey", bound=ColumnKey|str)
```

## 📋 **5. Class & Module Naming**

### **Consistent Patterns**
```python
# Core classes (keep current - they're good)
UnitedDataframe
UnitedArray  
UnitedScalar
RealUnitedScalar
ComplexUnitedScalar

# Helper classes
ColumnInformation    # clear
ColumnType          # clear
UnitQuantity        # clear

# Internal classes
_ColumnAccessor     # good
_RowAccessor        # good
_GroupBy            # could be GroupByOperator
```

## 📋 **6. Function/Method Categories**

### **Organized by Purpose**
```python
# Data Access
def column_get(key) -> UnitedArray
def row_get(index) -> dict
def cell_get(row, col) -> UnitedScalar

# Data Modification  
def column_set(key, values)
def row_add(values)
def cell_set(row, col, value)

# Data Analysis
def column_sum(key) -> UnitedScalar
def column_mean(key) -> UnitedScalar
def column_describe() -> pd.DataFrame

# Data Filtering
def filter_by_values(conditions) -> UnitedDataframe
def mask_from_condition(func) -> np.ndarray

# Data Conversion
def to_pandas() -> pd.DataFrame
def to_numpy(unit) -> np.ndarray
def from_pandas(df, metadata) -> UnitedDataframe
```

## 📋 **7. Specific Recommendations**

### **High Priority Renames**
1. `colfun_*` → `column_*`
2. `rowfun_*` → `rows_*` or `row_*`
3. `maskfun_*` → `mask_*`
4. `cols` → `column_count`
5. `rows` → `row_count`

### **Medium Priority**
1. Long private attributes → shorter equivalents
2. Type variables → more descriptive
3. Method parameter names → consistent patterns

### **Implementation Strategy**
1. Create aliases for backward compatibility
2. Add deprecation warnings
3. Update documentation
4. Gradual migration over versions 