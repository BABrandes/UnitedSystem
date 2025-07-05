#!/usr/bin/env python3
"""
Concrete Example: Refactoring UnitedDataframe Methods

This shows the actual before/after refactoring for key methods
from your UnitedDataframe class.
"""

import warnings
from typing import TypeVar, Generic, Literal
import pandas as pd
import numpy as np

# Updated type variables
ColumnKeyType = TypeVar("ColumnKeyType", bound="ColumnKey|str")

class UnitedDataframeRefactored(Generic[ColumnKeyType]):
    """Refactored UnitedDataframe with improved naming conventions."""
    
    def __init__(self):
        # Cleaner private attribute names
        self._data: pd.DataFrame = pd.DataFrame()  # was: _internal_canonical_dataframe
        self._column_info: dict = {}               # was: _column_information
        self._read_lock = None                     # was: _rlock
        self._write_lock = None                    # was: _wlock
    
    # =============================================================================
    # PROPERTY IMPROVEMENTS
    # =============================================================================
    
    @property
    def column_count(self) -> int:
        """Get the number of columns in the dataframe."""
        with self._read_lock:
            return len(self._data.columns)
    
    @property
    def row_count(self) -> int:
        """Get the number of rows in the dataframe."""
        with self._read_lock:
            return len(self._data)
    
    # Backward compatibility with deprecation warnings
    @property
    def cols(self) -> int:
        """.. deprecated:: 2.0.0 Use `column_count` instead."""
        warnings.warn(
            "Property 'cols' is deprecated. Use 'column_count' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.column_count
    
    @property
    def rows(self) -> int:
        """.. deprecated:: 2.0.0 Use `row_count` instead."""
        warnings.warn(
            "Property 'rows' is deprecated. Use 'row_count' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.row_count

    # =============================================================================
    # COLUMN OPERATIONS - BEFORE & AFTER
    # =============================================================================
    
    def column_sum(self, column_key: ColumnKeyType) -> "RealUnitedScalar":
        """
        Calculate the sum of a numeric column.
        
        Args:
            column_key: The column to sum
            
        Returns:
            The sum with appropriate unit information
            
        Raises:
            ValueError: If the column doesn't exist or is not numeric
        """
        with self._read_lock:
            if not self.has_column(column_key):
                raise ValueError(f"Column '{column_key}' does not exist.")
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_name: str = self._get_internal_column_name(column_key)
            values: pd.Series = self._data[column_name]
            return RealUnitedScalar.create(
                np.sum(values), 
                self._get_display_unit(column_key)
            )
    
    def column_mean(self, column_key: ColumnKeyType) -> "RealUnitedScalar":
        """Calculate the mean of a numeric column."""
        with self._read_lock:
            if not self.has_column(column_key):
                raise ValueError(f"Column '{column_key}' does not exist.")
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_name: str = self._get_internal_column_name(column_key)
            values: pd.Series = self._data[column_name]
            return RealUnitedScalar.create(
                np.mean(values),
                self._get_display_unit(column_key)
            )
    
    def column_std(self, column_key: ColumnKeyType) -> "RealUnitedScalar":
        """Calculate the standard deviation of a numeric column."""
        with self._read_lock:
            if not self.has_column(column_key):
                raise ValueError(f"Column '{column_key}' does not exist.")
            if not self.is_numeric(column_key):
                raise ValueError(f"Column '{column_key}' is not numeric.")
            
            column_name: str = self._get_internal_column_name(column_key)
            values: pd.Series = self._data[column_name]
            return RealUnitedScalar.create(
                np.std(values),
                self._get_display_unit(column_key)
            )
    
    def column_as_numpy(self, column_key: ColumnKeyType, unit=None) -> np.ndarray:
        """
        Get a column as a numpy array in specified units.
        
        Args:
            column_key: The column to convert
            unit: The units to return the data in
            
        Returns:
            Column data as numpy array in specified units
        """
        with self._read_lock:
            if not self.has_column(column_key):
                raise ValueError(f"Column '{column_key}' does not exist.")
            
            column_name: str = self._get_internal_column_name(column_key)
            column_type = self._get_column_type(column_key)
            
            if not column_type.has_unit and unit is not None:
                raise ValueError(f"Column '{column_key}' has no unit but unit was specified.")
            if column_type.has_unit and unit is None:
                raise ValueError(f"Column '{column_key}' requires a unit to be specified.")
            
            values = self._data[column_name].to_numpy()
            
            if unit is not None:
                # Convert from canonical to specified unit
                unit_quantity = self._get_unit_quantity(column_key)
                if unit.unit_quantity != unit_quantity:
                    raise ValueError(f"Unit {unit} incompatible with column unit {unit_quantity}.")
                values = unit.from_canonical_value(values)
            
            return values
    
    def column_as_pandas(self, column_key: ColumnKeyType, unit=None) -> pd.Series:
        """Get a column as a pandas Series in specified units."""
        array = self.column_as_numpy(column_key, unit)
        return pd.Series(array, name=str(column_key))
    
    # Backward compatibility methods with deprecation warnings
    def colfun_sum(self, column_key: ColumnKeyType) -> "RealUnitedScalar":
        """.. deprecated:: 2.0.0 Use `column_sum` instead."""
        warnings.warn(
            "Method 'colfun_sum' is deprecated. Use 'column_sum' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.column_sum(column_key)
    
    def colfun_mean(self, column_key: ColumnKeyType) -> "RealUnitedScalar":
        """.. deprecated:: 2.0.0 Use `column_mean` instead."""
        warnings.warn(
            "Method 'colfun_mean' is deprecated. Use 'column_mean' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.column_mean(column_key)
    
    def column_values_as_numpy_array(self, column_key: ColumnKeyType, in_units=None) -> np.ndarray:
        """.. deprecated:: 2.0.0 Use `column_as_numpy` instead."""
        warnings.warn(
            "Method 'column_values_as_numpy_array' is deprecated. Use 'column_as_numpy' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.column_as_numpy(column_key, in_units)

    # =============================================================================
    # ROW OPERATIONS - IMPROVED NAMING
    # =============================================================================
    
    def rows_head(self, n: int = 5) -> "UnitedDataframeRefactored":
        """Get the first n rows of the dataframe."""
        with self._read_lock:
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            
            head_data = self._data.head(n)
            return self._create_with_new_data(head_data)
    
    def rows_tail(self, n: int = 5) -> "UnitedDataframeRefactored":
        """Get the last n rows of the dataframe.""" 
        with self._read_lock:
            if n < 0:
                raise ValueError(f"n must be non-negative, got {n}")
            
            tail_data = self._data.tail(n)
            return self._create_with_new_data(tail_data)
    
    def row_get(self, index: int) -> dict[ColumnKeyType, "UnitedScalar"]:
        """Get a single row as a dictionary of column keys to values."""
        with self._read_lock:
            if not 0 <= index < len(self._data):
                raise ValueError(f"Row index {index} out of bounds.")
            
            return {
                column_key: self.cell_get(index, column_key) 
                for column_key in self._get_column_keys()
            }
    
    def row_add(self, values: dict[ColumnKeyType, "UnitedScalar"]) -> None:
        """Add a new row to the dataframe."""
        with self._write_lock:
            if self._read_only:
                raise ValueError("Dataframe is read-only.")
            
            # Validate all column keys exist
            for column_key in values.keys():
                if not self.has_column(column_key):
                    raise ValueError(f"Column '{column_key}' does not exist.")
            
            # Add the row (implementation details would go here)
            # ...
    
    # Backward compatibility
    def rowfun_head(self, n: int = 5) -> "UnitedDataframeRefactored":
        """.. deprecated:: 2.0.0 Use `rows_head` instead."""
        warnings.warn(
            "Method 'rowfun_head' is deprecated. Use 'rows_head' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.rows_head(n)

    # =============================================================================
    # MASK OPERATIONS - CLEANER NAMING 
    # =============================================================================
    
    def mask_is_na(self, subset: list[ColumnKeyType] = None) -> np.ndarray:
        """
        Return a boolean mask indicating which values are NA/NaN.
        
        Args:
            subset: List of columns to check. If None, checks all columns.
            
        Returns:
            Boolean mask where True indicates NA values
        """
        with self._read_lock:
            if subset is None:
                return self._data.isna().values
            else:
                column_names = [self._get_internal_column_name(col) for col in subset]
                return self._data[column_names].isna().values
    
    def mask_not_na(self, subset: list[ColumnKeyType] = None) -> np.ndarray:
        """Return a boolean mask indicating which values are not NA/NaN."""
        return ~self.mask_is_na(subset)
    
    def filter_by_values(self, conditions: dict[ColumnKeyType, "UnitedScalar"]) -> "UnitedDataframeRefactored":
        """
        Filter rows by exact value matches.
        
        Args:
            conditions: Dictionary mapping column keys to required values
            
        Returns:
            New dataframe with filtered rows
        """
        with self._read_lock:
            filtered_data = self._data.copy()
            
            for column_key, value in conditions.items():
                if not self.has_column(column_key):
                    raise ValueError(f"Column '{column_key}' does not exist.")
                
                column_name = self._get_internal_column_name(column_key)
                column_type = self._get_column_type(column_key)
                dataframe_value = column_type.cast_for_dataframe(value)
                
                filtered_data = filtered_data[
                    filtered_data[column_name] == dataframe_value
                ]
            
            return self._create_with_new_data(filtered_data)
    
    # Backward compatibility
    def maskfun_isna(self, subset=None) -> np.ndarray:
        """.. deprecated:: 2.0.0 Use `mask_is_na` instead."""
        warnings.warn(
            "Method 'maskfun_isna' is deprecated. Use 'mask_is_na' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.mask_is_na(subset)

    # =============================================================================
    # CELL OPERATIONS - SIMPLIFIED NAMING
    # =============================================================================
    
    def cell_get(self, row: int, column: ColumnKeyType) -> "UnitedScalar":
        """Get the value of a specific cell."""
        with self._read_lock:
            if not self.has_column(column):
                raise ValueError(f"Column '{column}' does not exist.")
            if not 0 <= row < len(self._data):
                raise ValueError(f"Row {row} out of bounds.")
            
            column_name = self._get_internal_column_name(column)
            raw_value = self._data.at[row, column_name]
            
            if pd.isna(raw_value):
                return None  # or appropriate NA representation
            
            # Convert to appropriate UnitedScalar type
            return self._convert_to_united_scalar(raw_value, column)
    
    def cell_set(self, row: int, column: ColumnKeyType, value: "UnitedScalar") -> None:
        """Set the value of a specific cell."""
        with self._write_lock:
            if self._read_only:
                raise ValueError("Dataframe is read-only.")
            if not self.has_column(column):
                raise ValueError(f"Column '{column}' does not exist.")
            if not 0 <= row < len(self._data):
                raise ValueError(f"Row {row} out of bounds.")
            
            # Validate value compatibility
            if not self._check_value_compatibility(column, value):
                raise ValueError(f"Value incompatible with column '{column}'.")
            
            column_name = self._get_internal_column_name(column)
            column_type = self._get_column_type(column)
            dataframe_value = column_type.cast_for_dataframe(value)
            
            self._data.at[row, column_name] = dataframe_value
    
    def cell_is_empty(self, row: int, column: ColumnKeyType) -> bool:
        """Check if a cell contains NA/NaN value."""
        with self._read_lock:
            if not self.has_column(column):
                raise ValueError(f"Column '{column}' does not exist.")
            if not 0 <= row < len(self._data):
                raise ValueError(f"Row {row} out of bounds.")
            
            column_name = self._get_internal_column_name(column)
            return pd.isna(self._data.at[row, column_name])
    
    # Backward compatibility
    def cell_value_get(self, row: int, column: ColumnKeyType) -> "UnitedScalar":
        """.. deprecated:: 2.0.0 Use `cell_get` instead."""
        warnings.warn(
            "Method 'cell_value_get' is deprecated. Use 'cell_get' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.cell_get(row, column)
    
    def cell_value_set(self, row: int, column: ColumnKeyType, value: "UnitedScalar") -> None:
        """.. deprecated:: 2.0.0 Use `cell_set` instead."""
        warnings.warn(
            "Method 'cell_value_set' is deprecated. Use 'cell_set' instead.",
            DeprecationWarning, stacklevel=2
        )
        return self.cell_set(row, column, value)
    
    # =============================================================================
    # HELPER METHODS (implementation details)
    # =============================================================================
    
    def _get_internal_column_name(self, column_key: ColumnKeyType) -> str:
        """Get internal pandas column name for a column key."""
        # Implementation would go here
        pass
    
    def _get_column_type(self, column_key: ColumnKeyType):
        """Get column type information."""
        # Implementation would go here  
        pass
    
    def _create_with_new_data(self, data: pd.DataFrame) -> "UnitedDataframeRefactored":
        """Create new UnitedDataframe instance with different data."""
        # Implementation would go here
        pass


# =============================================================================
# COMPARISON: BEFORE vs AFTER API
# =============================================================================

def compare_old_vs_new_api():
    """Demonstrate the improved API clarity."""
    
    print("🔴 OLD API (inconsistent, unclear):")
    print("df.cols                           # abbreviated property")
    print("df.colfun_sum('temperature')      # prefix + abbreviation")  
    print("df.column_values_as_numpy_array() # verbose")
    print("df.rowfun_head(10)               # prefix + function")
    print("df.maskfun_isna()                # prefix + abbreviation")
    print("df.cell_value_get(0, 'temp')     # verbose")
    print()
    
    print("🟢 NEW API (consistent, clear):")
    print("df.column_count                  # clear property name")
    print("df.column_sum('temperature')     # consistent pattern")
    print("df.column_as_numpy()             # concise conversion")
    print("df.rows_head(10)                 # clear row operation")
    print("df.mask_is_na()                  # clear mask operation")
    print("df.cell_get(0, 'temp')           # simple cell access")
    print()
    
    print("✅ Benefits of new naming:")
    print("• Consistent patterns across all operations")
    print("• Better IDE autocomplete (grouped by prefix)")
    print("• Clearer intent and functionality")
    print("• Easier to remember and discover")
    print("• Professional API design")


if __name__ == "__main__":
    compare_old_vs_new_api() 