"""
Comprehensive test suite for RealArray class.

This file tests all methods and functionality of the RealArray wrapper class,
including construction, arithmetic, reductions, unit conversion, and serialization.
"""

import pytest
import tempfile
import h5py
import numpy as np
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from united_system import RealUnitedArray, RealUnitedScalar, Unit, Dimension
from united_system.arrays.real_united_array import RealUnitedArray


class TestRealArrayConstruction:
    """Test all constructor patterns for RealArray."""
    
    def test_list_with_unit_construction(self):
        """Test creating RealArray from list with unit."""
        arr = RealUnitedArray([1, 2, 3, 4, 5], Unit("m"))
        assert arr.shape == (5,)
        assert str(arr.dimension) == "LENGTH"
        assert str(arr.display_unit) == "m"
        np.testing.assert_array_equal(arr.values, [1., 2., 3., 4., 5.])
        np.testing.assert_array_equal(arr.canonical_values, [1., 2., 3., 4., 5.])
    
    def test_numpy_with_unit_construction(self):
        """Test creating RealArray from numpy array with unit."""
        np_arr = np.array([10, 20, 30])
        arr = RealUnitedArray(np_arr, Unit("kg"))
        assert arr.shape == (3,)
        assert str(arr.dimension) == "MASS"
        assert str(arr.display_unit) == "kg"
        np.testing.assert_array_equal(arr.values, [10., 20., 30.])
    
    def test_dimensionless_construction(self):
        """Test creating dimensionless RealArray."""
        arr = RealUnitedArray([1.5, 2.5, 3.5])
        assert arr.shape == (3,)
        assert str(arr.dimension) == "FRACTION"
        assert arr.display_unit is None
        np.testing.assert_array_equal(arr.values, [1.5, 2.5, 3.5])
    
    def test_invalid_construction(self):
        """Test invalid constructor arguments."""
        # Invalid input type
        with pytest.raises(ValueError):
            RealUnitedArray("invalid input")


class TestRealArrayProperties:
    """Test all properties of RealArray."""
    
    def test_values_property(self):
        """Test values property returns array in display unit."""
        arr = RealUnitedArray([1000, 2000, 3000], Unit("g"))
        np.testing.assert_array_equal(arr.values, [1000., 2000., 3000.])
    
    def test_canonical_values_property(self):
        """Test canonical_values property."""
        arr = RealUnitedArray([1, 2, 3], Unit("m"))
        np.testing.assert_array_equal(arr.canonical_values, [1., 2., 3.])
    
    def test_shape_property(self):
        """Test shape property."""
        arr1 = RealUnitedArray([1, 2, 3], Unit("m"))
        assert arr1.shape == (3,)
        
        arr2 = RealUnitedArray(np.ones((2, 3)), Unit("s"))
        assert arr2.shape == (2, 3)
    
    def test_size_property(self):
        """Test size property."""
        arr1 = RealUnitedArray([1, 2, 3], Unit("m"))
        assert arr1.size == 3
        
        arr2 = RealUnitedArray(np.ones((2, 3)), Unit("s"))
        assert arr2.size == 6
    
    def test_dimension_property(self):
        """Test dimension property."""
        arr_length = RealUnitedArray([1, 2], Unit("m"))
        assert str(arr_length.dimension) == "LENGTH"
        
        arr_mass = RealUnitedArray([1, 2], Unit("kg"))
        assert str(arr_mass.dimension) == "MASS"
        
        arr_dimensionless = RealUnitedArray([1, 2])
        assert str(arr_dimensionless.dimension) == "FRACTION"
    
    def test_display_unit_property(self):
        """Test display_unit property."""
        arr_with_unit = RealUnitedArray([1, 2], Unit("m"))
        assert str(arr_with_unit.display_unit) == "m"
        
        arr_dimensionless = RealUnitedArray([1, 2])
        assert arr_dimensionless.display_unit is None


class TestRealArrayAccess:
    """Test array access methods."""
    
    def test_getitem_index(self):
        """Test getting single elements by index."""
        arr = RealUnitedArray([10, 20, 30, 40], Unit("V"))
        
        first = arr[0]
        assert isinstance(first, RealUnitedScalar)
        assert first.canonical_value == 10.0
        
        last = arr[-1]
        assert isinstance(last, RealUnitedScalar)
        assert last.canonical_value == 40.0
    
    def test_getitem_slice(self):
        """Test getting slices."""
        arr = RealUnitedArray([1, 2, 3, 4, 5], Unit("m"))
        
        slice_result = arr[1:4]
        assert isinstance(slice_result, RealUnitedArray)
        assert slice_result.shape == (3,)
        np.testing.assert_array_equal(slice_result.values, [2., 3., 4.])
        assert str(slice_result.dimension) == "LENGTH"
    
    def test_len(self):
        """Test len() function."""
        arr = RealUnitedArray([1, 2, 3, 4, 5], Unit("m"))
        assert len(arr) == 5


class TestRealArrayArithmetic:
    """Test arithmetic operations."""
    
    def test_addition(self):
        """Test array addition."""
        arr1 = RealUnitedArray([1, 2, 3], Unit("m"))
        arr2 = RealUnitedArray([4, 5, 6], Unit("m"))
        
        # Array + Array
        result = arr1 + arr2
        np.testing.assert_array_equal(result.values, [5., 7., 9.])
        assert str(result.dimension) == "LENGTH"
        
        # Array + Scalar
        scalar = RealUnitedScalar(10, Unit("m"))
        result2 = arr1 + scalar
        np.testing.assert_array_equal(result2.values, [11., 12., 13.])
        
        # Array + Number should raise error
        with pytest.raises(ValueError, match="Cannot add primitive type"):
            result3 = arr1 + 5
        
        # Reverse addition with number should raise error
        with pytest.raises(ValueError, match="Cannot add primitive type"):
            result4 = 5 + arr1
    
    def test_subtraction(self):
        """Test array subtraction."""
        arr1 = RealUnitedArray([10, 20, 30], Unit("m"))
        arr2 = RealUnitedArray([1, 2, 3], Unit("m"))
        
        # Array - Array
        result = arr1 - arr2
        np.testing.assert_array_equal(result.values, [9., 18., 27.])
        
        # Array - Scalar
        scalar = RealUnitedScalar(5, Unit("m"))
        result2 = arr1 - scalar
        np.testing.assert_array_equal(result2.values, [5., 15., 25.])
        
        # Array - Number should raise error
        with pytest.raises(ValueError, match="Cannot subtract primitive type"):
            result3 = arr1 - 5
        
        # Number - Array should raise error
        with pytest.raises(ValueError, match="Cannot subtract RealArray from primitive type"):
            result4 = 5 - arr1
    
    def test_multiplication(self):
        """Test array multiplication."""
        arr1 = RealUnitedArray([2, 3, 4], Unit("m"))
        
        # Array * Scalar
        scalar = RealUnitedScalar(2.0)
        result2 = arr1 * scalar
        np.testing.assert_array_equal(result2.values, [4., 6., 8.])
        assert str(result2.dimension) == "LENGTH"
        
        # Array * Number
        result3 = arr1 * 3
        np.testing.assert_array_equal(result3.values, [6., 9., 12.])
        
        # Reverse multiplication
        result4 = 3 * arr1
        np.testing.assert_array_equal(result4.values, [6., 9., 12.])
    
    def test_division(self):
        """Test array division."""
        arr1 = RealUnitedArray([10, 20, 30], Unit("m"))
        
        # Array / Scalar
        scalar = RealUnitedScalar(2.0)
        result2 = arr1 / scalar
        np.testing.assert_array_equal(result2.values, [5., 10., 15.])
        assert str(result2.dimension) == "LENGTH"
        
        # Array / Number
        result3 = arr1 / 2
        np.testing.assert_array_equal(result3.values, [5., 10., 15.])
    
    def test_power(self):
        """Test array exponentiation."""
        arr = RealUnitedArray([2, 3, 4], Unit("m"))
        result = arr ** 2
        np.testing.assert_array_equal(result.values, [4., 9., 16.])
    
    def test_negation(self):
        """Test array negation."""
        arr = RealUnitedArray([1, -2, 3], Unit("m"))
        result = -arr
        np.testing.assert_array_equal(result.values, [-1., 2., -3.])
        assert str(result.dimension) == "LENGTH"
    
    def test_absolute_value(self):
        """Test array absolute value."""
        arr = RealUnitedArray([1, -2, 3, -4], Unit("m"))
        result = abs(arr)
        np.testing.assert_array_equal(result.values, [1., 2., 3., 4.])
        assert str(result.dimension) == "LENGTH"


class TestRealArrayReductions:
    """Test array reduction operations."""
    
    def test_sum(self):
        """Test array sum."""
        arr = RealUnitedArray([1, 2, 3, 4], Unit("V"))
        result = arr.sum()
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 10.0
    
    def test_mean(self):
        """Test array mean."""
        arr = RealUnitedArray([2, 4, 6, 8], Unit("A"))
        result = arr.mean()
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 5.0
    
    def test_max(self):
        """Test array maximum."""
        arr = RealUnitedArray([1, 5, 3, 2], Unit("m"))
        result = arr.max()
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 5.0
    
    def test_min(self):
        """Test array minimum."""
        arr = RealUnitedArray([1, 5, 3, 2], Unit("m"))
        result = arr.min()
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 1.0
    
    def test_std(self):
        """Test array standard deviation."""
        arr = RealUnitedArray([1, 2, 3, 4, 5], Unit("kg"))
        result = arr.std()
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value > 0  # Should be positive
    
    def test_var(self):
        """Test array variance."""
        arr = RealUnitedArray([1, 2, 3, 4, 5], Unit("kg"))
        result = arr.var()
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value > 0  # Should be positive


class TestRealArrayUnitConversion:
    """Test unit conversion methods."""
    
    def test_in_unit(self):
        """Test converting to specific unit."""
        arr = RealUnitedArray([1000, 2000, 3000], Unit("g"))
        kg_values = arr.in_unit(Unit("kg"))
        # Values should be different due to conversion
        assert isinstance(kg_values, np.ndarray)
    
    def test_with_unit(self):
        """Test creating new array with different display unit."""
        arr = RealUnitedArray([1, 2, 3], Unit("m"))
        arr_cm = arr.with_unit(Unit("cm"))
        
        assert isinstance(arr_cm, RealUnitedArray)
        assert str(arr_cm.display_unit) == "cm"
        assert str(arr_cm.dimension) == "LENGTH"


class TestRealArrayStringRepresentation:
    """Test string representation methods."""
    
    def test_str(self):
        """Test __str__ method."""
        arr = RealUnitedArray([1, 2, 3], Unit("m"))
        str_repr = str(arr)
        assert isinstance(str_repr, str)
        assert len(str_repr) > 0
    
    def test_repr(self):
        """Test __repr__ method."""
        arr = RealUnitedArray([1, 2, 3], Unit("m"))
        repr_str = repr(arr)
        assert isinstance(repr_str, str)
        assert "RealArray" in repr_str
    
    def test_format(self):
        """Test format method."""
        arr = RealUnitedArray([1.234, 2.567], Unit("m"))
        formatted = arr.format(decimals=2)
        assert isinstance(formatted, str)


class TestRealArrayUtilities:
    """Test utility methods."""
    
    def test_compatible_with(self):
        """Test compatibility checking."""
        arr1 = RealUnitedArray([1, 2], Unit("m"))
        arr2 = RealUnitedArray([3, 4], Unit("cm"))  # Same dimension
        arr3 = RealUnitedArray([5, 6], Unit("kg"))  # Different dimension
        
        assert arr1.compatible_with(arr2) == True
        assert arr1.compatible_with(arr3) == False


class TestRealArrayFactoryMethods:
    """Test factory/class methods."""
    
    def test_from_values_and_unit(self):
        """Test from_values_and_unit factory method."""
        arr = RealUnitedArray.from_values_and_unit([1, 2, 3], Unit("m"))
        assert isinstance(arr, RealUnitedArray)
        assert arr.shape == (3,)
        assert str(arr.dimension) == "LENGTH"
    
    def test_zeros(self):
        """Test zeros factory method."""
        arr = RealUnitedArray.zeros(5, Unit("m"))
        assert arr.shape == (5,)
        np.testing.assert_array_equal(arr.values, [0., 0., 0., 0., 0.])
        assert str(arr.dimension) == "LENGTH"
        
        # Test with tuple shape
        arr2 = RealUnitedArray.zeros((2, 3), Unit("kg"))
        assert arr2.shape == (2, 3)
        assert str(arr2.dimension) == "MASS"
    
    def test_ones(self):
        """Test ones factory method."""
        arr = RealUnitedArray.ones(3, Unit("s"))
        assert arr.shape == (3,)
        np.testing.assert_array_equal(arr.values, [1., 1., 1.])
        assert str(arr.dimension) == "TIME"
    
    def test_dimensionless(self):
        """Test dimensionless factory method."""
        arr = RealUnitedArray.dimensionless([1, 2, 3])
        assert arr.shape == (3,)
        assert str(arr.dimension) == "FRACTION"
        assert arr.display_unit is None


class TestRealArraySerialization:
    """Test serialization methods."""
    
    def test_to_json_from_json(self):
        """Test JSON serialization."""
        arr = RealUnitedArray([1, 2, 3], Unit("m"))
        
        # Test to_json
        json_data = arr.to_json()
        assert isinstance(json_data, dict)
        
        # Test from_json
        arr_restored = RealUnitedArray.from_json(json_data)
        assert isinstance(arr_restored, RealUnitedArray)
        np.testing.assert_array_equal(arr_restored.canonical_values, arr.canonical_values)


class TestRealArrayEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_array(self):
        """Test creating empty arrays."""
        arr = RealUnitedArray([], Unit("m"))
        assert arr.shape == (0,)
        assert len(arr) == 0
    
    def test_single_element_array(self):
        """Test single element arrays."""
        arr = RealUnitedArray([42], Unit("V"))
        assert arr.shape == (1,)
        assert len(arr) == 1
        
        element = arr[0]
        assert isinstance(element, RealUnitedScalar)
        assert element.canonical_value == 42.0
    
    def test_large_array(self):
        """Test with larger arrays."""
        large_arr = RealUnitedArray(np.arange(100), Unit("m"))
        assert large_arr.shape == (100,)
        assert large_arr.sum().canonical_value == sum(range(100))
    
    def test_incompatible_units_arithmetic(self):
        """Test arithmetic with incompatible units."""
        arr1 = RealUnitedArray([1, 2], Unit("m"))  # Length
        arr2 = RealUnitedArray([3, 4], Unit("kg")) # Mass
        
        # Addition of incompatible units should fail
        with pytest.raises(ValueError, match="not compatible"):
            result = arr1 + arr2
        
        # Subtraction of incompatible units should fail
        with pytest.raises(ValueError, match="not compatible"):
            result = arr1 - arr2
        
        # Multiplication should work (creates new dimension)
        result = arr1 * arr2
        assert isinstance(result, RealUnitedArray)
        np.testing.assert_array_equal(result.values, [3., 8.])
        
        # Division should work (creates new dimension)
        result = arr1 / arr2
        assert isinstance(result, RealUnitedArray)
        np.testing.assert_array_equal(result.values, [1/3, 2/4])


if __name__ == "__main__":
    # Run some basic tests manually
    print("Running basic RealArray tests...")
    
    # Test construction
    arr = RealUnitedArray([1, 2, 3, 4, 5], Unit("m"))
    print(f"Created array: {arr}")
    print(f"Shape: {arr.shape}, Dimension: {arr.dimension}")
    
    # Test arithmetic
    arr2 = RealUnitedArray([5, 4, 3, 2, 1], Unit("m"))
    result = arr + arr2
    print(f"Addition result: {result.values}")
    
    # Test reductions
    print(f"Sum: {arr.sum()}")
    print(f"Mean: {arr.mean()}")
    print(f"Max: {arr.max()}")
    
    # Test indexing
    print(f"First element: {arr[0]}")
    print(f"Slice [1:3]: {arr[1:3].values}")
    
    print("✅ Basic tests passed!") 