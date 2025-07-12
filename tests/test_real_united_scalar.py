"""
Comprehensive tests for RealUnitedScalar class.

Tests all mixins and functionality including:
- Core functionality
- Arithmetic operations
- Comparison operations
- Unit conversions
- Formatting
- Factory methods
- Utility methods
- Serialization
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import json
import h5py

# Import the modules to test
from united_system.real_united_scalar import RealUnitedScalar
from united_system.dimension import Dimension
from united_system.unit import Unit
from united_system.named_dimensions import NamedDimension


class TestRealUnitedScalarCore:
    """Test core functionality of RealUnitedScalar."""
    
    def test_initialization_with_dimension(self):
        """Test initialization with Dimension object."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])  # Mass dimension
        scalar = RealUnitedScalar(5.0, dimension)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == dimension
        assert scalar._display_unit is None
    
    def test_initialization_with_named_dimension(self):
        """Test initialization with NamedDimension object."""
        from src.united_system.named_dimensions import NamedDimension
        mass_dim = NamedDimension.MASS
        scalar = RealUnitedScalar(5.0, mass_dim)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == mass_dim.dimension
    
    def test_initialization_with_display_unit(self):
        """Test initialization with display unit."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        unit = Unit.parse_string("kg")
        scalar = RealUnitedScalar(5.0, dimension, unit)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == dimension
        assert scalar._display_unit == unit
    
    def test_invalid_display_unit(self):
        """Test that incompatible display unit raises ValueError."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])  # Mass
        unit = Unit.parse_string("m")  # Length - incompatible
        
        with pytest.raises(ValueError, match="not compatible"):
            RealUnitedScalar(5.0, dimension, unit)
    
    def test_display_unit_property(self):
        """Test display_unit property."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        unit = Unit.parse_string("kg")
        scalar = RealUnitedScalar(5.0, dimension, unit)
        
        assert scalar.display_unit == unit
    
    def test_display_unit_property_none(self):
        """Test display_unit property when _display_unit is None."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar(5.0, dimension)
        
        # Should return canonical unit
        assert scalar.display_unit == dimension.canonical_unit
    
    def test_active_unit_property(self):
        """Test active_unit property."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        unit = Unit.parse_string("kg")
        scalar = RealUnitedScalar(5.0, dimension, unit)
        
        assert scalar.active_unit == unit
    
    def test_active_unit_property_none(self):
        """Test active_unit property when _display_unit is None."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar(5.0, dimension)
        
        assert scalar.active_unit == dimension.canonical_unit
    
    def test_active_float_property(self):
        """Test active_float property."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        unit = Unit.parse_string("kg")
        scalar = RealUnitedScalar(5.0, dimension, unit)
        
        # active_float should be canonical_value * unit.factor + unit.offset
        expected = 5.0 * unit.factor + unit.offset
        assert scalar.active_float == expected


class TestRealUnitedScalarArithmetic:
    """Test arithmetic operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        self.mass1 = RealUnitedScalar(5.0, self.mass_dim)
        self.mass2 = RealUnitedScalar(3.0, self.mass_dim)
    
    def test_addition_same_dimension(self):
        """Test addition of scalars with same dimension."""
        result = self.mass1 + self.mass2
        assert result.canonical_value == 8.0
        assert result.dimension == self.mass_dim
    
    def test_addition_different_dimensions(self):
        """Test that addition with different dimensions raises error."""
        length = RealUnitedScalar(2.0, self.length_dim)
        
        with pytest.raises(ValueError):
            self.mass1 + length
    
    def test_subtraction_same_dimension(self):
        """Test subtraction of scalars with same dimension."""
        result = self.mass1 - self.mass2
        assert result.canonical_value == 2.0
        assert result.dimension == self.mass_dim
    
    def test_subtraction_different_dimensions(self):
        """Test that subtraction with different dimensions raises error."""
        length = RealUnitedScalar(2.0, self.length_dim)
        
        with pytest.raises(ValueError):
            self.mass1 - length
    
    def test_multiplication(self):
        """Test multiplication of scalars."""
        result = self.mass1 * self.mass2
        # Mass * Mass = Mass^2
        expected_dim = Dimension.create([2, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result.canonical_value == 15.0
        assert result.dimension == expected_dim
    
    def test_division(self):
        """Test division of scalars."""
        result = self.mass1 / self.mass2
        # Mass / Mass = dimensionless
        expected_dim = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result.canonical_value == pytest.approx(5.0 / 3.0)
        assert result.dimension == expected_dim
    
    def test_power(self):
        """Test power operation."""
        result = self.mass1 ** 2
        expected_dim = Dimension.create([2, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result.canonical_value == 25.0
        assert result.dimension == expected_dim
    
    def test_negation(self):
        """Test negation."""
        result = -self.mass1
        assert result.canonical_value == -5.0
        assert result.dimension == self.mass_dim


class TestRealUnitedScalarComparison:
    """Test comparison operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.mass1 = RealUnitedScalar(5.0, self.mass_dim)
        self.mass2 = RealUnitedScalar(3.0, self.mass_dim)
        self.mass3 = RealUnitedScalar(5.0, self.mass_dim)
    
    def test_equality(self):
        """Test equality comparison."""
        assert self.mass1 == self.mass3
        assert self.mass1 != self.mass2
    
    def test_less_than(self):
        """Test less than comparison."""
        assert self.mass2 < self.mass1
        assert not self.mass1 < self.mass2
    
    def test_less_equal(self):
        """Test less than or equal comparison."""
        assert self.mass2 <= self.mass1
        assert self.mass1 <= self.mass3
        assert not self.mass1 <= self.mass2
    
    def test_greater_than(self):
        """Test greater than comparison."""
        assert self.mass1 > self.mass2
        assert not self.mass2 > self.mass1
    
    def test_greater_equal(self):
        """Test greater than or equal comparison."""
        assert self.mass1 >= self.mass2
        assert self.mass1 >= self.mass3
        assert not self.mass2 >= self.mass1
    
    def test_comparison_different_dimensions(self):
        """Test that comparison with different dimensions raises error."""
        length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        length = RealUnitedScalar(2.0, length_dim)
        
        with pytest.raises(ValueError):
            self.mass1 < length


class TestRealUnitedScalarConversion:
    """Test unit conversion operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.kg_unit = Unit.parse_string("kg")
        self.g_unit = Unit.parse_string("g")
        self.scalar = RealUnitedScalar(1.0, self.mass_dim, self.kg_unit)
    
    def test_in_unit(self):
        """Test conversion to different unit."""
        result = self.scalar.in_unit(self.g_unit)
        assert result.canonical_value == 1.0  # Same canonical value
        assert result.dimension == self.mass_dim
        assert result._display_unit == self.g_unit
    
    def test_in_unit_incompatible(self):
        """Test conversion to incompatible unit raises error."""
        m_unit = Unit.parse_string("m")
        
        with pytest.raises(ValueError):
            self.scalar.in_unit(m_unit)
    
    def test_to_canonical(self):
        """Test conversion to canonical unit."""
        result = self.scalar.to_canonical()
        assert result.canonical_value == 1.0
        assert result.dimension == self.mass_dim
        assert result._display_unit is None
    
    def test_value_in_unit(self):
        """Test getting value in specific unit."""
        value = self.scalar.value_in_unit(self.g_unit)
        assert value == 1000.0  # 1 kg = 1000 g
    
    def test_value_in_unit_incompatible(self):
        """Test getting value in incompatible unit raises error."""
        m_unit = Unit.parse_string("m")
        
        with pytest.raises(ValueError):
            self.scalar.value_in_unit(m_unit)


class TestRealUnitedScalarFormatting:
    """Test formatting operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.kg_unit = Unit.parse_string("kg")
        self.scalar = RealUnitedScalar(1.0, self.mass_dim, self.kg_unit)
    
    def test_str_representation(self):
        """Test string representation."""
        result = str(self.scalar)
        assert "1.0" in result
        assert "kg" in result
    
    def test_repr_representation(self):
        """Test repr representation."""
        result = repr(self.scalar)
        assert "RealUnitedScalar" in result
        assert "1.0" in result
    
    def test_format_method(self):
        """Test format method."""
        result = self.scalar.format("g", decimals=2)
        assert "1000.00" in result
        assert "g" in result
    
    def test_format_method_invalid_unit(self):
        """Test format method with invalid unit."""
        with pytest.raises(ValueError):
            self.scalar.format("m", decimals=2)


class TestRealUnitedScalarFactory:
    """Test factory methods."""
    
    def test_create_from_value_and_unit(self):
        """Test create_from_value_and_unit factory method."""
        kg_unit = Unit.parse_string("kg")
        scalar = RealUnitedScalar.create_from_value_and_unit(5.0, kg_unit)
        
        assert scalar.canonical_value == 5.0 * kg_unit.factor
        assert scalar.dimension == kg_unit.dimension
        assert scalar._display_unit == kg_unit
    
    def test_parse_string(self):
        """Test parse_string factory method."""
        scalar = RealUnitedScalar.parse_string("5.0 kg")
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == Unit.parse_string("kg").dimension
        assert scalar._display_unit == Unit.parse_string("kg")
    
    def test_parse_string_complex(self):
        """Test parse_string with complex units."""
        scalar = RealUnitedScalar.parse_string("10.5 m/s^2")
        
        expected_dim = Dimension.create([0, -2, 1, 0, 0, 0, 0], [0, 0])
        assert scalar.dimension == expected_dim
    
    def test_parse_string_invalid(self):
        """Test parse_string with invalid string."""
        with pytest.raises(ValueError):
            RealUnitedScalar.parse_string("invalid string")
    
    def test_zero(self):
        """Test zero factory method."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar.zero(mass_dim)
        
        assert scalar.canonical_value == 0.0
        assert scalar.dimension == mass_dim
    
    def test_one(self):
        """Test one factory method."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar.one(mass_dim)
        
        assert scalar.canonical_value == 1.0
        assert scalar.dimension == mass_dim


class TestRealUnitedScalarUtility:
    """Test utility methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.positive = RealUnitedScalar(5.0, self.mass_dim)
        self.negative = RealUnitedScalar(-3.0, self.mass_dim)
        self.zero = RealUnitedScalar(0.0, self.mass_dim)
    
    def test_is_positive(self):
        """Test is_positive method."""
        assert self.positive.is_positive()
        assert not self.negative.is_positive()
        assert not self.zero.is_positive()
    
    def test_is_negative(self):
        """Test is_negative method."""
        assert not self.positive.is_negative()
        assert self.negative.is_negative()
        assert not self.zero.is_negative()
    
    def test_is_zero(self):
        """Test is_zero method."""
        assert not self.positive.is_zero()
        assert not self.negative.is_zero()
        assert self.zero.is_zero()
    
    def test_abs(self):
        """Test abs method."""
        abs_negative = abs(self.negative)
        assert abs_negative.canonical_value == 3.0
        assert abs_negative.dimension == self.mass_dim
    
    def test_clamp(self):
        """Test clamp method."""
        clamped = self.positive.clamp(0.0, 3.0)
        assert clamped.canonical_value == 3.0
        
        clamped = self.negative.clamp(-5.0, 0.0)
        assert clamped.canonical_value == -3.0
    
    def test_sum(self):
        """Test sum method."""
        scalars = [self.positive, self.negative, self.zero]
        total = RealUnitedScalar.sum(scalars)
        
        assert total.canonical_value == 2.0
        assert total.dimension == self.mass_dim
    
    def test_mean(self):
        """Test mean method."""
        scalars = [self.positive, self.negative, self.zero]
        avg = RealUnitedScalar.mean(scalars)
        
        assert avg.canonical_value == pytest.approx(2.0 / 3.0)
        assert avg.dimension == self.mass_dim
    
    def test_sum_empty_list(self):
        """Test sum with empty list."""
        with pytest.raises(ValueError):
            RealUnitedScalar.sum([])
    
    def test_mean_empty_list(self):
        """Test mean with empty list."""
        with pytest.raises(ValueError):
            RealUnitedScalar.mean([])


class TestRealUnitedScalarSerialization:
    """Test serialization methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.kg_unit = Unit.parse_string("kg")
        self.scalar = RealUnitedScalar(5.0, self.mass_dim, self.kg_unit)
    
    def test_to_json(self):
        """Test to_json method."""
        json_data = self.scalar.to_json()
        
        assert "canonical_value" in json_data
        assert "dimension" in json_data
        assert "display_unit" in json_data
        assert json_data["canonical_value"] == 5.0
    
    def test_from_json(self):
        """Test from_json method."""
        json_data = {
            "canonical_value": 5.0,
            "dimension": self.mass_dim.to_json(),
            "display_unit": self.kg_unit.to_json()
        }
        
        scalar = RealUnitedScalar.from_json(json_data)
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == self.mass_dim
        assert scalar._display_unit == self.kg_unit
    
    def test_to_hdf5(self):
        """Test to_hdf5 method."""
        with h5py.File("test_scalar.h5", "w") as f:
            group = f.create_group("scalar")
            self.scalar.to_hdf5(group)
            
            # Verify data was written
            assert "canonical_value" in group
            assert "dimension" in group
            assert "display_unit" in group
    
    def test_from_hdf5(self):
        """Test from_hdf5 method."""
        # First write to file
        with h5py.File("test_scalar.h5", "w") as f:
            group = f.create_group("scalar")
            self.scalar.to_hdf5(group)
        
        # Then read from file
        with h5py.File("test_scalar.h5", "r") as f:
            group = f["scalar"]
            scalar = RealUnitedScalar.from_hdf5(group)
            
            assert scalar.canonical_value == 5.0
            assert scalar.dimension == self.mass_dim
            assert scalar._display_unit == self.kg_unit


class TestRealUnitedScalarEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_dimensionless_scalar(self):
        """Test dimensionless scalar."""
        dimensionless_dim = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar(1.0, dimensionless_dim)
        
        assert scalar.canonical_value == 1.0
        assert scalar.dimension == dimensionless_dim
    
    def test_very_large_value(self):
        """Test with very large value."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        large_value = 1e20
        scalar = RealUnitedScalar(large_value, mass_dim)
        
        assert scalar.canonical_value == large_value
    
    def test_very_small_value(self):
        """Test with very small value."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        small_value = 1e-20
        scalar = RealUnitedScalar(small_value, mass_dim)
        
        assert scalar.canonical_value == small_value
    
    def test_nan_value(self):
        """Test with NaN value."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar(float('nan'), mass_dim)
        
        assert np.isnan(scalar.canonical_value)
    
    def test_inf_value(self):
        """Test with infinity value."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        scalar = RealUnitedScalar(float('inf'), mass_dim)
        
        assert np.isinf(scalar.canonical_value)


if __name__ == "__main__":
    pytest.main([__file__])
