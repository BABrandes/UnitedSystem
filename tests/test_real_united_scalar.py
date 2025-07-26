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
- Mathematical operations (log, exp)
- Unit operations
"""

import pytest
import numpy as np
import math
import h5py
import tempfile
import os

# Import the modules to test
from src.united_system.real_united_scalar import RealUnitedScalar
from src.united_system.dimension import Dimension
from src.united_system.unit import Unit

class TestRealUnitedScalarCore:
    """Test core functionality of RealUnitedScalar."""
    
    def test_initialization_with_dimension(self):
        """Test initialization with Dimension object."""
        dimension: Dimension = Dimension("M")  # Mass dimension
        scalar = RealUnitedScalar(5.0, dimension)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == dimension
        assert scalar._display_unit is None  # type: ignore
    
    def test_initialization_with_named_quantity(self):
        """Test initialization with NamedQuantity object."""
        mass_dim = Dimension("M")
        scalar = RealUnitedScalar(5.0, mass_dim)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == mass_dim
    
    def test_initialization_with_unit(self):
        """Test initialization with Unit object."""
        unit = Unit("kg")
        scalar = RealUnitedScalar(5.0, unit)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == unit.dimension
        assert scalar.display_unit == unit
    
    def test_initialization_with_unit_string(self):
        """Test initialization with unit string."""
        scalar = RealUnitedScalar(5.0, "kg")
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == Unit("kg").dimension
        assert scalar.display_unit == Unit("kg")
    
    def test_initialization_with_string(self):
        """Test initialization with string containing value and unit."""
        scalar = RealUnitedScalar("5.0 kg")
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == Unit("kg").dimension
        assert scalar.display_unit == Unit("kg")
    
    def test_initialization_dimensionless(self):
        """Test initialization of dimensionless scalar."""
        scalar = RealUnitedScalar(42.0)
        
        assert scalar.canonical_value == 42.0
        assert scalar.dimension.is_dimensionless
        assert scalar._display_unit is None  # type: ignore
    
    def test_display_unit_property(self):
        """Test display_unit property."""
        unit = Unit("kg")
        scalar = RealUnitedScalar(5.0, unit)
        
        assert scalar.display_unit == unit
    
    def test_display_unit_property_none(self):
        """Test display_unit property when display_unit is None."""
        dimension = Dimension("M")
        scalar = RealUnitedScalar(5.0, dimension)
        
        # Should return canonical unit
        assert scalar.display_unit == dimension.canonical_unit
    
    def test_unit_property(self):
        """Test unit property."""
        unit = Unit("kg")
        scalar = RealUnitedScalar(5.0, unit)
        
        assert scalar.unit == unit
    
    def test_unit_property_none(self):
        """Test unit property when display_unit is None."""
        dimension = Dimension("M")
        scalar = RealUnitedScalar(5.0, dimension)
        
        assert scalar.unit == dimension.canonical_unit
    
    def test_active_float_property(self):
        """Test active_float property."""
        unit = Unit("kg")
        scalar = RealUnitedScalar(5.0, unit)
        
        # active_float should be the value in the display unit
        assert scalar.active_float == 5.0
    
    def test_compatible_to(self):
        """Test compatible_to method."""
        mass_dim = Dimension("M")
        length_dim = Dimension("L")
        
        mass1 = RealUnitedScalar(5.0, mass_dim)
        mass2 = RealUnitedScalar(3.0, mass_dim)
        length = RealUnitedScalar(2.0, length_dim)
        
        assert mass1.compatible_to(mass2)
        assert not mass1.compatible_to(length)
        assert mass1.compatible_to(mass1, mass2)  # Multiple arguments


class TestRealUnitedScalarArithmetic:
    """Test arithmetic operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension("M")
        self.length_dim = Dimension("L")
        self.mass1 = RealUnitedScalar.create_from_canonical_value(5.0, self.mass_dim)
        self.mass2 = RealUnitedScalar.create_from_canonical_value(3.0, self.mass_dim)
    
    def test_addition_same_dimension(self):
        """Test addition of scalars with same dimension."""
        result = self.mass1 + self.mass2
        assert result.canonical_value == 8.0
        assert result.dimension == self.mass_dim
    
    def test_addition_different_dimensions(self):
        """Test that addition with different dimensions raises error."""
        length = RealUnitedScalar.create_from_canonical_value(2.0, self.length_dim)
        
        with pytest.raises(ValueError):
            _ = self.mass1 + length
    
    def test_subtraction_same_dimension(self):
        """Test subtraction of scalars with same dimension."""
        result = self.mass1 - self.mass2
        assert result.canonical_value == 2.0
        assert result.dimension == self.mass_dim
    
    def test_subtraction_different_dimensions(self):
        """Test that subtraction with different dimensions raises error."""
        length = RealUnitedScalar.create_from_canonical_value(2.0, self.length_dim)
        
        with pytest.raises(ValueError):
            _ = self.mass1 - length
    
    def test_multiplication(self):
        """Test multiplication of scalars."""
        result = self.mass1 * self.mass2
        # Mass * Mass = Mass^2
        expected_dim = Dimension("M^2")
        assert result.canonical_value == 15.0
        assert result.dimension == expected_dim
    
    def test_division(self):
        """Test division of scalars."""
        result = self.mass1 / self.mass2
        # Mass / Mass = dimensionless
        expected_dim = Dimension("")
        assert result.canonical_value == pytest.approx(5.0 / 3.0)  # type: ignore
        assert result.dimension == expected_dim
    
    def test_power(self):
        """Test power operation."""
        result = self.mass1 ** 2
        expected_dim = Dimension("M^2")
        assert result.canonical_value == 25.0
        assert result.dimension == expected_dim
    
    def test_negation(self):
        """Test negation."""
        result = -self.mass1
        assert result.canonical_value == -5.0
        assert result.dimension == self.mass_dim
    
    def test_abs(self):
        """Test absolute value."""
        negative_mass = RealUnitedScalar.create_from_canonical_value(-3.0, self.mass_dim)
        result = abs(negative_mass)
        assert result.canonical_value == 3.0
        assert result.dimension == self.mass_dim


class TestRealUnitedScalarComparison:
    """Test comparison operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension("M")
        self.mass1 = RealUnitedScalar.create_from_canonical_value(5.0, self.mass_dim)
        self.mass2 = RealUnitedScalar.create_from_canonical_value(3.0, self.mass_dim)
        self.mass3 = RealUnitedScalar.create_from_canonical_value(5.0, self.mass_dim)
    
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
        length_dim = Dimension("L")
        length = RealUnitedScalar.create_from_canonical_value(2.0, length_dim)
        
        with pytest.raises(ValueError):
            _ = self.mass1 < length


class TestRealUnitedScalarConversion:
    """Test unit conversion operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension("M")
        self.kg_unit = Unit("kg")
        self.g_unit = Unit("g")
        self.scalar = RealUnitedScalar(1.0, self.kg_unit)
    
    def test_in_unit(self):
        """Test conversion to different unit."""
        result = self.scalar.in_unit(self.g_unit)
        assert result.canonical_value == 1.0  # Same canonical value
        assert result.dimension == self.mass_dim
        assert result.display_unit == self.g_unit
    
    def test_in_unit_incompatible(self):
        """Test conversion to incompatible unit raises error."""
        m_unit = Unit("m")
        
        with pytest.raises(ValueError):
            self.scalar.in_unit(m_unit)
    
    def test_to_canonical(self):
        """Test conversion to canonical unit."""
        result = self.scalar.to_canonical()
        assert result.canonical_value == 1.0
        assert result.dimension == self.mass_dim
        assert result._display_unit is None  # type: ignore
    
    def test_to_canonical_with_display_unit(self):
        """Test conversion to canonical unit with display unit set."""
        result = self.scalar.to_canonical(with_display_unit=True)
        assert result.canonical_value == 1.0
        assert result.dimension == self.mass_dim
        assert result.display_unit == self.mass_dim.canonical_unit
    
    def test_value_in_unit(self):
        """Test getting value in specific unit."""
        value = self.scalar.value_in_unit(self.g_unit)
        assert value == 1000.0  # 1 kg = 1000 g
    
    def test_value_in_unit_incompatible(self):
        """Test getting value in incompatible unit raises error."""
        m_unit = Unit("m")
        
        with pytest.raises(ValueError):
            self.scalar.value_in_unit(m_unit)
    
    def test_value_in_canonical_unit(self):
        """Test getting value in canonical unit."""
        value = self.scalar.value_in_canonical_unit()
        assert value == 1.0  # 1 kg in canonical units (kg)
    
    def test_value(self):
        """Test getting value in display unit."""
        value = self.scalar.value()
        assert value == 1.0  # 1 kg in display unit (kg)
    
    def test_value_no_display_unit(self):
        """Test getting value in display unit when no display unit is set."""
        # Create scalar without display unit
        scalar_no_display = RealUnitedScalar(1.0, self.mass_dim)
        
        with pytest.raises(ValueError, match="No display unit set"):
            scalar_no_display.value()
    
    def test_float_in_unit(self):
        """Test float_in_unit method."""
        value = self.scalar.float_in_unit(self.g_unit)
        assert value == 1000.0  # 1 kg = 1000 g
    
    def test_canonical_float(self):
        """Test canonical_float method."""
        value = self.scalar.canonical_float()
        assert value == 1.0  # 1 kg in canonical units
    
    def test_display_float(self):
        """Test display_float method."""
        value = self.scalar.display_float()
        assert value == 1.0  # 1 kg in display unit
    
    def test_display_float_no_display_unit(self):
        """Test display_float when no display unit is set."""
        scalar_no_display = RealUnitedScalar(1.0, self.mass_dim)
        
        with pytest.raises(ValueError, match="No display unit set"):
            scalar_no_display.display_float()

class TestRealUnitedScalarFormatting:
    """Test formatting operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.kg_unit = Unit("kg")
        self.scalar = RealUnitedScalar(1.0, self.kg_unit)
    
    def test_str_representation(self):
        """Test string representation."""
        result = str(self.scalar)
        assert "1.000" in result
        assert "kg" in result
    
    def test_repr_representation(self):
        """Test repr representation."""
        result = repr(self.scalar)
        assert "RealUnitedScalar" in result
        assert "1.0" in result
    
    def test_format_method(self):
        """Test format method."""
        result = self.scalar.format("g", max_decimals=2, trailing_zeros=True)
        assert "1000.00" in result
        assert "g" in result
    
    def test_format_method_invalid_unit(self):
        """Test format method with invalid unit."""
        with pytest.raises(ValueError):
            self.scalar.format("m", max_decimals=2)
    
    def test_simple_str_with_display_unit(self):
        """Test simple_str method with display unit."""
        result = self.scalar.simple_str()
        assert "1.0" in result
        assert "kg" in result
    
    def test_simple_str_without_display_unit(self):
        """Test simple_str method without display unit."""
        # Create scalar without display unit
        scalar_no_display = RealUnitedScalar.create_from_canonical_value(1.0, self.kg_unit.dimension)
        result = scalar_no_display.simple_str()
        assert "1.0" in result
        # Should have 'kg' as the canonical unit
        assert "kg" in result
    
    def test_simple_str_with_different_values(self):
        """Test simple_str with different numeric values."""
        # Test with zero
        zero_scalar = RealUnitedScalar(0.0, self.kg_unit)
        result = zero_scalar.simple_str()
        assert "0.0" in result
        
        # Test with negative value
        negative_scalar = RealUnitedScalar(-2.5, self.kg_unit)
        result = negative_scalar.simple_str()
        assert "-2.5" in result
        
        # Test with large value - shows the actual value in the display unit
        large_scalar = RealUnitedScalar(1000.0, self.kg_unit)
        result = large_scalar.simple_str()
        assert "1000.0" in result  # Shows 1000.0 kg
        assert "kg" in result


class TestRealUnitedScalarFactory:
    """Test factory methods."""
    
    def test_create_from_value_and_unit(self):
        """Test create_from_value_and_unit factory method."""
        kg_unit = Unit("kg")
        scalar = RealUnitedScalar.create_from_value_and_unit(5.0, kg_unit)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == kg_unit.dimension
        assert scalar.display_unit == kg_unit
    
    def test_parse_string(self):
        """Test parse_string factory method."""
        scalar = RealUnitedScalar.parse_string("5.0 kg")
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == Unit("kg").dimension
        assert scalar.display_unit == Unit("kg")
    
    def test_parse_string_complex(self):
        """Test parse_string with complex units."""
        scalar = RealUnitedScalar.parse_string("10.5 m/s^2")
        
        expected_dim = Dimension("L/T^2")
        assert scalar.dimension == expected_dim
    
    def test_parse_string_invalid(self):
        """Test parse_string with invalid string."""
        with pytest.raises(ValueError):
            RealUnitedScalar.parse_string("invalid string")
    
    def test_zero(self):
        """Test zero factory method."""
        mass_dim = Dimension("M")
        scalar = RealUnitedScalar.zero(mass_dim)
        
        assert scalar.canonical_value == 0.0
        assert scalar.dimension == mass_dim
    
    def test_one(self):
        """Test one factory method."""
        mass_dim = Dimension("M")
        scalar = RealUnitedScalar.one(mass_dim)
        
        assert scalar.canonical_value == 1.0
        assert scalar.dimension == mass_dim
    
    def test_create_dimensionless(self):
        """Test create_dimensionless factory method."""
        scalar = RealUnitedScalar.create_dimensionless(42.0)
        
        assert scalar.canonical_value == 42.0
        # Should be dimensionless
        expected_dim = Dimension("")
        assert scalar.dimension == expected_dim
    
    def test_create_from_canonical_value_with_unit(self):
        """Test create_from_canonical_value with unit."""
        mass_dim = Dimension("M")
        kg_unit = Unit("kg")
        scalar = RealUnitedScalar.create_from_canonical_value(5.0, mass_dim, kg_unit)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == mass_dim
        assert scalar.display_unit == kg_unit
    
    def test_create_from_canonical_value_without_unit(self):
        """Test create_from_canonical_value without unit."""
        mass_dim = Dimension("M")
        scalar = RealUnitedScalar.create_from_canonical_value(5.0, mass_dim)
        
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == mass_dim
        assert scalar._display_unit is None  # type: ignore
    
    def test_create_from_canonical_value_incompatible_unit(self):
        """Test create_from_canonical_value with incompatible unit."""
        mass_dim = Dimension("M")
        m_unit = Unit("m")  # Length unit, incompatible with mass
        
        with pytest.raises(ValueError, match="not compatible"):
            RealUnitedScalar.create_from_canonical_value(5.0, mass_dim, m_unit)


class TestRealUnitedScalarUtility:
    """Test utility methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension("M")
        self.positive = RealUnitedScalar.create_from_canonical_value(5.0, self.mass_dim)
        self.negative = RealUnitedScalar.create_from_canonical_value(-3.0, self.mass_dim)
        self.zero = RealUnitedScalar.create_from_canonical_value(0.0, self.mass_dim)
    
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
    
    def test_is_non_zero(self):
        """Test is_non_zero method."""
        assert self.positive.is_non_zero()
        assert self.negative.is_non_zero()
        assert not self.zero.is_non_zero()
    
    def test_is_nan(self):
        """Test is_nan method."""
        nan_scalar = RealUnitedScalar.create_from_canonical_value(float('nan'), self.mass_dim)
        assert nan_scalar.is_nan()
        assert not self.positive.is_nan()
    
    def test_is_finite(self):
        """Test is_finite method."""
        assert self.positive.is_finite()
        assert self.negative.is_finite()
        assert self.zero.is_finite()
        
        inf_scalar = RealUnitedScalar.create_from_canonical_value(float('inf'), self.mass_dim)
        assert not inf_scalar.is_finite()
        
        nan_scalar = RealUnitedScalar.create_from_canonical_value(float('nan'), self.mass_dim)
        assert not nan_scalar.is_finite()
    
    def test_is_infinite(self):
        """Test is_infinite method."""
        inf_scalar = RealUnitedScalar.create_from_canonical_value(float('inf'), self.mass_dim)
        assert inf_scalar.is_infinite()
        assert not self.positive.is_infinite()
    
    def test_is_positive_or_zero(self):
        """Test is_positive_or_zero method."""
        assert self.positive.is_positive_or_zero()
        assert not self.negative.is_positive_or_zero()
        assert self.zero.is_positive_or_zero()
    
    def test_is_negative_or_zero(self):
        """Test is_negative_or_zero method."""
        assert not self.positive.is_negative_or_zero()
        assert self.negative.is_negative_or_zero()
        assert self.zero.is_negative_or_zero()
    
    def test_is_strictly_positive(self):
        """Test is_strictly_positive method."""
        assert self.positive.is_strictly_positive()
        assert not self.negative.is_strictly_positive()
        assert not self.zero.is_strictly_positive()
    
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
        
        assert avg.canonical_value == pytest.approx(2.0 / 3.0)  # type: ignore
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
        self.mass_dim = Dimension("M")
        self.kg_unit = Unit("kg")
        self.scalar = RealUnitedScalar(5.0, self.kg_unit)
    
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
        assert scalar.display_unit == self.kg_unit
    
    def test_to_hdf5(self):
        """Test to_hdf5 method."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            try:
                with h5py.File(tmp_file.name, "w") as f:
                    group = f.create_group("scalar")  # type: ignore
                    self.scalar.to_hdf5(group)
                    
                    # Verify data was written
                    assert "canonical_value" in group
                    assert "dimension" in group
                    assert "display_unit" in group
            finally:
                os.unlink(tmp_file.name)
    
    def test_from_hdf5(self):
        """Test from_hdf5 method."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            try:
                # First write to file
                with h5py.File(tmp_file.name, "w") as f:
                    group = f.create_group("scalar")  # type: ignore
                    self.scalar.to_hdf5(group)
                
                # Then read from file
                with h5py.File(tmp_file.name, "r") as f:
                    group = f["scalar"]
                    scalar = RealUnitedScalar.from_hdf5(group)  # type: ignore
                    
                    assert scalar.canonical_value == 5.0
                    assert scalar.dimension == self.mass_dim
                    assert scalar.display_unit == self.kg_unit
            finally:
                os.unlink(tmp_file.name)


class TestRealUnitedScalarEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_dimensionless_scalar(self):
        """Test dimensionless scalar."""
        dimensionless_dim = Dimension("")
        scalar = RealUnitedScalar.create_from_canonical_value(1.0, dimensionless_dim)
        
        assert scalar.canonical_value == 1.0
        assert scalar.dimension == dimensionless_dim
    
    def test_very_large_value(self):
        """Test with very large value."""
        mass_dim = Dimension("M")
        large_value = 1e20
        scalar = RealUnitedScalar.create_from_canonical_value(large_value, mass_dim)
        
        assert scalar.canonical_value == large_value
    
    def test_very_small_value(self):
        """Test with very small value."""
        mass_dim = Dimension("M")
        small_value = 1e-20
        scalar = RealUnitedScalar.create_from_canonical_value(small_value, mass_dim)
        
        assert scalar.canonical_value == small_value
    
    def test_nan_value(self):
        """Test with NaN value."""
        mass_dim = Dimension("M")
        scalar = RealUnitedScalar.create_from_canonical_value(float('nan'), mass_dim)
        
        assert np.isnan(scalar.canonical_value)
    
    def test_inf_value(self):
        """Test with infinity value."""
        mass_dim = Dimension("M")
        scalar = RealUnitedScalar.create_from_canonical_value(float('inf'), mass_dim)
        
        assert np.isinf(scalar.canonical_value)


class TestRealUnitedScalarMathematical:
    """Test mathematical operations (logarithmic and exponential functions)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.kg_unit = Unit("kg")
        self.m_unit = Unit("m")
        self.s_unit = Unit("s")
        self.positive_scalar = RealUnitedScalar(10.0, self.kg_unit)
        self.negative_scalar = RealUnitedScalar(-5.0, self.kg_unit)
        self.zero_scalar = RealUnitedScalar(0.0, self.kg_unit)
        self.one_scalar = RealUnitedScalar(1.0, self.kg_unit)
        
        # Import LOG_LEVEL_DIMENSION for testing
        from united_system.dimension import LOG_LEVEL_DIMENSION
        self.log_level_dim = LOG_LEVEL_DIMENSION
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_default_base(self):
        """Test log() with default base (natural logarithm)."""
        result = self.positive_scalar.log()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(math.log(10.0))  # type: ignore
        assert result.dimension == self.log_level_dim
        assert result.display_unit is not None
        assert "dec" in str(result.display_unit)  # Should have log level unit
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_custom_base(self):
        """Test log() with custom base."""
        result = self.positive_scalar.log(base=2.0)
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(math.log(10.0, 2.0))  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log10(self):
        """Test log10() method."""
        result = self.positive_scalar.log10()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(math.log10(10.0))  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log2(self):
        """Test log2() method."""
        result = self.positive_scalar.log2()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(math.log2(10.0))  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_of_one(self):
        """Test log of 1 (should be 0)."""
        result = self.one_scalar.log()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(0.0)  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_of_zero(self):
        """Test log of 0 (should be -inf)."""
        result = self.zero_scalar.log()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == -math.inf
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_of_negative(self):
        """Test log of negative value (should be NaN)."""
        result = self.negative_scalar.log()
        
        assert isinstance(result, RealUnitedScalar)
        assert math.isnan(result.canonical_value)
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_of_log_level_scalar(self):
        """Test that log of a log level scalar raises error."""
        log_scalar = RealUnitedScalar.create_from_canonical_value(2.0, self.log_level_dim)
        
        with pytest.raises(ValueError, match="Cannot take the logarithm of a scalar with dimension LOG_LEVEL_DIMENSION"):
            log_scalar.log()
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp(self):
        """Test exp() method."""
        # Create a scalar with value 2.0
        scalar = RealUnitedScalar(2.0, self.kg_unit)
        result = scalar.exp()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(math.exp(2.0))  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp10(self):
        """Test exp10() method."""
        # Create a scalar with value 2.0
        scalar = RealUnitedScalar(2.0, self.kg_unit)
        result = scalar.exp10()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(10.0 ** 2.0)  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp2(self):
        """Test exp2() method."""
        # Create a scalar with value 3.0
        scalar = RealUnitedScalar(3.0, self.kg_unit)
        result = scalar.exp2()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(2.0 ** 3.0)  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_of_zero(self):
        """Test exp of 0 (should be 1)."""
        result = self.zero_scalar.exp()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(1.0)  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_of_negative(self):
        """Test exp of negative value."""
        result = self.negative_scalar.exp()
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == pytest.approx(math.exp(-5.0))  # type: ignore
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_exp_roundtrip(self):
        """Test that log and exp are inverse operations."""
        original_value = 7.5
        scalar = RealUnitedScalar(original_value, self.kg_unit)
        
        # log then exp should give us back the original value
        log_result = scalar.log()
        exp_result = log_result.exp()
        
        assert exp_result.canonical_value == pytest.approx(original_value)  # type: ignore
        assert exp_result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_log_roundtrip(self):
        """Test that exp and log are inverse operations."""
        original_value = 2.0
        scalar = RealUnitedScalar(original_value, self.kg_unit)
        
        # exp then log should give us back the original value
        exp_result = scalar.exp()
        log_result = exp_result.log()
        
        assert log_result.canonical_value == pytest.approx(original_value)  # type: ignore
        assert log_result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_with_display_unit(self):
        """Test log with display unit preserved."""
        result = self.positive_scalar.log()
        
        # Should have a display unit with log level
        assert result.display_unit is not None
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_without_display_unit(self):
        """Test log without display unit."""
        # Create scalar without display unit
        scalar_no_display = RealUnitedScalar.create_from_canonical_value(10.0, self.kg_unit.dimension)
        result = scalar_no_display.log()
        
        assert result.display_unit is None
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_with_display_unit(self):
        """Test exp with display unit preserved."""
        result = self.positive_scalar.exp()
        
        # Should have a display unit with log level
        assert result.display_unit is not None
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_without_display_unit(self):
        """Test exp without display unit."""
        # Create scalar without display unit
        scalar_no_display = RealUnitedScalar.create_from_canonical_value(2.0, self.kg_unit.dimension)
        result = scalar_no_display.exp()
        
        assert result.display_unit is None
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_edge_cases(self):
        """Test log with edge cases."""
        # Test with very small positive value
        small_scalar = RealUnitedScalar(1e-10, self.kg_unit)
        result = small_scalar.log()
        assert result.canonical_value == pytest.approx(math.log(1e-10))  # type: ignore
        
        # Test with very large value
        large_scalar = RealUnitedScalar(1e10, self.kg_unit)
        result = large_scalar.log()
        assert result.canonical_value == pytest.approx(math.log(1e10))  # type: ignore
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_edge_cases(self):
        """Test exp with edge cases."""
        # Test with very small value
        small_scalar = RealUnitedScalar(1e-10, self.kg_unit)
        result = small_scalar.exp()
        assert result.canonical_value == pytest.approx(math.exp(1e-10))  # type: ignore
        
        # Test with very large value
        large_scalar = RealUnitedScalar(100.0, self.kg_unit)
        result = large_scalar.exp()
        assert result.canonical_value == pytest.approx(math.exp(100.0))  # type: ignore
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_nan_input(self):
        """Test log with NaN input."""
        nan_scalar = RealUnitedScalar.create_from_canonical_value(float('nan'), self.kg_unit.dimension)
        result = nan_scalar.log()
        
        assert math.isnan(result.canonical_value)
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_nan_input(self):
        """Test exp with NaN input."""
        nan_scalar = RealUnitedScalar.create_from_canonical_value(float('nan'), self.kg_unit.dimension)
        result = nan_scalar.exp()
        
        assert math.isnan(result.canonical_value)
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_log_inf_input(self):
        """Test log with infinity input."""
        inf_scalar = RealUnitedScalar.create_from_canonical_value(float('inf'), self.kg_unit.dimension)
        result = inf_scalar.log()
        
        assert result.canonical_value == float('inf')
        assert result.dimension == self.log_level_dim
    
    @pytest.mark.skip(reason="Log/exp operations not yet implemented")
    def test_exp_inf_input(self):
        """Test exp with infinity input."""
        inf_scalar = RealUnitedScalar.create_from_canonical_value(float('inf'), self.kg_unit.dimension)
        result = inf_scalar.exp()
        
        assert result.canonical_value == float('inf')
        assert result.dimension == self.log_level_dim


class TestRealUnitedScalarUnitOperations:
    """Test unit operations that create scalars from numbers and units."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.kg_unit = Unit("kg")
        self.m_unit = Unit("m")
        self.s_unit = Unit("s")
    
    def test_unit_rmul_with_float(self):
        """Test creating scalar by multiplying float with unit."""
        scalar = 5.0 * self.kg_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 5.0
        assert scalar.dimension == self.kg_unit.dimension
        assert scalar.display_unit == self.kg_unit
    
    def test_unit_rmul_with_int(self):
        """Test creating scalar by multiplying int with unit."""
        scalar = 3 * self.m_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 3.0
        assert scalar.dimension == self.m_unit.dimension
        assert scalar.display_unit == self.m_unit
    
    def test_unit_rmul_with_zero(self):
        """Test creating scalar by multiplying zero with unit."""
        scalar = 0 * self.kg_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 0.0
        assert scalar.dimension == self.kg_unit.dimension
        assert scalar.display_unit == self.kg_unit
    
    def test_unit_rmul_with_negative(self):
        """Test creating scalar by multiplying negative number with unit."""
        scalar = -2.5 * self.s_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == -2.5
        assert scalar.dimension == self.s_unit.dimension
        assert scalar.display_unit == self.s_unit
    
    def test_unit_rtruediv_with_float(self):
        """Test creating scalar by dividing float by unit."""
        scalar = 10.0 / self.kg_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 10.0
        assert scalar.dimension == self.kg_unit.dimension.invert()
        assert scalar.display_unit == ~self.kg_unit
    
    def test_unit_rtruediv_with_int(self):
        """Test creating scalar by dividing int by unit."""
        scalar = 6 / self.m_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 6.0
        assert scalar.dimension == self.m_unit.dimension.invert()
        assert scalar.display_unit == ~self.m_unit
    
    def test_unit_rtruediv_with_zero(self):
        """Test creating scalar by dividing zero by unit."""
        scalar = 0 / self.s_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 0.0
        assert scalar.dimension == self.s_unit.dimension.invert()
        assert scalar.display_unit == ~self.s_unit
    
    def test_unit_rtruediv_with_negative(self):
        """Test creating scalar by dividing negative number by unit."""
        scalar = -4.0 / self.kg_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == -4.0
        assert scalar.dimension == self.kg_unit.dimension.invert()
        assert scalar.display_unit == ~self.kg_unit
    
    def test_unit_operations_with_complex_units(self):
        """Test unit operations with complex units."""
        # Test with compound unit
        velocity_unit = Unit("m/s")
        scalar = 15.0 * velocity_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 15.0
        assert scalar.dimension == velocity_unit.dimension
        assert scalar.display_unit == velocity_unit
    
    def test_unit_operations_with_prefixed_units(self):
        """Test unit operations with prefixed units."""
        # Test with prefixed unit
        km_unit = Unit("km")
        scalar = 2.5 * km_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 2500.0  # 2.5 km = 2500 m
        assert scalar.dimension == km_unit.dimension
        assert scalar.display_unit == km_unit
    
    def test_unit_operations_chain(self):
        """Test chaining unit operations."""
        # Test: (5 * kg) / (2 * m)
        mass_scalar: RealUnitedScalar = 5 * self.kg_unit
        length_scalar: RealUnitedScalar = 2 * self.m_unit
        result: RealUnitedScalar = mass_scalar / length_scalar
        
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 2.5
        assert result.dimension == self.kg_unit.dimension / self.m_unit.dimension
    
    def test_unit_operations_with_dimensionless(self):
        """Test unit operations with dimensionless units."""
        # Test with dimensionless unit (like rad)
        rad_unit = Unit("rad")
        scalar = 3.14 * rad_unit
        
        assert isinstance(scalar, RealUnitedScalar)
        assert scalar.canonical_value == 3.14
        assert scalar.dimension == rad_unit.dimension
        assert scalar.display_unit == rad_unit
    
    def test_chained_scalar_unit_multiplication(self):
        """Test chained multiplication: 3 * Unit('m/s') * 4 * Unit('h')"""
        velocity = 3 * Unit('m/s')
        time = 4 * Unit('h')
        result = velocity * time
        # 3 m/s * 4 h = 3 * 4 * m * h / s
        # 4 h = 14400 s, so canonical value: 3 * 14400 = 43200 m
        assert isinstance(result, RealUnitedScalar)
        assert result.dimension == Unit('m').dimension
        assert result.display_unit.dimension == Unit('m').dimension
        assert result.canonical_value == 43200.0
        # Optionally, print for debug
        print(f"Result: {result}")

    def test_chained_addition_with_units(self):
        """Test addition: (2 * Unit('m')) + (3 * Unit('m'))"""
        a = 2 * Unit('m')
        b = 3 * Unit('m')
        result = a + b
        assert isinstance(result, RealUnitedScalar)
        assert result.dimension == Unit('m').dimension
        assert result.canonical_value == 5.0
        assert result.display_unit == Unit('m')

    def test_chained_subtraction_with_units(self):
        """Test subtraction: (5 * Unit('s')) - (2 * Unit('s'))"""
        a = 5 * Unit('s')
        b = 2 * Unit('s')
        result = a - b
        assert isinstance(result, RealUnitedScalar)
        assert result.dimension == Unit('s').dimension
        assert result.canonical_value == 3.0
        assert result.display_unit == Unit('s')

    def test_chained_multiplication_with_units(self):
        """Test multiplication: (2 * Unit('m')) * (3 * Unit('s'))"""
        a = 2 * Unit('m')
        b = 3 * Unit('s')
        result = a * b
        assert isinstance(result, RealUnitedScalar)
        assert result.dimension == (Unit('m').dimension * Unit('s').dimension)
        assert result.canonical_value == 6.0
        # Display unit is a compound unit
        assert result.display_unit.dimension == (Unit('m').dimension * Unit('s').dimension)

    def test_chained_division_with_units(self):
        """Test division: (10 * Unit('m')) / (2 * Unit('s'))"""
        a = 10 * Unit('m')
        b = 2 * Unit('s')
        result = a / b
        assert isinstance(result, RealUnitedScalar)
        assert result.dimension == (Unit('m').dimension / Unit('s').dimension)
        assert result.canonical_value == 5.0
        # Display unit is a compound unit
        assert result.display_unit.dimension == (Unit('m').dimension / Unit('s').dimension)

    def test_mixed_operations_with_scalars_and_units(self):
        """Test mixed operations: (2 * Unit('m')) * 3 + (4 * Unit('m')) / 2"""
        a = 2 * Unit('m')
        b = 4 * Unit('m')
        result = a * 3 + b / 2
        # (2 * 3) + (4 / 2) = 6 + 2 = 8
        assert isinstance(result, RealUnitedScalar)
        assert result.dimension == Unit('m').dimension
        assert result.canonical_value == 8.0
        assert result.display_unit == Unit('m')


if __name__ == "__main__":
    pytest.main([__file__])
