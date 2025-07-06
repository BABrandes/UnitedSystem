"""
Tests for RealUnitedScalar class.
"""
import pytest
import math
from src.united_system.scalars.real_united_scalar.real_united_scalar import RealUnitedScalar
from united_system.units.simple.simple_unit import SimpleUnit
from united_system.units.named_simple_dimensions import NamedSimpleDimension


class TestRealUnitedScalarBasics:
    """Test basic functionality of RealUnitedScalar."""
    
    def test_creation_from_value_and_unit(self):
        """Test creating RealUnitedScalar from value and unit."""
        meter = SimpleUnit.parse_string("m")
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        assert length.canonical_value == 5.0
        assert length.display_unit == meter
        assert length.unit_dimension == meter.dimension
    
    def test_parse_string(self):
        """Test parsing from string."""
        length = RealUnitedScalar.parse_string("5.0 m")
        assert length.canonical_value == 5.0
        
        voltage = RealUnitedScalar.parse_string("12 V")
        assert voltage.canonical_value == 12.0
    
    def test_string_representation(self):
        """Test string representation."""
        meter = SimpleUnit.parse_string("m")
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        # This tests the __str__ method
        str_repr = str(length)
        assert "5" in str_repr
        assert "m" in str_repr


class TestRealUnitedScalarArithmetic:
    """Test arithmetic operations with proper unit handling."""
    
    def test_addition_same_units(self):
        """Test addition with same units."""
        meter = SimpleUnit.parse_string("m")
        length1 = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        length2 = RealUnitedScalar.create_from_value_and_unit(3.0, meter)
        
        result = length1 + length2
        assert result.canonical_value == 8.0
        assert result.unit_dimension == meter.dimension
    
    def test_addition_different_units_same_dimension(self):
        """Test addition with different units but same dimension - THIS IS THE BUG!"""
        km = SimpleUnit.parse_string("km")
        m = SimpleUnit.parse_string("m")
        
        # 0.01 km = 10 m, so 0.01 km + 5000 m should be 5010 m canonical
        length1 = RealUnitedScalar.create_from_value_and_unit(0.01, km)  # 10 m canonical
        length2 = RealUnitedScalar.create_from_value_and_unit(5000, m)   # 5000 m canonical
        
        result = length1 + length2
        
        # Check canonical value is correct
        assert result.canonical_value == 5010.0, f"Expected 5010.0 but got {result.canonical_value}"
        
        # The display unit should be from the left operand or canonical
        # Let's test what we get
        print(f"Result display unit: {result.display_unit}")
        print(f"Result canonical value: {result.canonical_value}")
        print(f"Result string: {str(result)}")
        
        # If display unit is km, the displayed value should be 5.01 km
        # If display unit is m, the displayed value should be 5010 m
        if result.display_unit and result.display_unit.format_string() == "km":
            display_value = result.display_unit.from_canonical_value(result.canonical_value)
            expected_display = 5.01
            assert abs(display_value - expected_display) < 0.001, f"Expected ~{expected_display} km but got {display_value} km"
        
    def test_subtraction_different_units_same_dimension(self):
        """Test subtraction with different units but same dimension."""
        km = SimpleUnit.parse_string("km")
        m = SimpleUnit.parse_string("m")
        
        # 1 km = 1000 m, so 1 km - 500 m should be 500 m canonical
        length1 = RealUnitedScalar.create_from_value_and_unit(1.0, km)   # 1000 m canonical
        length2 = RealUnitedScalar.create_from_value_and_unit(500, m)    # 500 m canonical
        
        result = length1 - length2
        
        # Check canonical value is correct
        assert result.canonical_value == 500.0, f"Expected 500.0 but got {result.canonical_value}"
    
    def test_addition_incompatible_units(self):
        """Test addition with incompatible units should raise error."""
        meter = SimpleUnit.parse_string("m")
        volt = SimpleUnit.parse_string("V")
        
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        voltage = RealUnitedScalar.create_from_value_and_unit(12.0, volt)
        
        with pytest.raises(ValueError, match="incompatible dimensions"):
            length + voltage
    
    def test_multiplication_by_scalar(self):
        """Test multiplication by scalar."""
        meter = SimpleUnit.parse_string("m")
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        result = length * 2.0
        assert result.canonical_value == 10.0
        assert result.unit_dimension == meter.dimension
    
    def test_multiplication_by_united_scalar(self):
        """Test multiplication by another RealUnitedScalar."""
        meter = SimpleUnit.parse_string("m")
        second = SimpleUnit.parse_string("s")
        
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        time = RealUnitedScalar.create_from_value_and_unit(2.0, second)
        
        result = length * time
        assert result.canonical_value == 10.0
        # The unit dimension should be meter * second
        
    def test_division_by_scalar(self):
        """Test division by scalar."""
        meter = SimpleUnit.parse_string("m")
        length = RealUnitedScalar.create_from_value_and_unit(10.0, meter)
        
        result = length / 2.0
        assert result.canonical_value == 5.0
        assert result.unit_dimension == meter.dimension
    
    def test_division_by_united_scalar(self):
        """Test division by another RealUnitedScalar."""
        meter = SimpleUnit.parse_string("m")
        second = SimpleUnit.parse_string("s")
        
        length = RealUnitedScalar.create_from_value_and_unit(10.0, meter)
        time = RealUnitedScalar.create_from_value_and_unit(2.0, second)
        
        result = length / time
        assert result.canonical_value == 5.0
        # The unit dimension should be meter / second


class TestRealUnitedScalarComparisons:
    """Test comparison operations."""
    
    def test_equality_same_units(self):
        """Test equality with same units."""
        meter = SimpleUnit.parse_string("m")
        length1 = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        length2 = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        assert length1 == length2
    
    def test_equality_different_units_same_dimension(self):
        """Test equality with different units but same dimension."""
        km = SimpleUnit.parse_string("km")
        m = SimpleUnit.parse_string("m")
        
        # 1 km = 1000 m
        length1 = RealUnitedScalar.create_from_value_and_unit(1.0, km)
        length2 = RealUnitedScalar.create_from_value_and_unit(1000.0, m)
        
        assert length1 == length2
    
    def test_less_than_same_units(self):
        """Test less than with same units."""
        meter = SimpleUnit.parse_string("m")
        length1 = RealUnitedScalar.create_from_value_and_unit(3.0, meter)
        length2 = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        assert length1 < length2
        assert not length2 < length1
    
    def test_comparison_incompatible_units(self):
        """Test comparison with incompatible units should raise error."""
        meter = SimpleUnit.parse_string("m")
        volt = SimpleUnit.parse_string("V")
        
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        voltage = RealUnitedScalar.create_from_value_and_unit(12.0, volt)
        
        with pytest.raises(ValueError, match="incompatible dimensions"):
            length < voltage


class TestRealUnitedScalarConversions:
    """Test unit conversion operations."""
    
    def test_display_as_different_unit(self):
        """Test changing display unit."""
        meter = SimpleUnit.parse_string("m")
        km = SimpleUnit.parse_string("km")
        
        length = RealUnitedScalar.create_from_value_and_unit(1000.0, meter)
        length_km = length.display_as(km)
        
        assert length_km.canonical_value == 1000.0  # Same canonical value
        assert length_km.display_unit == km
        
        # Display value should be 1.0 km
        display_value = length_km.display_unit.from_canonical_value(length_km.canonical_value)
        assert abs(display_value - 1.0) < 0.001
    
    def test_in_unit(self):
        """Test getting value in specific unit."""
        meter = SimpleUnit.parse_string("m")
        km = SimpleUnit.parse_string("km")
        
        length = RealUnitedScalar.create_from_value_and_unit(1000.0, meter)
        value_in_km = length.in_unit(km)
        
        assert abs(value_in_km - 1.0) < 0.001


class TestRealUnitedScalarUtilities:
    """Test utility methods."""
    
    def test_is_positive(self):
        """Test is_positive method."""
        meter = SimpleUnit.parse_string("m")
        positive_length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        negative_length = RealUnitedScalar.create_from_value_and_unit(-3.0, meter)
        
        assert positive_length.is_positive()
        assert not negative_length.is_positive()
    
    def test_is_zero(self):
        """Test is_zero method."""
        meter = SimpleUnit.parse_string("m")
        zero_length = RealUnitedScalar.create_from_value_and_unit(0.0, meter)
        nonzero_length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        assert zero_length.is_zero()
        assert not nonzero_length.is_zero()
    
    def test_abs(self):
        """Test absolute value."""
        meter = SimpleUnit.parse_string("m")
        negative_length = RealUnitedScalar.create_from_value_and_unit(-5.0, meter)
        
        abs_length = negative_length.abs()
        assert abs_length.canonical_value == 5.0
        assert abs_length.unit_dimension == meter.dimension
    
    def test_compatible_to(self):
        """Test compatibility checking."""
        meter = SimpleUnit.parse_string("m")
        km = SimpleUnit.parse_string("km")
        volt = SimpleUnit.parse_string("V")
        
        length1 = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        length2 = RealUnitedScalar.create_from_value_and_unit(3.0, km)
        voltage = RealUnitedScalar.create_from_value_and_unit(12.0, volt)
        
        assert length1.compatible_to(length2)
        assert not length1.compatible_to(voltage)


class TestRealUnitedScalarEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_nan_handling(self):
        """Test NaN handling."""
        meter = SimpleUnit.parse_string("m")
        nan_length = RealUnitedScalar.create_from_canonical_value(float('nan'), meter.dimension, meter)
        
        assert nan_length.is_nan()
        assert not nan_length.is_finite()
    
    def test_infinity_handling(self):
        """Test infinity handling."""
        meter = SimpleUnit.parse_string("m")
        inf_length = RealUnitedScalar.create_from_canonical_value(float('inf'), meter.dimension, meter)
        
        assert not inf_length.is_finite()
        assert inf_length.is_positive()
    
    def test_zero_division(self):
        """Test division by zero."""
        meter = SimpleUnit.parse_string("m")
        length = RealUnitedScalar.create_from_value_and_unit(5.0, meter)
        
        result = length / 0.0
        assert math.isinf(result.canonical_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 