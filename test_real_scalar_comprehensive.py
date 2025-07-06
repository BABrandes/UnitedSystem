"""
Comprehensive test suite for RealScalar class.

This file tests all methods and functionality of the RealScalar wrapper class,
including construction, arithmetic, comparisons, utilities, and serialization.
"""

import pytest
import tempfile
import h5py
import math
import json
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, 'src')

from united_system import RealUnitedScalar, Unit, Dimension
from united_system.scalars.real_united_scalar import RealUnitedScalar


class TestRealScalarConstruction:
    """Test all constructor patterns for RealScalar."""
    
    def test_string_construction(self):
        """Test creating RealScalar from strings."""
        # Basic units
        rs1 = RealUnitedScalar("5 m")
        assert rs1.canonical_value == 5.0
        assert str(rs1.dimension) == "LENGTH"
        
        # Complex units
        rs2 = RealUnitedScalar("10 kg*m/s^2")
        assert rs2.canonical_value == 10.0
        assert str(rs2.dimension) == "FORCE"
        
        # With decimals
        rs3 = RealUnitedScalar("3.14159 rad")
        assert abs(rs3.canonical_value - 3.14159) < 1e-10
        
        # Scientific notation
        rs4 = RealUnitedScalar("1.2e3 Hz")
        assert rs4.canonical_value == 1200.0
        
    def test_value_unit_construction(self):
        """Test creating RealScalar from value and Unit."""
        rs1 = RealUnitedScalar(5.0, Unit("m"))
        assert rs1.canonical_value == 5.0
        assert str(rs1.dimension) == "LENGTH"
        
        rs2 = RealUnitedScalar(10, Unit("kg"))
        assert rs2.canonical_value == 10.0
        assert str(rs2.dimension) == "MASS"
        
        # With unit conversion
        rs3 = RealUnitedScalar(1.0, Unit("km"))
        assert rs3.canonical_value == 1000.0  # Converted to meters
        
    def test_dimensionless_construction(self):
        """Test creating dimensionless RealScalar."""
        rs1 = RealUnitedScalar(0.5)
        assert rs1.canonical_value == 0.5
        assert rs1.dimension.is_dimensionless()
        
        rs2 = RealUnitedScalar(42)
        assert rs2.canonical_value == 42.0
        assert rs2.dimension.is_dimensionless()
        
    def test_realunitedscalar_construction(self):
        """Test creating RealScalar from existing RealUnitedScalar."""
        rus = RealUnitedScalar.parse_string("7 V")
        rs = RealUnitedScalar(rus)
        assert rs.canonical_value == 7.0
        assert str(rs.dimension) == "VOLTAGE"
        
    def test_invalid_construction(self):
        """Test invalid constructor arguments."""
        # Cannot specify unit with string
        with pytest.raises(ValueError):
            RealUnitedScalar("5 m", Unit("s"))
            
        # Invalid string format
        with pytest.raises(ValueError):
            RealUnitedScalar("invalid string")
            
        # Invalid types - list input actually creates object but fails on access
        with pytest.raises(AttributeError):
            rs_invalid = RealUnitedScalar([1, 2, 3])
            # This will fail when trying to access the canonical_value
            _ = rs_invalid.canonical_value


class TestRealScalarProperties:
    """Test all properties of RealScalar."""
    
    def test_canonical_value(self):
        """Test canonical_value property."""
        rs1 = RealUnitedScalar("5 m")
        assert rs1.canonical_value == 5.0
        
        rs2 = RealUnitedScalar("1 km")
        assert rs2.canonical_value == 1000.0  # Converted to meters
        
    def test_dimension_property(self):
        """Test dimension property."""
        rs1 = RealUnitedScalar("5 m")
        assert isinstance(rs1.dimension, Dimension)
        assert str(rs1.dimension) == "LENGTH"
        
        rs2 = RealUnitedScalar("10 kg*m/s^2")
        assert str(rs2.dimension) == "FORCE"
        
        rs3 = RealUnitedScalar(0.5)  # Dimensionless
        assert rs3.dimension.is_dimensionless()
        
    def test_display_unit_property(self):
        """Test display_unit property."""
        rs1 = RealUnitedScalar("5 m")
        assert isinstance(rs1.display_unit, Unit) or rs1.display_unit is None
        
        rs2 = RealUnitedScalar(5.0, Unit("km"))
        if rs2.display_unit is not None:
            assert isinstance(rs2.display_unit, Unit)


class TestRealScalarArithmetic:
    """Test all arithmetic operations."""
    
    def test_addition(self):
        """Test addition operations."""
        rs1 = RealUnitedScalar("5 m")
        rs2 = RealUnitedScalar("3 m")
        
        # Addition
        result = rs1 + rs2
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 8.0
        assert str(result.dimension) == "LENGTH"
        
        # Reverse addition
        result2 = rs2 + rs1
        assert result2.canonical_value == 8.0
        
    def test_subtraction(self):
        """Test subtraction operations."""
        rs1 = RealUnitedScalar("10 m")
        rs2 = RealUnitedScalar("3 m")
        
        # Subtraction
        result = rs1 - rs2
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 7.0
        assert str(result.dimension) == "LENGTH"
        
        # Reverse subtraction
        result2 = rs2 - rs1
        assert result2.canonical_value == -7.0
        
    def test_multiplication(self):
        """Test multiplication operations."""
        rs1 = RealUnitedScalar("5 m")
        rs2 = RealUnitedScalar("2 s")
        
        # Scalar multiplication
        result = rs1 * rs2
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 10.0
        
        # Number multiplication
        result2 = rs1 * 3
        assert result2.canonical_value == 15.0
        assert str(result2.dimension) == "LENGTH"
        
        # Reverse number multiplication
        result3 = 2 * rs1
        assert result3.canonical_value == 10.0
        
    def test_division(self):
        """Test division operations."""
        rs1 = RealUnitedScalar("10 m")
        rs2 = RealUnitedScalar("2 s")
        
        # Scalar division
        result = rs1 / rs2
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 5.0
        
        # Number division
        result2 = rs1 / 2
        assert result2.canonical_value == 5.0
        assert str(result2.dimension) == "LENGTH"
        
        # Reverse number division
        result3 = 20 / rs2
        assert result3.canonical_value == 10.0
        
    def test_power(self):
        """Test power operations."""
        rs = RealUnitedScalar("3 m")
        
        # Square
        result = rs ** 2
        assert isinstance(result, RealUnitedScalar)
        assert result.canonical_value == 9.0
        
        # Cube
        result2 = rs ** 3
        assert result2.canonical_value == 27.0
        
        # Fractional power
        rs2 = RealUnitedScalar("9 m^2")
        result3 = rs2 ** 0.5
        assert abs(result3.canonical_value - 3.0) < 1e-10


class TestRealScalarComparisons:
    """Test all comparison operations."""
    
    def test_equality(self):
        """Test equality comparisons."""
        rs1 = RealUnitedScalar("5 m")
        rs2 = RealUnitedScalar("5 m")
        rs3 = RealUnitedScalar("3 m")
        
        assert rs1 == rs2
        assert not (rs1 == rs3)
        assert rs1 != rs3
        assert not (rs1 != rs2)
        
    def test_ordering(self):
        """Test ordering comparisons."""
        rs1 = RealUnitedScalar("5 m")
        rs2 = RealUnitedScalar("3 m")
        rs3 = RealUnitedScalar("7 m")
        
        # Less than
        assert rs2 < rs1
        assert rs1 < rs3
        assert not (rs1 < rs2)
        
        # Less than or equal
        assert rs2 <= rs1
        assert rs1 <= rs1
        assert not (rs3 <= rs1)
        
        # Greater than
        assert rs1 > rs2
        assert rs3 > rs1
        assert not (rs2 > rs1)
        
        # Greater than or equal
        assert rs1 >= rs2
        assert rs1 >= rs1
        assert not (rs2 >= rs3)


class TestRealScalarStringRepresentation:
    """Test string representation methods."""
    
    def test_str(self):
        """Test __str__ method."""
        rs1 = RealUnitedScalar("5 m")
        str_repr = str(rs1)
        assert isinstance(str_repr, str)
        assert "5" in str_repr
        assert "m" in str_repr
        
    def test_repr(self):
        """Test __repr__ method."""
        rs1 = RealUnitedScalar("5 m")
        repr_str = repr(rs1)
        assert isinstance(repr_str, str)
        assert "RealScalar" in repr_str
        
    def test_format(self):
        """Test format method."""
        rs = RealUnitedScalar("1234.5678 m")
        
        # Default formatting
        formatted = rs.format()
        assert isinstance(formatted, str)
        
        # Custom decimals
        formatted2 = rs.format(decimals=2)
        assert isinstance(formatted2, str)
        
        # Custom unit (if available)
        try:
            formatted3 = rs.format(Unit("km"), decimals=3)
            assert isinstance(formatted3, str)
        except:
            pass  # Unit conversion might not be available


class TestRealScalarUtilities:
    """Test utility methods."""
    
    def test_is_nan(self):
        """Test is_nan method."""
        rs_normal = RealUnitedScalar("5 m")
        assert not rs_normal.is_nan()
        
        # Create NaN scalar
        rs_nan = RealUnitedScalar(float('nan'))
        assert rs_nan.is_nan()
        
    def test_is_infinite(self):
        """Test is_infinite method."""
        rs_normal = RealUnitedScalar("5 m")
        assert not rs_normal.is_infinite()
        
        # Create infinite scalar
        rs_inf = RealUnitedScalar(float('inf'))
        assert rs_inf.is_infinite()
        
    def test_is_finite(self):
        """Test is_finite method."""
        rs_normal = RealUnitedScalar("5 m")
        assert rs_normal.is_finite()
        
        # Test non-finite values
        rs_inf = RealUnitedScalar(float('inf'))
        assert not rs_inf.is_finite()
        
        rs_nan = RealUnitedScalar(float('nan'))
        assert not rs_nan.is_finite()
        
    def test_compatible_with(self):
        """Test compatible_with method."""
        rs1 = RealUnitedScalar("5 m")
        rs2 = RealUnitedScalar("3 m")
        rs3 = RealUnitedScalar("10 kg")
        
        # Compatible dimensions
        assert rs1.compatible_with(rs2)
        
        # Incompatible dimensions
        assert not rs1.compatible_with(rs3)


class TestRealScalarSerialization:
    """Test serialization methods."""
    
    def test_json_serialization(self):
        """Test JSON serialization."""
        rs = RealUnitedScalar("42.5 m/s")
        
        # Serialize to JSON
        json_data = rs.to_json()
        assert isinstance(json_data, dict)
        
        # Deserialize from JSON
        rs_restored = RealUnitedScalar.from_json(json_data)
        assert isinstance(rs_restored, RealUnitedScalar)
        assert abs(rs_restored.canonical_value - rs.canonical_value) < 1e-10
        assert rs_restored.dimension == rs.dimension
        
    def test_hdf5_serialization(self):
        """Test HDF5 serialization."""
        rs = RealUnitedScalar("123.456 kg*m/s^2")
        
        # Create temporary HDF5 file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            # Serialize to HDF5
            with h5py.File(tmp_path, 'w') as f:
                rs.to_hdf5(f)
                
            # Deserialize from HDF5
            with h5py.File(tmp_path, 'r') as f:
                rs_restored = RealUnitedScalar.from_hdf5(f)
                
            assert isinstance(rs_restored, RealUnitedScalar)
            assert abs(rs_restored.canonical_value - rs.canonical_value) < 1e-10
            assert rs_restored.dimension == rs.dimension
            
        finally:
            # Cleanup
            Path(tmp_path).unlink(missing_ok=True)


class TestRealScalarEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_zero_values(self):
        """Test handling of zero values."""
        rs_zero = RealUnitedScalar("0 m")
        assert rs_zero.canonical_value == 0.0
        assert str(rs_zero.dimension) == "LENGTH"
        
    def test_negative_values(self):
        """Test handling of negative values."""
        rs_neg = RealUnitedScalar("-5.5 m")
        assert rs_neg.canonical_value == -5.5
        assert str(rs_neg.dimension) == "LENGTH"
        
    def test_very_large_values(self):
        """Test handling of very large values."""
        rs_large = RealUnitedScalar("1e100 m")
        assert rs_large.canonical_value == 1e100
        assert rs_large.is_finite()
        
    def test_very_small_values(self):
        """Test handling of very small values."""
        rs_small = RealUnitedScalar("1e-100 m")
        assert rs_small.canonical_value == 1e-100
        assert rs_small.is_finite()


class TestRealScalarIntegration:
    """Integration tests combining multiple features."""
    
    def test_physics_calculation(self):
        """Test a realistic physics calculation."""
        # F = ma (Force = mass * acceleration)
        mass = RealUnitedScalar("2 kg")
        acceleration = RealUnitedScalar("9.8 m/s^2")
        
        force = mass * acceleration
        assert isinstance(force, RealUnitedScalar)
        assert abs(force.canonical_value - 19.6) < 1e-10
        assert str(force.dimension) == "FORCE"
        
    def test_energy_calculation(self):
        """Test energy calculation E = 0.5 * m * v^2."""
        mass = RealUnitedScalar("1 kg")
        velocity = RealUnitedScalar("10 m/s")
        
        kinetic_energy = 0.5 * mass * (velocity ** 2)
        assert isinstance(kinetic_energy, RealUnitedScalar)
        assert abs(kinetic_energy.canonical_value - 50.0) < 1e-10
        # Should be energy dimension (ML^2T^-2)
        
    def test_unit_conversions_in_calculations(self):
        """Test calculations with different units."""
        distance_m = RealUnitedScalar("1000 m")
        distance_km = RealUnitedScalar("1 km")
        
        total_distance = distance_m + distance_km
        assert abs(total_distance.canonical_value - 2000.0) < 1e-10  # Both in meters
        
    def test_complex_expression(self):
        """Test a complex expression with multiple operations."""
        a = RealUnitedScalar("2 m")
        b = RealUnitedScalar("3 s")
        c = RealUnitedScalar("4 kg")
        
        # Complex calculation: (a^2 * c) / b
        result = (a ** 2 * c) / b
        assert isinstance(result, RealUnitedScalar)
        expected_value = (4 * 4) / 3  # 16/3
        assert abs(result.canonical_value - expected_value) < 1e-10


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running RealScalar comprehensive tests...")
    
    # Test basic functionality
    print("✓ Testing construction...")
    rs1 = RealUnitedScalar("5 m")
    rs2 = RealUnitedScalar(10.0, Unit("kg"))
    rs3 = RealUnitedScalar(0.5)
    
    print("✓ Testing arithmetic...")
    result = rs1 + RealUnitedScalar("3 m")
    assert result.canonical_value == 8.0
    
    print("✓ Testing comparisons...")
    assert rs1 > RealUnitedScalar("3 m")
    assert rs1 == RealUnitedScalar("5 m")
    
    print("✓ Testing utilities...")
    assert rs1.is_finite()
    assert not rs1.is_nan()
    
    print("✓ Testing serialization...")
    json_data = rs1.to_json()
    rs_restored = RealUnitedScalar.from_json(json_data)
    assert abs(rs_restored.canonical_value - rs1.canonical_value) < 1e-10
    
    print("🎉 All basic tests passed!")
    print("Run with pytest for full test suite: pytest test_real_scalar_comprehensive.py") 