"""
Comprehensive tests for Dimension class.

Tests all functionality including:
- Creation and initialization
- Arithmetic operations
- Comparison operations
- String representation
- Canonical unit generation
- Serialization
- Edge cases and error conditions
"""

import pytest
import numpy as np
import h5py
import json

# Import the modules to test
from united_system.dimension import Dimension
from united_system.unit import Unit


class TestDimensionCreation:
    """Test dimension creation and initialization."""
    
    def test_create_with_tuples(self):
        """Test creation with tuple inputs."""
        dim_exponents = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        pseudo_exponents = (0, 0)
        dimension = Dimension.create(dim_exponents, pseudo_exponents)
        
        assert dimension.dimension_exponents == dim_exponents
        assert dimension.pseudo_dimension_exponents == pseudo_exponents
    
    def test_create_with_lists(self):
        """Test creation with list inputs."""
        dim_exponents = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        pseudo_exponents = [0, 0]
        dimension = Dimension.create(dim_exponents, pseudo_exponents)
        
        assert dimension.dimension_exponents == tuple(dim_exponents)
        assert dimension.pseudo_dimension_exponents == tuple(pseudo_exponents)
    
    def test_create_mass_dimension(self):
        """Test creation of mass dimension."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        assert dimension.dimension_exponents[0] == 1.0  # Mass exponent
        assert all(exp == 0.0 for exp in dimension.dimension_exponents[1:])
    
    def test_create_length_dimension(self):
        """Test creation of length dimension."""
        dimension = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        assert dimension.dimension_exponents[2] == 1.0  # Length exponent
        assert all(exp == 0.0 for exp in dimension.dimension_exponents[:2])
        assert all(exp == 0.0 for exp in dimension.dimension_exponents[3:])
    
    def test_create_time_dimension(self):
        """Test creation of time dimension."""
        dimension = Dimension.create([0, 1, 0, 0, 0, 0, 0], [0, 0])
        assert dimension.dimension_exponents[1] == 1.0  # Time exponent
        assert all(exp == 0.0 for exp in dimension.dimension_exponents[:1])
        assert all(exp == 0.0 for exp in dimension.dimension_exponents[2:])
    
    def test_create_complex_dimension(self):
        """Test creation of complex dimension (e.g., force)."""
        # Force = Mass * Length / Time^2
        dimension = Dimension.create([1, -2, 1, 0, 0, 0, 0], [0, 0])
        assert dimension.dimension_exponents[0] == 1.0   # Mass
        assert dimension.dimension_exponents[1] == -2.0  # Time^-2
        assert dimension.dimension_exponents[2] == 1.0   # Length
        assert all(exp == 0.0 for exp in dimension.dimension_exponents[3:])
    
    def test_create_with_pseudo_dimensions(self):
        """Test creation with pseudo dimensions (angle, log level)."""
        dimension = Dimension.create([0, 0, 0, 0, 0, 0, 0], [1, 0])  # Angle
        assert dimension.pseudo_dimension_exponents[0] == 1  # Angle
        assert dimension.pseudo_dimension_exponents[1] == 0  # Log level
    
    def test_zero_dimension(self):
        """Test zero dimension creation."""
        dimension = Dimension.zero_dimension()
        assert dimension.is_zero()
        assert all(exp == 0.0 for exp in dimension.dimension_exponents)
        assert all(exp == 0 for exp in dimension.pseudo_dimension_exponents)


class TestDimensionArithmetic:
    """Test arithmetic operations on dimensions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        self.time_dim = Dimension.create([0, 1, 0, 0, 0, 0, 0], [0, 0])
        self.zero_dim = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
    
    def test_addition_same_dimension(self):
        """Test addition of same dimensions."""
        result = self.mass_dim + self.mass_dim
        expected = Dimension.create([2, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_addition_different_dimensions(self):
        """Test addition of different dimensions."""
        result = self.mass_dim + self.length_dim
        expected = Dimension.create([1, 0, 1, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_addition_with_pseudo_dimensions(self):
        """Test addition with pseudo dimensions."""
        angle_dim = Dimension.create([0, 0, 0, 0, 0, 0, 0], [1, 0])
        result = self.mass_dim + angle_dim
        expected = Dimension.create([1, 0, 0, 0, 0, 0, 0], [1, 0])
        assert result == expected
    
    def test_subtraction_same_dimension(self):
        """Test subtraction of same dimensions."""
        result = self.mass_dim - self.mass_dim
        expected = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_subtraction_different_dimensions(self):
        """Test subtraction of different dimensions."""
        result = self.mass_dim - self.length_dim
        expected = Dimension.create([1, 0, -1, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_multiplication_by_scalar(self):
        """Test multiplication by scalar."""
        result = self.mass_dim * 2.0
        expected = Dimension.create([2, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_division_by_scalar(self):
        """Test division by scalar."""
        result = self.mass_dim / 2.0
        expected = Dimension.create([0.5, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_invert(self):
        """Test dimension inversion."""
        result = self.mass_dim.invert()
        expected = Dimension.create([-1, 0, 0, 0, 0, 0, 0], [0, 0])
        assert result == expected
    
    def test_invert_complex_dimension(self):
        """Test inversion of complex dimension."""
        force_dim = Dimension.create([1, -2, 1, 0, 0, 0, 0], [0, 0])
        result = force_dim.invert()
        expected = Dimension.create([-1, 2, -1, 0, 0, 0, 0], [0, 0])
        assert result == expected


class TestDimensionComparison:
    """Test comparison operations on dimensions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        self.zero_dim = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
    
    def test_equality_same_dimension(self):
        """Test equality of same dimensions."""
        mass_dim2 = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        assert self.mass_dim == mass_dim2
    
    def test_equality_different_dimensions(self):
        """Test equality of different dimensions."""
        assert self.mass_dim != self.length_dim
    
    def test_equality_with_non_dimension(self):
        """Test equality with non-dimension object."""
        assert self.mass_dim != "not a dimension"
        assert self.mass_dim != 42
    
    def test_inequality(self):
        """Test inequality."""
        assert self.mass_dim != self.length_dim
        assert not (self.mass_dim != self.mass_dim)
    
    def test_hash_consistency(self):
        """Test that equal dimensions have same hash."""
        mass_dim2 = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        assert hash(self.mass_dim) == hash(mass_dim2)
    
    def test_is_zero(self):
        """Test is_zero method."""
        assert self.zero_dim.is_zero()
        assert not self.mass_dim.is_zero()
        assert not self.length_dim.is_zero()


class TestDimensionStringRepresentation:
    """Test string representation of dimensions."""
    
    def test_str_mass_dimension(self):
        """Test string representation of mass dimension."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        result = str(dimension)
        assert "M" in result or "mass" in result.lower()
    
    def test_str_length_dimension(self):
        """Test string representation of length dimension."""
        dimension = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        result = str(dimension)
        assert "L" in result or "length" in result.lower()
    
    def test_str_time_dimension(self):
        """Test string representation of time dimension."""
        dimension = Dimension.create([0, 1, 0, 0, 0, 0, 0], [0, 0])
        result = str(dimension)
        assert "T" in result or "time" in result.lower()
    
    def test_str_complex_dimension(self):
        """Test string representation of complex dimension."""
        # Force = Mass * Length / Time^2
        dimension = Dimension.create([1, -2, 1, 0, 0, 0, 0], [0, 0])
        result = str(dimension)
        # Should show something like "M T^-2 L" or "force"
        assert len(result) > 0
    
    def test_str_dimensionless(self):
        """Test string representation of dimensionless quantity."""
        dimension = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        result = str(dimension)
        assert "dimensionless" in result.lower()
    
    def test_str_with_pseudo_dimensions(self):
        """Test string representation with pseudo dimensions."""
        dimension = Dimension.create([0, 0, 0, 0, 0, 0, 0], [1, 0])  # Angle
        result = str(dimension)
        assert "A" in result or "angle" in result.lower()
    
    def test_str_with_log_level(self):
        """Test string representation with log level."""
        dimension = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 1])  # Log level
        result = str(dimension)
        assert "Log" in result


class TestDimensionCanonicalUnit:
    """Test canonical unit generation."""
    
    def test_canonical_unit_mass(self):
        """Test canonical unit for mass dimension."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        canonical_unit = dimension.canonical_unit
        assert canonical_unit is not None
        assert canonical_unit.dimension == dimension
    
    def test_canonical_unit_length(self):
        """Test canonical unit for length dimension."""
        dimension = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        canonical_unit = dimension.canonical_unit
        assert canonical_unit is not None
        assert canonical_unit.dimension == dimension
    
    def test_canonical_unit_time(self):
        """Test canonical unit for time dimension."""
        dimension = Dimension.create([0, 1, 0, 0, 0, 0, 0], [0, 0])
        canonical_unit = dimension.canonical_unit
        assert canonical_unit is not None
        assert canonical_unit.dimension == dimension
    
    def test_canonical_unit_complex(self):
        """Test canonical unit for complex dimension."""
        # Force = Mass * Length / Time^2
        dimension = Dimension.create([1, -2, 1, 0, 0, 0, 0], [0, 0])
        canonical_unit = dimension.canonical_unit
        assert canonical_unit is not None
        assert canonical_unit.dimension == dimension
    
    def test_canonical_unit_dimensionless(self):
        """Test canonical unit for dimensionless quantity."""
        dimension = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        canonical_unit = dimension.canonical_unit
        assert canonical_unit is not None
        # Should be empty unit for dimensionless
    
    def test_canonical_unit_caching(self):
        """Test that canonical units are cached."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        unit1 = dimension.canonical_unit
        unit2 = dimension.canonical_unit
        assert unit1 is unit2  # Same object due to caching


class TestDimensionCompatibility:
    """Test dimension compatibility checking."""
    
    def test_compatible_to_same_dimension(self):
        """Test compatibility with same dimension."""
        dim1 = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        dim2 = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        assert dim1.compatible_to(dim2)
    
    def test_compatible_to_different_dimension(self):
        """Test compatibility with different dimension."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        assert not mass_dim.compatible_to(length_dim)
    
    def test_compatible_to_with_unit(self):
        """Test compatibility with unit."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        kg_unit = Unit.parse_string("kg")
        assert mass_dim.compatible_to(kg_unit)
    
    def test_compatible_to_with_incompatible_unit(self):
        """Test compatibility with incompatible unit."""
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        m_unit = Unit.parse_string("m")
        assert not mass_dim.compatible_to(m_unit)


class TestDimensionSerialization:
    """Test serialization methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
    
    def test_to_json(self):
        """Test to_json method."""
        json_data = self.mass_dim.to_json()
        
        assert "dimension_exponents" in json_data
        assert "pseudo_dimension_exponents" in json_data
        assert json_data["dimension_exponents"] == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        assert json_data["pseudo_dimension_exponents"] == [0, 0]
    
    def test_from_json(self):
        """Test from_json method."""
        json_data = {
            "dimension_exponents": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "pseudo_dimension_exponents": [0, 0]
        }
        
        dimension = Dimension.from_json(json_data)
        assert dimension == self.mass_dim
    
    def test_to_hdf5(self):
        """Test to_hdf5 method."""
        with h5py.File("test_dimension.h5", "w") as f:
            group = f.create_group("dimension")
            self.mass_dim.to_hdf5(group)
            
            # Verify data was written
            assert "dimension_exponents" in group
            assert "pseudo_dimension_exponents" in group
    
    def test_from_hdf5(self):
        """Test from_hdf5 method."""
        # First write to file
        with h5py.File("test_dimension.h5", "w") as f:
            group = f.create_group("dimension")
            self.mass_dim.to_hdf5(group)
        
        # Then read from file
        with h5py.File("test_dimension.h5", "r") as f:
            group = f["dimension"]
            dimension = Dimension.from_hdf5(group)
            
            assert dimension == self.mass_dim


class TestDimensionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_exponents(self):
        """Test with very large exponents."""
        large_exponents = [1e10, 0, 0, 0, 0, 0, 0]
        dimension = Dimension.create(large_exponents, [0, 0])
        assert dimension.dimension_exponents[0] == 1e10
    
    def test_very_small_exponents(self):
        """Test with very small exponents."""
        small_exponents = [1e-10, 0, 0, 0, 0, 0, 0]
        dimension = Dimension.create(small_exponents, [0, 0])
        assert dimension.dimension_exponents[0] == 1e-10
    
    def test_negative_exponents(self):
        """Test with negative exponents."""
        negative_exponents = [-1, 0, 0, 0, 0, 0, 0]
        dimension = Dimension.create(negative_exponents, [0, 0])
        assert dimension.dimension_exponents[0] == -1.0
    
    def test_fractional_exponents(self):
        """Test with fractional exponents."""
        fractional_exponents = [0.5, 0, 0, 0, 0, 0, 0]
        dimension = Dimension.create(fractional_exponents, [0, 0])
        assert dimension.dimension_exponents[0] == 0.5
    
    def test_zero_exponents(self):
        """Test with all zero exponents."""
        zero_exponents = [0, 0, 0, 0, 0, 0, 0]
        dimension = Dimension.create(zero_exponents, [0, 0])
        assert dimension.is_zero()
    
    def test_immutability(self):
        """Test that dimensions are immutable."""
        dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        
        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            dimension.dimension_exponents = (2, 0, 0, 0, 0, 0, 0)


class TestDimensionPerformance:
    """Test performance characteristics."""
    
    def test_creation_performance(self):
        """Test that dimension creation is fast."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
    
    def test_canonical_unit_caching_performance(self):
        """Test that canonical unit caching improves performance."""
        dimension = Dimension.create([1, -2, 1, 0, 0, 0, 0], [0, 0])  # Force
        
        import time
        
        # First call (no cache)
        start_time = time.time()
        unit1 = dimension.canonical_unit
        first_call_time = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        unit2 = dimension.canonical_unit
        second_call_time = time.time() - start_time
        
        # Second call should be faster
        assert second_call_time < first_call_time
        assert unit1 is unit2


if __name__ == "__main__":
    pytest.main([__file__])
