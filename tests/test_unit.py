"""
Comprehensive tests for Unit class.

Tests all functionality including:
- Creation and initialization
- String parsing
- Arithmetic operations
- Unit conversions
- Compatibility checking
- Serialization
- Edge cases and error conditions
"""

import pytest
import numpy as np
import h5py
import json

# Import the modules to test
from united_system.unit import Unit
from united_system.dimension import Dimension


class TestUnitCreation:
    """Test unit creation and initialization."""
    
    def test_create_empty(self):
        """Test creation of empty unit."""
        unit = Unit.create_empty()
        assert len(unit.unit_elements) == 0
        assert unit.factor == 1.0
        assert unit.offset == 0.0
    
    def test_create_from_simple_unit_element(self):
        """Test creation from SimpleUnitElement."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        # Create a meter unit element
        meter_element = SimpleUnitElement.parse_string("m", "nominator")
        unit = Unit.create(meter_element)
        
        assert len(unit.unit_elements) == 1
        assert unit.unit_elements[0] == meter_element
    
    def test_create_from_multiple_elements(self):
        """Test creation from multiple unit elements."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        meter_element = SimpleUnitElement.parse_string("m", "nominator")
        second_element = SimpleUnitElement.parse_string("s", "denominator")
        unit = Unit.create(meter_element, second_element)
        
        assert len(unit.unit_elements) == 2
        assert meter_element in unit.unit_elements
        assert second_element in unit.unit_elements
    
    def test_create_from_existing_unit(self):
        """Test creation from existing unit."""
        original_unit = Unit.parse_string("m/s")
        new_unit = Unit.create(original_unit)
        
        assert new_unit.unit_elements == original_unit.unit_elements
        assert new_unit.factor == original_unit.factor
        assert new_unit.offset == original_unit.offset
    
    def test_create_from_list_of_units(self):
        """Test creation from list of units."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("s")
        combined_unit = Unit.create([unit1, unit2])
        
        # Should combine elements from both units
        assert len(combined_unit.unit_elements) == 2
    
    def test_create_from_set_of_units(self):
        """Test creation from set of units."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("s")
        combined_unit = Unit.create({unit1, unit2})
        
        # Should combine elements from both units
        assert len(combined_unit.unit_elements) == 2
    
    def test_create_invalid_argument(self):
        """Test creation with invalid argument."""
        with pytest.raises(ValueError):
            Unit.create("invalid")
    
    def test_create_with_offset_conflict(self):
        """Test creation with conflicting offsets."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        # Create elements with different offsets
        celsius_element = SimpleUnitElement.parse_string("°C", "nominator")
        fahrenheit_element = SimpleUnitElement.parse_string("°F", "nominator")
        
        with pytest.raises(ValueError, match="Cannot have two non-zero offsets"):
            Unit.create(celsius_element, fahrenheit_element)


class TestUnitParsing:
    """Test unit string parsing."""
    
    def test_parse_simple_unit(self):
        """Test parsing simple unit."""
        unit = Unit.parse_string("m")
        assert len(unit.unit_elements) == 1
        assert str(unit) == "m"
    
    def test_parse_unit_with_prefix(self):
        """Test parsing unit with prefix."""
        unit = Unit.parse_string("km")
        assert len(unit.unit_elements) == 1
        assert unit.factor == 1000.0  # kilo prefix
    
    def test_parse_composite_unit(self):
        """Test parsing composite unit."""
        unit = Unit.parse_string("m/s")
        assert len(unit.unit_elements) == 2
        # One in numerator, one in denominator
    
    def test_parse_complex_unit(self):
        """Test parsing complex unit."""
        unit = Unit.parse_string("kg*m/s^2")
        assert len(unit.unit_elements) == 3
        # kg in numerator, m in numerator, s^2 in denominator
    
    def test_parse_unit_with_negative_exponent(self):
        """Test parsing unit with negative exponent."""
        unit = Unit.parse_string("m*s^-1")
        assert len(unit.unit_elements) == 2
    
    def test_parse_dimensionless(self):
        """Test parsing dimensionless unit."""
        unit = Unit.parse_string("")
        assert len(unit.unit_elements) == 0
        assert unit.factor == 1.0
    
    def test_parse_dimensionless_one(self):
        """Test parsing dimensionless unit with '1'."""
        unit = Unit.parse_string("1")
        assert len(unit.unit_elements) == 0
        assert unit.factor == 1.0
    
    def test_parse_invalid_string(self):
        """Test parsing invalid string."""
        with pytest.raises(ValueError):
            Unit.parse_string("invalid/unit/string")
    
    def test_parse_cached_result(self):
        """Test that parsing results are cached."""
        unit1 = Unit.parse_string("m/s")
        unit2 = Unit.parse_string("m/s")
        assert unit1 is unit2  # Same object due to caching
    
    def test_parse_temperature_unit(self):
        """Test parsing temperature unit with offset."""
        unit = Unit.parse_string("°C")
        assert unit.offset != 0.0  # Celsius has offset from Kelvin
    
    def test_parse_fractional_unit(self):
        """Test parsing unit with fractional notation."""
        unit = Unit.parse_string("1/s")
        assert len(unit.unit_elements) == 1
        # Should be equivalent to s^-1


class TestUnitArithmetic:
    """Test arithmetic operations on units."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.meter_unit = Unit.parse_string("m")
        self.second_unit = Unit.parse_string("s")
        self.kilogram_unit = Unit.parse_string("kg")
    
    def test_multiplication(self):
        """Test unit multiplication."""
        result = self.meter_unit * self.second_unit
        expected = Unit.parse_string("m*s")
        assert result.unit_elements == expected.unit_elements
    
    def test_division(self):
        """Test unit division."""
        result = self.meter_unit / self.second_unit
        expected = Unit.parse_string("m/s")
        assert result.unit_elements == expected.unit_elements
    
    def test_power(self):
        """Test unit power operation."""
        result = self.meter_unit ** 2
        expected = Unit.parse_string("m^2")
        assert result.unit_elements == expected.unit_elements
    
    def test_power_negative(self):
        """Test unit power with negative exponent."""
        result = self.meter_unit ** (-1)
        expected = Unit.parse_string("1/m")
        assert result.unit_elements == expected.unit_elements
    
    def test_power_zero(self):
        """Test unit power with zero exponent."""
        result = self.meter_unit ** 0
        expected = Unit.parse_string("")
        assert result.unit_elements == expected.unit_elements
    
    def test_power_fractional(self):
        """Test unit power with fractional exponent."""
        result = self.meter_unit ** 0.5
        # Should create unit with fractional exponent
        assert len(result.unit_elements) == 1
        assert result.unit_elements[0].exponent == 0.5


class TestUnitConversion:
    """Test unit conversion operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.meter_unit = Unit.parse_string("m")
        self.kilometer_unit = Unit.parse_string("km")
        self.centimeter_unit = Unit.parse_string("cm")
    
    def test_from_canonical_value_float(self):
        """Test converting canonical value to unit value."""
        # 1000 canonical meters = 1 km
        value = self.kilometer_unit.from_canonical_value(1000.0)
        assert value == 1.0
    
    def test_from_canonical_value_int(self):
        """Test converting canonical value (int) to unit value."""
        value = self.kilometer_unit.from_canonical_value(1000)
        assert value == 1.0
    
    def test_from_canonical_value_numpy_float(self):
        """Test converting canonical value (numpy float) to unit value."""
        value = self.kilometer_unit.from_canonical_value(np.float64(1000.0))
        assert value == 1.0
        assert isinstance(value, np.float64)
    
    def test_from_canonical_value_complex(self):
        """Test converting canonical value (complex) to unit value."""
        value = self.kilometer_unit.from_canonical_value(1000.0 + 0j)
        assert value == 1.0 + 0j
    
    def test_from_canonical_value_numpy_array(self):
        """Test converting canonical value (numpy array) to unit value."""
        canonical_values = np.array([1000.0, 2000.0, 3000.0])
        values = self.kilometer_unit.from_canonical_value(canonical_values)
        expected = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(values, expected)
    
    def test_to_canonical_value_float(self):
        """Test converting unit value to canonical value."""
        # 1 km = 1000 canonical meters
        canonical_value = self.kilometer_unit.to_canonical_value(1.0)
        assert canonical_value == 1000.0
    
    def test_to_canonical_value_int(self):
        """Test converting unit value (int) to canonical value."""
        canonical_value = self.kilometer_unit.to_canonical_value(1)
        assert canonical_value == 1000.0
    
    def test_to_canonical_value_numpy_float(self):
        """Test converting unit value (numpy float) to canonical value."""
        canonical_value = self.kilometer_unit.to_canonical_value(np.float64(1.0))
        assert canonical_value == 1000.0
        assert isinstance(canonical_value, np.float64)
    
    def test_to_canonical_value_complex(self):
        """Test converting unit value (complex) to canonical value."""
        canonical_value = self.kilometer_unit.to_canonical_value(1.0 + 0j)
        assert canonical_value == 1000.0 + 0j
    
    def test_to_canonical_value_numpy_array(self):
        """Test converting unit value (numpy array) to canonical value."""
        unit_values = np.array([1.0, 2.0, 3.0])
        canonical_values = self.kilometer_unit.to_canonical_value(unit_values)
        expected = np.array([1000.0, 2000.0, 3000.0])
        np.testing.assert_array_almost_equal(canonical_values, expected)
    
    def test_conversion_with_offset(self):
        """Test conversion with temperature units (offset)."""
        celsius_unit = Unit.parse_string("°C")
        kelvin_unit = Unit.parse_string("K")
        
        # 0°C = 273.15 K
        celsius_value = celsius_unit.from_canonical_value(273.15)
        assert celsius_value == 0.0
        
        # 0°C = 273.15 K (reverse)
        kelvin_value = celsius_unit.to_canonical_value(0.0)
        assert kelvin_value == 273.15
    
    def test_invalid_canonical_value_type(self):
        """Test conversion with invalid canonical value type."""
        with pytest.raises(ValueError):
            self.meter_unit.from_canonical_value("invalid")
        
        with pytest.raises(ValueError):
            self.meter_unit.to_canonical_value("invalid")


class TestUnitCompatibility:
    """Test unit compatibility checking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.meter_unit = Unit.parse_string("m")
        self.kilometer_unit = Unit.parse_string("km")
        self.second_unit = Unit.parse_string("s")
        self.mass_dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
    
    def test_compatible_same_unit(self):
        """Test compatibility with same unit."""
        assert Unit.compatible(self.meter_unit, self.meter_unit)
    
    def test_compatible_different_units_same_dimension(self):
        """Test compatibility with different units of same dimension."""
        assert Unit.compatible(self.meter_unit, self.kilometer_unit)
    
    def test_compatible_different_dimensions(self):
        """Test compatibility with different dimensions."""
        assert not Unit.compatible(self.meter_unit, self.second_unit)
    
    def test_compatible_with_dimension(self):
        """Test compatibility with dimension."""
        assert Unit.compatible(self.meter_unit, self.meter_unit.dimension)
        assert not Unit.compatible(self.meter_unit, self.second_unit.dimension)
    
    def test_compatible_multiple_units(self):
        """Test compatibility with multiple units."""
        assert Unit.compatible(self.meter_unit, self.kilometer_unit, self.meter_unit.dimension)
        assert not Unit.compatible(self.meter_unit, self.kilometer_unit, self.second_unit)
    
    def test_compatible_no_arguments(self):
        """Test compatibility with no arguments."""
        assert Unit.compatible()
    
    def test_compatible_single_argument(self):
        """Test compatibility with single argument."""
        assert Unit.compatible(self.meter_unit)
        assert Unit.compatible(self.meter_unit.dimension)


class TestUnitStringRepresentation:
    """Test string representation of units."""
    
    def test_str_simple_unit(self):
        """Test string representation of simple unit."""
        unit = Unit.parse_string("m")
        assert str(unit) == "m"
    
    def test_str_unit_with_prefix(self):
        """Test string representation of unit with prefix."""
        unit = Unit.parse_string("km")
        assert str(unit) == "km"
    
    def test_str_composite_unit(self):
        """Test string representation of composite unit."""
        unit = Unit.parse_string("m/s")
        assert str(unit) == "m/s"
    
    def test_str_complex_unit(self):
        """Test string representation of complex unit."""
        unit = Unit.parse_string("kg*m/s^2")
        result = str(unit)
        assert "kg" in result
        assert "m" in result
        assert "s" in result
    
    def test_str_dimensionless(self):
        """Test string representation of dimensionless unit."""
        unit = Unit.parse_string("")
        assert str(unit) == ""
    
    def test_reduced_unit(self):
        """Test reduced unit representation."""
        unit = Unit.parse_string("kg*m/s^2")
        reduced = unit.reduced_unit()
        assert reduced.dimension == unit.dimension
        # Should have same dimension but potentially different representation
    
    def test_format_string_no_fraction(self):
        """Test format_string with no_fraction=True."""
        unit = Unit.parse_string("m/s")
        formatted = unit.format_string(no_fraction=True)
        # Should avoid fraction notation if possible
        assert len(formatted) > 0
    
    def test_format_string_with_fraction(self):
        """Test format_string with no_fraction=False."""
        unit = Unit.parse_string("m/s")
        formatted = unit.format_string(no_fraction=False)
        # May use fraction notation
        assert len(formatted) > 0


class TestUnitSuggestions:
    """Test unit suggestion functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dimension = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        self.length_dimension = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
    
    def test_suggest_units_mass(self):
        """Test unit suggestions for mass dimension."""
        best_unit, suggestions = Unit.suggest_units(self.mass_dimension)
        
        assert best_unit is not None
        assert len(suggestions) > 0
        assert all(unit.dimension == self.mass_dimension for unit in suggestions)
    
    def test_suggest_units_length(self):
        """Test unit suggestions for length dimension."""
        best_unit, suggestions = Unit.suggest_units(self.length_dimension)
        
        assert best_unit is not None
        assert len(suggestions) > 0
        assert all(unit.dimension == self.length_dimension for unit in suggestions)
    
    def test_suggest_units_with_canonical_value(self):
        """Test unit suggestions with canonical value."""
        best_unit, suggestions = Unit.suggest_units(self.mass_dimension, canonical_value=1000.0)
        
        assert best_unit is not None
        assert len(suggestions) > 0
    
    def test_suggest_units_with_must_include(self):
        """Test unit suggestions with must_include constraint."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        kg_element = SimpleUnitElement.parse_string("kg", "nominator")
        best_unit, suggestions = Unit.suggest_units(
            self.mass_dimension, 
            must_include=[kg_element]
        )
        
        assert best_unit is not None
        # Should include kg in suggestions
        kg_units = [unit for unit in suggestions if any(elem.unit_symbol.value.string == "kg" for elem in unit.unit_elements)]
        assert len(kg_units) > 0
    
    def test_suggest_units_with_limit(self):
        """Test unit suggestions with limit."""
        best_unit, suggestions = Unit.suggest_units(self.mass_dimension, n=5)
        
        assert best_unit is not None
        assert len(suggestions) <= 5
    
    def test_suggest_units_dimensionless(self):
        """Test unit suggestions for dimensionless quantity."""
        dimensionless = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        best_unit, suggestions = Unit.suggest_units(dimensionless)
        
        assert best_unit is not None
        assert len(suggestions) > 0


class TestUnitSerialization:
    """Test serialization methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.meter_unit = Unit.parse_string("m")
    
    def test_to_json(self):
        """Test to_json method."""
        json_data = self.meter_unit.to_json()
        
        assert "unit_elements" in json_data
        assert "factor" in json_data
        assert "offset" in json_data
        assert "dimension" in json_data
    
    def test_from_json(self):
        """Test from_json method."""
        json_data = self.meter_unit.to_json()
        unit = Unit.from_json(json_data)
        
        assert unit.unit_elements == self.meter_unit.unit_elements
        assert unit.factor == self.meter_unit.factor
        assert unit.offset == self.meter_unit.offset
    
    def test_to_hdf5(self):
        """Test to_hdf5 method."""
        with h5py.File("test_unit.h5", "w") as f:
            group = f.create_group("unit")
            self.meter_unit.to_hdf5(group)
            
            # Verify data was written
            assert "unit_elements" in group
            assert "factor" in group
            assert "offset" in group
            assert "dimension" in group
    
    def test_from_hdf5(self):
        """Test from_hdf5 method."""
        # First write to file
        with h5py.File("test_unit.h5", "w") as f:
            group = f.create_group("unit")
            self.meter_unit.to_hdf5(group)
        
        # Then read from file
        with h5py.File("test_unit.h5", "r") as f:
            group = f["unit"]
            unit = Unit.from_hdf5(group)
            
            assert unit.unit_elements == self.meter_unit.unit_elements
            assert unit.factor == self.meter_unit.factor
            assert unit.offset == self.meter_unit.offset


class TestUnitEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_factor(self):
        """Test unit with very large factor."""
        # Create unit with large factor through prefix
        unit = Unit.parse_string("Ym")  # Yottameter
        assert unit.factor > 1e20
    
    def test_very_small_factor(self):
        """Test unit with very small factor."""
        # Create unit with small factor through prefix
        unit = Unit.parse_string("ym")  # Yoctometer
        assert unit.factor < 1e-20
    
    def test_zero_factor(self):
        """Test unit with zero factor."""
        # This should not be possible with valid units, but test edge case
        with pytest.raises(ValueError):
            # Try to create invalid unit
            pass
    
    def test_very_large_offset(self):
        """Test unit with very large offset."""
        # Temperature units can have large offsets
        celsius_unit = Unit.parse_string("°C")
        assert celsius_unit.offset > 200  # Celsius offset from Kelvin
    
    def test_immutability(self):
        """Test that units are immutable."""
        unit = Unit.parse_string("m")
        
        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            unit.factor = 2.0
        
        with pytest.raises(AttributeError):
            unit.unit_elements = ()
    
    def test_hash_consistency(self):
        """Test that equal units have same hash."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("m")
        assert hash(unit1) == hash(unit2)
    
    def test_equality_consistency(self):
        """Test unit equality."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("m")
        unit3 = Unit.parse_string("s")
        
        assert unit1 == unit2
        assert unit1 != unit3
        assert unit1 != "not a unit"


class TestUnitPerformance:
    """Test performance characteristics."""
    
    def test_parsing_performance(self):
        """Test that unit parsing is fast."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            Unit.parse_string("kg*m/s^2")
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
    
    def test_conversion_performance(self):
        """Test that unit conversions are fast."""
        import time
        
        unit = Unit.parse_string("km")
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)  # 1000 values
        
        start_time = time.time()
        for _ in range(100):
            unit.from_canonical_value(values)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
    
    def test_suggestion_performance(self):
        """Test that unit suggestions are reasonably fast."""
        import time
        
        mass_dim = Dimension.create([1, 0, 0, 0, 0, 0, 0], [0, 0])
        
        start_time = time.time()
        Unit.suggest_units(mass_dim, n=10)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0


if __name__ == "__main__":
    pytest.main([__file__])
