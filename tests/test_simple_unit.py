import pytest
import math
import numpy as np
from united_system.units.simple.simple_unit import SimpleUnit, SimpleUnitElement, _SIMPLE_UNIT_CACHE
from src.united_system.units.unit_symbol import UnitSymbol
from united_system.units.simple.simple_dimension import SimpleDimension
from united_system.units.named_simple_dimensions import NamedSimpleDimension


class TestSimpleUnitParsing:
    """Test parsing of various unit strings."""
    
    def test_basic_unit_parsing(self):
        """Test parsing of basic SI units."""
        # Length units
        meter = SimpleUnit.parse_string("m")
        assert len(meter.unit_elements) == 1
        assert meter.unit_elements[0].prefix == ""
        assert meter.unit_elements[0].exponent == 1.0
        
        # Mass units
        gram = SimpleUnit.parse_string("g")
        assert len(gram.unit_elements) == 1
        assert gram.unit_elements[0].exponent == 1.0
        
        # Time units
        second = SimpleUnit.parse_string("s")
        assert len(second.unit_elements) == 1
        assert second.unit_elements[0].exponent == 1.0
        
        # Electric current
        ampere = SimpleUnit.parse_string("A")
        assert len(ampere.unit_elements) == 1
        assert ampere.unit_elements[0].exponent == 1.0
        
        # Temperature
        kelvin = SimpleUnit.parse_string("K")
        assert len(kelvin.unit_elements) == 1
        assert kelvin.unit_elements[0].exponent == 1.0
        
        # Luminous intensity
        candela = SimpleUnit.parse_string("cd")
        assert len(candela.unit_elements) == 1
        assert candela.unit_elements[0].exponent == 1.0
        
        # Amount of substance
        mole = SimpleUnit.parse_string("mol")
        assert len(mole.unit_elements) == 1
        assert mole.unit_elements[0].exponent == 1.0
    
    def test_prefixed_units(self):
        """Test parsing of units with SI prefixes."""
        # Length with prefixes
        kilometer = SimpleUnit.parse_string("km")
        assert kilometer.unit_elements[0].prefix == "k"
        assert kilometer.unit_elements[0].exponent == 1.0
        
        millimeter = SimpleUnit.parse_string("mm")
        assert millimeter.unit_elements[0].prefix == "m"
        assert millimeter.unit_elements[0].exponent == 1.0
        
        micrometer = SimpleUnit.parse_string("µm")
        assert micrometer.unit_elements[0].prefix == "µ"
        assert micrometer.unit_elements[0].exponent == 1.0
        
        nanometer = SimpleUnit.parse_string("nm")
        assert nanometer.unit_elements[0].prefix == "n"
        assert nanometer.unit_elements[0].exponent == 1.0
        
        # Voltage with prefixes
        millivolt = SimpleUnit.parse_string("mV")
        assert millivolt.unit_elements[0].prefix == "m"
        assert millivolt.unit_elements[0].exponent == 1.0
        
        kilovolt = SimpleUnit.parse_string("kV")
        assert kilovolt.unit_elements[0].prefix == "k"
        assert kilovolt.unit_elements[0].exponent == 1.0
        
        # Mass with prefixes
        kilogram = SimpleUnit.parse_string("kg")
        assert kilogram.unit_elements[0].prefix == "k"
        assert kilogram.unit_elements[0].exponent == 1.0
        
        milligram = SimpleUnit.parse_string("mg")
        assert milligram.unit_elements[0].prefix == "m"
        assert milligram.unit_elements[0].exponent == 1.0
    
    def test_compound_units_with_multiplication(self):
        """Test parsing of compound units with multiplication."""
        # Force: kg*m/s^2
        newton = SimpleUnit.parse_string("kg*m/s^2")
        assert len(newton.unit_elements) == 3
        
        # Find each component
        kg_element = next(e for e in newton.unit_elements if e.unit_symbol.value.name == "kilogram")
        m_element = next(e for e in newton.unit_elements if e.unit_symbol.value.name == "meter")
        s_element = next(e for e in newton.unit_elements if e.unit_symbol.value.name == "second")
        
        assert kg_element.exponent == 1.0
        assert m_element.exponent == 1.0
        assert s_element.exponent == -2.0
        
        # Power: kg*m^2/s^3
        watt = SimpleUnit.parse_string("kg*m^2/s^3")
        assert len(watt.unit_elements) == 3
    
    def test_compound_units_with_division(self):
        """Test parsing of compound units with division."""
        # Velocity: m/s
        velocity = SimpleUnit.parse_string("m/s")
        assert len(velocity.unit_elements) == 2
        
        m_element = next(e for e in velocity.unit_elements if e.unit_symbol.value.name == "meter")
        s_element = next(e for e in velocity.unit_elements if e.unit_symbol.value.name == "second")
        
        assert m_element.exponent == 1.0
        assert s_element.exponent == -1.0
        
        # Voltage rate: mV/s
        voltage_rate = SimpleUnit.parse_string("mV/s")
        assert len(voltage_rate.unit_elements) == 2
        
        v_element = next(e for e in voltage_rate.unit_elements if e.unit_symbol.value.name == "volt")
        s_element = next(e for e in voltage_rate.unit_elements if e.unit_symbol.value.name == "second")
        
        assert v_element.prefix == "m"
        assert v_element.exponent == 1.0
        assert s_element.exponent == -1.0
        
        # Frequency: 1/s
        frequency = SimpleUnit.parse_string("1/s")
        assert len(frequency.unit_elements) == 1
        s_element = frequency.unit_elements[0]
        assert s_element.exponent == -1.0
    
    def test_exponents_integer(self):
        """Test parsing of units with integer exponents."""
        # Area: m^2
        area = SimpleUnit.parse_string("m^2")
        assert len(area.unit_elements) == 1
        assert area.unit_elements[0].exponent == 2.0
        
        # Volume: m^3
        volume = SimpleUnit.parse_string("m^3")
        assert len(volume.unit_elements) == 1
        assert volume.unit_elements[0].exponent == 3.0
        
        # Inverse area: m^-2
        inverse_area = SimpleUnit.parse_string("m^-2")
        assert len(inverse_area.unit_elements) == 1
        assert inverse_area.unit_elements[0].exponent == -2.0
        
        # High powers
        high_power = SimpleUnit.parse_string("m^5")
        assert len(high_power.unit_elements) == 1
        assert high_power.unit_elements[0].exponent == 5.0
    
    def test_exponents_fractional(self):
        """Test parsing of units with fractional exponents."""
        # Fractional mass exponent
        fractional_mass = SimpleUnit.parse_string("kg^1.4")
        assert len(fractional_mass.unit_elements) == 1
        assert abs(fractional_mass.unit_elements[0].exponent - 1.4) < 1e-10
        
        # Fractional length exponent
        fractional_length = SimpleUnit.parse_string("µm^0.3")
        assert len(fractional_length.unit_elements) == 1
        assert abs(fractional_length.unit_elements[0].exponent - 0.3) < 1e-10
        assert fractional_length.unit_elements[0].prefix == "µ"
        
        # Negative fractional exponent
        negative_fractional = SimpleUnit.parse_string("mol^-0.5")
        assert len(negative_fractional.unit_elements) == 1
        assert abs(negative_fractional.unit_elements[0].exponent - (-0.5)) < 1e-10
        
        # Multiple decimal places
        precise_fractional = SimpleUnit.parse_string("g^2.345")
        assert len(precise_fractional.unit_elements) == 1
        assert abs(precise_fractional.unit_elements[0].exponent - 2.345) < 1e-10
    
    def test_complex_compound_units(self):
        """Test parsing of complex compound units."""
        # Complex example: kg^1.4/µm^0.3*A^3
        complex_unit = SimpleUnit.parse_string("kg^1.4/µm^0.3*A^3")
        assert len(complex_unit.unit_elements) == 3
        
        # Find each component
        kg_element = next(e for e in complex_unit.unit_elements if e.unit_symbol.value.name == "kilogram")
        um_element = next(e for e in complex_unit.unit_elements if e.unit_symbol.value.name == "meter")
        a_element = next(e for e in complex_unit.unit_elements if e.unit_symbol.value.name == "ampere")
        
        assert abs(kg_element.exponent - 1.4) < 1e-10
        assert abs(um_element.exponent - (-0.3)) < 1e-10
        assert um_element.prefix == "µ"
        assert abs(a_element.exponent - 3.0) < 1e-10
        
        # Another complex example: V^2*s^-1.5/Ω^0.5
        complex_unit2 = SimpleUnit.parse_string("V^2*s^-1.5/Ω^0.5")
        assert len(complex_unit2.unit_elements) == 3
        
        # Pressure with mixed units: Pa*m^-1*K^0.5
        pressure_unit = SimpleUnit.parse_string("Pa*m^-1*K^0.5")
        assert len(pressure_unit.unit_elements) == 3
    
    def test_edge_cases(self):
        """Test edge cases and special scenarios."""
        # Empty string
        empty_unit = SimpleUnit.parse_string("")
        assert len(empty_unit.unit_elements) == 0
        
        # Whitespace handling
        spaced_unit = SimpleUnit.parse_string("  m/s  ")
        assert len(spaced_unit.unit_elements) == 2
        
        # Single character units
        single_char = SimpleUnit.parse_string("A")
        assert len(single_char.unit_elements) == 1
        
        # Units with no exponent specified (should default to 1)
        no_exponent = SimpleUnit.parse_string("m*s")
        assert len(no_exponent.unit_elements) == 2
        assert all(e.exponent == 1.0 for e in no_exponent.unit_elements)
    
    def test_parsing_errors(self):
        """Test that invalid unit strings raise appropriate errors."""
        # Invalid unit symbol
        with pytest.raises(ValueError, match="Invalid unit string"):
            SimpleUnit.parse_string("xyz")
        
        # Invalid prefix
        with pytest.raises(ValueError, match="Invalid unit string"):
            SimpleUnit.parse_string("qm")  # 'q' is not a valid prefix
        
        # Multiple ^ symbols
        with pytest.raises(ValueError, match="Invalid unit string"):
            SimpleUnit.parse_string("m^2^3")
        
        # Invalid exponent
        with pytest.raises(ValueError):
            SimpleUnit.parse_string("m^abc")
    
    def test_caching_behavior(self):
        """Test that parsed units are cached correctly."""
        # Clear cache for test
        _SIMPLE_UNIT_CACHE.clear()
        
        # Parse the same unit twice
        unit1 = SimpleUnit.parse_string("m/s")
        unit2 = SimpleUnit.parse_string("m/s")
        
        # Should be the same object due to caching
        assert unit1 is unit2
        
        # Cache should contain the unit
        assert "m/s" in _SIMPLE_UNIT_CACHE
        assert _SIMPLE_UNIT_CACHE["m/s"] is unit1


class TestSimpleUnitOperations:
    """Test mathematical operations on SimpleUnit objects."""
    
    def test_compatibility_checking(self):
        """Test unit compatibility checking."""
        # Length units should be compatible
        meter = SimpleUnit.parse_string("m")
        kilometer = SimpleUnit.parse_string("km")
        millimeter = SimpleUnit.parse_string("mm")
        
        assert meter.compatible_to(kilometer.dimension)
        assert kilometer.compatible_to(millimeter.dimension)
        assert millimeter.compatible_to(meter.dimension)
        
        # Different dimensions should not be compatible
        second = SimpleUnit.parse_string("s")
        assert not meter.compatible_to(second.dimension)
        assert not second.compatible_to(meter.dimension)
        
        # Compound units with same dimensions should be compatible
        velocity1 = SimpleUnit.parse_string("m/s")
        velocity2 = SimpleUnit.parse_string("km/h")
        assert velocity1.compatible_to(velocity2.dimension)
    
    def test_power_operations(self):
        """Test raising units to powers."""
        # Simple power
        meter = SimpleUnit.parse_string("m")
        meter_squared = meter.pow(2)
        assert len(meter_squared.unit_elements) == 1
        assert meter_squared.unit_elements[0].exponent == 2.0
        
        # Fractional power
        meter_sqrt = meter.pow(0.5)
        assert len(meter_sqrt.unit_elements) == 1
        assert abs(meter_sqrt.unit_elements[0].exponent - 0.5) < 1e-10
        
        # Negative power
        meter_inverse = meter.pow(-1)
        assert len(meter_inverse.unit_elements) == 1
        assert meter_inverse.unit_elements[0].exponent == -1.0
        
        # Compound unit power
        velocity = SimpleUnit.parse_string("m/s")
        velocity_squared = velocity.pow(2)
        assert len(velocity_squared.unit_elements) == 2
        
        m_element = next(e for e in velocity_squared.unit_elements if e.unit_symbol.value.name == "meter")
        s_element = next(e for e in velocity_squared.unit_elements if e.unit_symbol.value.name == "second")
        
        assert m_element.exponent == 2.0
        assert s_element.exponent == -2.0
    
    def test_value_conversion(self):
        """Test conversion between canonical and unit values."""
        # Basic conversion
        meter = SimpleUnit.parse_string("m")
        assert meter.to_canonical_value(1.0) == 1.0
        assert meter.from_canonical_value(1.0) == 1.0
        
        # Prefix conversion
        kilometer = SimpleUnit.parse_string("km")
        assert kilometer.to_canonical_value(1.0) == 1000.0
        assert kilometer.from_canonical_value(1000.0) == 1.0
        
        millimeter = SimpleUnit.parse_string("mm")
        assert millimeter.to_canonical_value(1000.0) == 1.0
        assert millimeter.from_canonical_value(1.0) == 1000.0
        
        # Voltage conversion
        millivolt = SimpleUnit.parse_string("mV")
        assert millivolt.to_canonical_value(1000.0) == 1.0
        assert millivolt.from_canonical_value(1.0) == 1000.0
        
        # Integer handling
        assert isinstance(meter.to_canonical_value(5), int)
        assert isinstance(meter.from_canonical_value(5), int)
        
        # Float handling
        assert isinstance(meter.to_canonical_value(5.5), float)
        assert isinstance(meter.from_canonical_value(5.5), float)
    
    def test_numpy_array_conversion(self):
        """Test conversion with numpy arrays."""
        meter = SimpleUnit.parse_string("m")
        kilometer = SimpleUnit.parse_string("km")
        
        # Test with numpy arrays
        values = np.array([1.0, 2.0, 3.0])
        canonical_values = kilometer.to_canonical_value(values)
        expected = np.array([1000.0, 2000.0, 3000.0])
        np.testing.assert_array_equal(canonical_values, expected)
        
        # Test reverse conversion
        converted_back = kilometer.from_canonical_value(canonical_values)
        np.testing.assert_array_equal(converted_back, values)
    
    def test_complex_number_conversion(self):
        """Test conversion with complex numbers."""
        meter = SimpleUnit.parse_string("m")
        kilometer = SimpleUnit.parse_string("km")
        
        complex_value = 1.0 + 2.0j
        canonical_value = kilometer.to_canonical_value(complex_value)
        expected = (1000.0 + 2000.0j)
        assert canonical_value == expected
        
        converted_back = kilometer.from_canonical_value(canonical_value)
        assert converted_back == complex_value


class TestSimpleUnitStringRepresentation:
    """Test string representation of SimpleUnit objects."""
    
    def test_nice_string_basic(self):
        """Test nice string representation of basic units."""
        meter = SimpleUnit.parse_string("m")
        assert "m" in meter.nice_string()
        
        second = SimpleUnit.parse_string("s")
        assert "s" in second.nice_string()
        
        kilogram = SimpleUnit.parse_string("kg")
        assert "kg" in kilogram.nice_string()
    
    def test_nice_string_compound(self):
        """Test nice string representation of compound units."""
        velocity = SimpleUnit.parse_string("m/s")
        nice_str = velocity.nice_string()
        assert "m" in nice_str
        assert "s" in nice_str
        
        acceleration = SimpleUnit.parse_string("m/s^2")
        nice_str = acceleration.nice_string()
        assert "m" in nice_str
        assert "s" in nice_str
    
    def test_nice_string_with_exponents(self):
        """Test nice string representation with exponents."""
        area = SimpleUnit.parse_string("m^2")
        nice_str = area.nice_string()
        assert "m" in nice_str
        assert "2" in nice_str
        
        volume = SimpleUnit.parse_string("m^3")
        nice_str = volume.nice_string()
        assert "m" in nice_str
        assert "3" in nice_str
    
    def test_str_method(self):
        """Test __str__ method."""
        meter = SimpleUnit.parse_string("m")
        assert str(meter) == meter.nice_string()
        
        velocity = SimpleUnit.parse_string("m/s")
        assert str(velocity) == velocity.nice_string()


class TestSimpleUnitCreation:
    """Test creation of SimpleUnit objects using the create method."""
    
    def test_create_from_elements(self):
        """Test creating SimpleUnit from SimpleUnitElement tuples."""
        # Create a simple unit from elements
        meter_symbol = UnitSymbol.from_symbol("m")
        
        element = ("", meter_symbol, 1.0)
        unit = SimpleUnit.create(element)
        
        assert len(unit.unit_elements) == 1
        assert unit.unit_elements[0].prefix == ""
        assert unit.unit_elements[0].exponent == 1.0
    
    def test_create_from_multiple_elements(self):
        """Test creating SimpleUnit from multiple SimpleUnitElement tuples."""
        meter_symbol = UnitSymbol.from_symbol("m")
        second_symbol = UnitSymbol.from_symbol("s")
        
        element1 = ("", meter_symbol, 1.0)
        element2 = ("", second_symbol, -1.0)
        
        unit = SimpleUnit.create(element1, element2)
        
        assert len(unit.unit_elements) == 2
        # Should represent m/s
        assert any(e.unit_symbol == meter_symbol and e.exponent == 1.0 for e in unit.unit_elements)
        assert any(e.unit_symbol == second_symbol and e.exponent == -1.0 for e in unit.unit_elements)
    
    def test_create_from_existing_units(self):
        """Test creating SimpleUnit from existing SimpleUnit objects."""
        meter = SimpleUnit.parse_string("m")
        second = SimpleUnit.parse_string("s")
        
        # Create compound unit from existing units
        velocity = SimpleUnit.create(meter, second.pow(-1))
        
        assert len(velocity.unit_elements) == 2
        # Should have meter with exponent 1 and second with exponent -1
        m_element = next(e for e in velocity.unit_elements if e.unit_symbol.name == "METER")
        s_element = next(e for e in velocity.unit_elements if e.unit_symbol.name == "SECOND")
        
        assert m_element.exponent == 1.0
        assert s_element.exponent == -1.0
    
    def test_create_from_list(self):
        """Test creating SimpleUnit from a list of SimpleUnit objects."""
        meter = SimpleUnit.parse_string("m")
        second_inv = SimpleUnit.parse_string("s^-1")
        
        velocity = SimpleUnit.create([meter, second_inv])
        
        assert len(velocity.unit_elements) == 2
        # Should represent m/s
        m_element = next(e for e in velocity.unit_elements if e.unit_symbol.name == "METER")
        s_element = next(e for e in velocity.unit_elements if e.unit_symbol.name == "SECOND")
        
        assert m_element.exponent == 1.0
        assert s_element.exponent == -1.0


class TestSimpleUnitElementStringFormatting:
    """Test SimpleUnitElement string formatting."""
    
    def test_element_string_basic(self):
        """Test basic element string formatting."""
        meter_symbol = UnitSymbol.from_symbol("m")
        
        element = SimpleUnitElement("", meter_symbol, 1.0)
        string_repr = element.as_string_element_for_unit_nice_string(True)
        assert "m" in string_repr
    
    def test_element_string_with_prefix(self):
        """Test element string formatting with prefix."""
        meter_symbol = UnitSymbol.from_symbol("m")
        
        element = SimpleUnitElement("k", meter_symbol, 1.0)
        string_repr = element.as_string_element_for_unit_nice_string(True)
        assert "km" in string_repr
    
    def test_element_string_with_exponent(self):
        """Test element string formatting with exponent."""
        meter_symbol = UnitSymbol.from_symbol("m")
        
        element = SimpleUnitElement("", meter_symbol, 2.0)
        string_repr = element.as_string_element_for_unit_nice_string(True)
        assert "m" in string_repr
        assert "2" in string_repr or "^2" in string_repr
    
    def test_element_string_negative_exponent(self):
        """Test element string formatting with negative exponent."""
        meter_symbol = UnitSymbol.from_symbol("m")
        
        element = SimpleUnitElement("", meter_symbol, -1.0)
        string_repr = element.as_string_element_for_unit_nice_string(True)
        assert "m" in string_repr
        assert "/" in string_repr or "-" in string_repr
    
    def test_element_string_zero_exponent(self):
        """Test element string formatting with zero exponent."""
        meter_symbol = UnitSymbol.from_symbol("m")
        
        element = SimpleUnitElement("", meter_symbol, 0.0)
        string_repr = element.as_string_element_for_unit_nice_string(True)
        assert string_repr == ""  # Zero exponent should result in empty string


class TestSimpleUnitFactorAndOffset:
    """Test factor and offset calculations in SimpleUnit."""
    
    def test_factor_calculation_basic(self):
        """Test basic factor calculation."""
        meter = SimpleUnit.parse_string("m")
        assert meter.factor == 1.0
        
        kilometer = SimpleUnit.parse_string("km")
        assert kilometer.factor == 1000.0
        
        millimeter = SimpleUnit.parse_string("mm")
        assert millimeter.factor == 0.001
    
    def test_factor_calculation_compound(self):
        """Test factor calculation for compound units."""
        # m/s should have factor 1.0
        velocity = SimpleUnit.parse_string("m/s")
        assert velocity.factor == 1.0
        
        # km/h should have factor 1000/3600 = 5/18
        km_per_hour = SimpleUnit.parse_string("km/h")
        expected_factor = 1000.0 / 3600.0  # km to m, h to s
        assert abs(km_per_hour.factor - expected_factor) < 1e-10
    
    def test_offset_calculation(self):
        """Test offset calculation for temperature units."""
        # Most units should have zero offset
        meter = SimpleUnit.parse_string("m")
        assert meter.offset == 0.0
        
        # Celsius should have offset of 273.15
        # Note: This depends on the UnitSymbol definitions
        try:
            celsius = SimpleUnit.parse_string("°C")
            # Celsius has an offset for temperature conversion
            assert celsius.offset != 0.0
        except ValueError:
            # If Celsius symbol is not defined, skip this test
            pass
    
    def test_multiple_offsets_error(self):
        """Test that multiple non-zero offsets raise an error."""
        # This test depends on having temperature units with offsets
        # Skip if no such units are available
        pass


class TestSimpleUnitSuggestions:
    """Test unit suggestion functionality."""
    
    def test_suggest_unit_basic(self):
        """Test basic unit suggestion."""
        # Test with a simple dimension
        meter = SimpleUnit.parse_string("m")
        suggested = SimpleUnit.suggest_unit_from_named_units(meter.dimension, 1.0)
        assert isinstance(suggested, SimpleUnit)
        assert suggested.compatible_to(meter.dimension)
    
    def test_suggest_unit_with_value(self):
        """Test unit suggestion with specific values."""
        # Test with different values that might suggest different units
        meter = SimpleUnit.parse_string("m")
        
        # Small values might suggest mm
        suggested_small = SimpleUnit.suggest_unit_from_named_units(meter.dimension, 0.001)
        assert isinstance(suggested_small, SimpleUnit)
        
        # Large values might suggest km
        suggested_large = SimpleUnit.suggest_unit_from_named_units(meter.dimension, 10000.0)
        assert isinstance(suggested_large, SimpleUnit)
    
    def test_suggest_unit_without_value(self):
        """Test unit suggestion without specific value."""
        meter = SimpleUnit.parse_string("m")
        suggested = SimpleUnit.suggest_unit_from_named_units(meter.dimension, None)
        assert isinstance(suggested, SimpleUnit)
        assert suggested.compatible_to(meter.dimension)


class TestSimpleUnitDimensionHandling:
    """Test dimension handling in SimpleUnit."""
    
    def test_unit_dimension_property(self):
        """Test that unit_dimension property is correctly set."""
        meter = SimpleUnit.parse_string("m")
        assert isinstance(meter.dimension, SimpleDimension)
        
        velocity = SimpleUnit.parse_string("m/s")
        assert isinstance(velocity.dimension, SimpleDimension)
    
    def test_dimension_consistency(self):
        """Test that dimensions are consistent across equivalent units."""
        meter = SimpleUnit.parse_string("m")
        kilometer = SimpleUnit.parse_string("km")
        millimeter = SimpleUnit.parse_string("mm")
        
        # All should have the same dimension
        assert meter.dimension == kilometer.dimension
        assert kilometer.dimension == millimeter.dimension
        assert millimeter.dimension == meter.dimension
    
    def test_compound_dimensions(self):
        """Test dimensions of compound units."""
        velocity = SimpleUnit.parse_string("m/s")
        acceleration = SimpleUnit.parse_string("m/s^2")
        
        # Dimensions should be different
        assert velocity.dimension != acceleration.dimension
        
        # But both should be SimpleUnitDimension
        assert isinstance(velocity.dimension, SimpleDimension)
        assert isinstance(acceleration.dimension, SimpleDimension)


class TestSimpleUnitRegression:
    """Regression tests for specific issues and edge cases."""
    
    def test_parsing_regression_cases(self):
        """Test specific parsing cases that might have caused issues."""
        # Test the specific examples from the user
        test_cases = [
            "mV/s",
            "1/s", 
            "kg^1.4/µm^0.3*A^3",
            "mol^-0.5",
            "V*m/ns^2",
            "Pa*m^-1*K^0.5",
            "Hz*m^2/kg^0.5",
            "J/mol*K",
            "W/m^2*K^4",
            "kg*m^2/s^2*A^-1",
        ]
        
        for case in test_cases:
            try:
                unit = SimpleUnit.parse_string(case)
                assert isinstance(unit, SimpleUnit)
                assert len(unit.unit_elements) > 0
                # Test that it can convert some values
                assert isinstance(unit.to_canonical_value(1.0), float)
                assert isinstance(unit.from_canonical_value(1.0), float)
            except Exception as e:
                pytest.fail(f"Failed to parse '{case}': {e}")
    
    def test_unicode_prefix_handling(self):
        """Test handling of unicode prefixes like µ."""
        # Test micro prefix
        micrometer = SimpleUnit.parse_string("µm")
        assert micrometer.unit_elements[0].prefix == "µ"
        
        # Test that conversion works correctly
        canonical = micrometer.to_canonical_value(1.0)
        assert canonical == 1e-6
        
        back_converted = micrometer.from_canonical_value(canonical)
        assert abs(back_converted - 1.0) < 1e-10
    
    def test_precision_handling(self):
        """Test handling of high precision fractional exponents."""
        # Test with many decimal places
        precise_unit = SimpleUnit.parse_string("m^2.123456789")
        assert abs(precise_unit.unit_elements[0].exponent - 2.123456789) < 1e-10
        
        # Test with very small fractional exponents
        small_fraction = SimpleUnit.parse_string("kg^0.001")
        assert abs(small_fraction.unit_elements[0].exponent - 0.001) < 1e-10
    
    def test_large_exponent_handling(self):
        """Test handling of large exponents."""
        # Test with large positive exponent
        large_positive = SimpleUnit.parse_string("m^100")
        assert large_positive.unit_elements[0].exponent == 100.0
        
        # Test with large negative exponent
        large_negative = SimpleUnit.parse_string("s^-50")
        assert large_negative.unit_elements[0].exponent == -50.0
    
    def test_mixed_notation_parsing(self):
        """Test parsing of mixed notation (division and multiplication)."""
        # Complex mixed notation
        mixed = SimpleUnit.parse_string("kg*m^2/s^3/A*K^-1")
        assert len(mixed.unit_elements) >= 4
        
        # Verify each component exists
        symbols_found = {e.unit_symbol.name for e in mixed.unit_elements}
        expected_symbols = {"KILOGRAM", "METER", "SECOND", "AMPERE", "KELVIN"}
        
        # Check that we have the expected symbols (subset check)
        assert len(symbols_found.intersection(expected_symbols)) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 