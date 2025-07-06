import pytest
from united_system.units.simple.simple_unit_element import SimpleUnitElement
from src.united_system.units.unit_symbol import UnitSymbol


class TestSimpleUnitElementParsing:
    """Test the SimpleUnitElement.parse method"""
    
    def test_basic_unit_parsing_nominator(self):
        """Test parsing basic units as nominator"""
        # Test meter
        element = SimpleUnitElement.parse_string("m", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 1.0
        
        # Test gram
        element = SimpleUnitElement.parse_string("g", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 1.0
        
        # Test second
        element = SimpleUnitElement.parse_string("s", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.SECOND
        assert element.exponent == 1.0
    
    def test_basic_unit_parsing_denominator(self):
        """Test parsing basic units as denominator"""
        # Test second as denominator
        element = SimpleUnitElement.parse_string("s", "denominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.SECOND
        assert element.exponent == -1.0
        
        # Test meter as denominator
        element = SimpleUnitElement.parse_string("m", "denominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == -1.0
    
    def test_prefixed_unit_parsing(self):
        """Test parsing units with prefixes"""
        # Test kilometer
        element = SimpleUnitElement.parse_string("km", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 1.0
        
        # Test kilogram
        element = SimpleUnitElement.parse_string("kg", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 1.0
        
        # Test nanovolt
        element = SimpleUnitElement.parse_string("nV", "nominator")
        assert element.prefix == "n"
        assert element.unit_symbol == UnitSymbol.VOLT
        assert element.exponent == 1.0
        
        # Test millivolt
        element = SimpleUnitElement.parse_string("mV", "nominator")
        assert element.prefix == "m"
        assert element.unit_symbol == UnitSymbol.VOLT
        assert element.exponent == 1.0
    
    def test_units_with_integer_exponents(self):
        """Test parsing units with integer exponents"""
        # Test square meter
        element = SimpleUnitElement.parse_string("m^2", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 2.0
        
        # Test cubic meter
        element = SimpleUnitElement.parse_string("m^3", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 3.0
        
        # Test joule cubed as denominator
        element = SimpleUnitElement.parse_string("J^3", "denominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.JOULE
        assert element.exponent == -3.0
        
        # Test nanovolt to the 4th power
        element = SimpleUnitElement.parse_string("nV^4", "nominator")
        assert element.prefix == "n"
        assert element.unit_symbol == UnitSymbol.VOLT
        assert element.exponent == 4.0
    
    def test_units_with_negative_exponents(self):
        """Test parsing units with negative exponents"""
        # Test inverse meter
        element = SimpleUnitElement.parse_string("m^-1", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == -1.0
        
        # Test inverse second squared
        element = SimpleUnitElement.parse_string("s^-2", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.SECOND
        assert element.exponent == -2.0
    
    def test_units_with_fractional_exponents(self):
        """Test parsing units with fractional exponents"""
        # Test square root of meter
        element = SimpleUnitElement.parse_string("m^0.5", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 0.5
        
        # Test gram to the power of 2.5
        element = SimpleUnitElement.parse_string("g^2.5", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 2.5
        
        # Test negative fractional exponent
        element = SimpleUnitElement.parse_string("kg^-0.5", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == -0.5
    
    def test_complex_prefixed_units_with_exponents(self):
        """Test parsing complex prefixed units with exponents"""
        # Test micrometer cubed
        element = SimpleUnitElement.parse_string("µm^3", "nominator")
        assert element.prefix == "µ"
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 3.0
        
        # Test kilogram to the power of 1.4
        element = SimpleUnitElement.parse_string("kg^1.4", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 1.4
        
        # Test microamp squared as denominator
        element = SimpleUnitElement.parse_string("µA^2", "denominator")
        assert element.prefix == "µ"
        assert element.unit_symbol == UnitSymbol.AMPERE
        assert element.exponent == -2.0
    
    def test_units_with_digit_suffixes(self):
        """Test parsing units with digit suffixes like 'cm2', 'm3'"""
        # Test square centimeter
        element = SimpleUnitElement.parse_string("cm2", "nominator")
        assert element.prefix == "c"
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 2.0
        
        # Test cubic meter
        element = SimpleUnitElement.parse_string("m3", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 3.0
        
        # Test square millimeter (milli prefix + meter)
        element = SimpleUnitElement.parse_string("mm2", "nominator")
        assert element.prefix == "m"
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == 2.0
        
        # Test kilogram to the 4th power
        element = SimpleUnitElement.parse_string("kg4", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 4.0
    
    def test_units_with_negative_digit_suffixes(self):
        """Test parsing units with negative digit suffixes like 'nV-1', 'm-2'"""
        # Test inverse nanovolt
        element = SimpleUnitElement.parse_string("nV-1", "nominator")
        assert element.prefix == "n"
        assert element.unit_symbol == UnitSymbol.VOLT
        assert element.exponent == -1.0
        
        # Test inverse square meter
        element = SimpleUnitElement.parse_string("m-2", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == -2.0
        
        # Test inverse kilogram
        element = SimpleUnitElement.parse_string("kg-1", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == -1.0
    
    def test_units_with_fractional_digit_suffixes(self):
        """Test parsing units with fractional digit suffixes like 'kg1.5', 'm-0.5'"""
        # Test kilogram to the 1.5 power
        element = SimpleUnitElement.parse_string("kg1.5", "nominator")
        assert element.prefix == "k"
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 1.5
        
        # Test inverse square root of meter
        element = SimpleUnitElement.parse_string("m-0.5", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == -0.5
        
        # Test gram to the 2.25 power
        element = SimpleUnitElement.parse_string("g2.25", "nominator")
        assert element.prefix == ""
        assert element.unit_symbol == UnitSymbol.GRAM
        assert element.exponent == 2.25
    
    def test_digit_suffix_with_denominator(self):
        """Test that digit suffixes work correctly with denominator designation"""
        # Test cm2 in denominator (should become -2.0)
        element = SimpleUnitElement.parse_string("cm2", "denominator")
        assert element.prefix == "c"
        assert element.unit_symbol == UnitSymbol.METER
        assert element.exponent == -2.0
        
        # Test nV-1 in denominator (should become 1.0)
        element = SimpleUnitElement.parse_string("nV-1", "denominator")
        assert element.prefix == "n"
        assert element.unit_symbol == UnitSymbol.VOLT
        assert element.exponent == 1.0
    
    def test_parsing_errors(self):
        """Test that parsing raises appropriate errors"""
        # Test invalid unit symbol
        with pytest.raises(ValueError, match="Could not parse unit string"):
            SimpleUnitElement.parse_string("xyz", "nominator")
        
        # Test invalid exponent
        with pytest.raises(ValueError, match="Invalid exponent"):
            SimpleUnitElement.parse_string("m^abc", "nominator")
        
        # Test multiple exponents
        with pytest.raises(ValueError, match="Invalid unit string"):
            SimpleUnitElement.parse_string("m^2^3", "nominator")
        
        # Test invalid digit suffix (should not match malformed numbers)
        with pytest.raises(ValueError, match="Could not parse unit string"):
            SimpleUnitElement.parse_string("m2.5.3", "nominator")


class TestSimpleUnitElementStringFormatting:
    """Test the SimpleUnitElement.format_string method"""
    
    def test_basic_unit_formatting_no_fraction(self):
        """Test formatting basic units without fractions"""
        # Test meter
        element = SimpleUnitElement("", UnitSymbol.METER, 1.0)
        result = element.format_string(no_fraction=False)
        assert result == ("m", "nominator")
        
        # Test gram
        element = SimpleUnitElement("", UnitSymbol.GRAM, 1.0)
        result = element.format_string(no_fraction=False)
        assert result == ("g", "nominator")
        
        # Test second in denominator
        element = SimpleUnitElement("", UnitSymbol.SECOND, -1.0)
        result = element.format_string(no_fraction=False)
        assert result == ("s", "denominator")
    
    def test_prefixed_unit_formatting(self):
        """Test formatting prefixed units"""
        # Test kilometer
        element = SimpleUnitElement("k", UnitSymbol.METER, 1.0)
        result = element.format_string(no_fraction=False)
        assert result == ("km", "nominator")
        
        # Test kilogram
        element = SimpleUnitElement("k", UnitSymbol.GRAM, 1.0)
        result = element.format_string(no_fraction=False)
        assert result == ("kg", "nominator")
        
        # Test nanovolt
        element = SimpleUnitElement("n", UnitSymbol.VOLT, 1.0)
        result = element.format_string(no_fraction=False)
        assert result == ("nV", "nominator")
    
    def test_units_with_integer_exponents(self):
        """Test formatting units with integer exponents"""
        # Test square meter
        element = SimpleUnitElement("", UnitSymbol.METER, 2.0)
        result = element.format_string(no_fraction=False)
        assert result == ("m^2", "nominator")
        
        # Test cubic meter
        element = SimpleUnitElement("", UnitSymbol.METER, 3.0)
        result = element.format_string(no_fraction=False)
        assert result == ("m^3", "nominator")
        
        # Test joule cubed as denominator
        element = SimpleUnitElement("", UnitSymbol.JOULE, -3.0)
        result = element.format_string(no_fraction=False)
        assert result == ("J^3", "denominator")
        
        # Test nanovolt to the 4th power
        element = SimpleUnitElement("n", UnitSymbol.VOLT, 4.0)
        result = element.format_string(no_fraction=False)
        assert result == ("nV^4", "nominator")
    
    def test_units_with_negative_exponents_no_fraction(self):
        """Test formatting units with negative exponents using no_fraction=True"""
        # Test inverse meter
        element = SimpleUnitElement("", UnitSymbol.METER, -1.0)
        result = element.format_string(no_fraction=True)
        assert result == ("m^-1", "nominator")
        
        # Test inverse joule cubed
        element = SimpleUnitElement("", UnitSymbol.JOULE, -3.0)
        result = element.format_string(no_fraction=True)
        assert result == ("J^-3", "nominator")
    
    def test_units_with_fractional_exponents(self):
        """Test formatting units with fractional exponents"""
        # Test square root of meter
        element = SimpleUnitElement("", UnitSymbol.METER, 0.5)
        result = element.format_string(no_fraction=False)
        assert result == ("m^0.5", "nominator")
        
        # Test gram to the power of 2.5
        element = SimpleUnitElement("", UnitSymbol.GRAM, 2.5)
        result = element.format_string(no_fraction=False)
        assert result == ("g^2.5", "nominator")
        
        # Test negative fractional exponent
        element = SimpleUnitElement("k", UnitSymbol.GRAM, -0.5)
        result = element.format_string(no_fraction=False)
        assert result == ("kg^0.5", "denominator")
    
    def test_zero_exponent_formatting(self):
        """Test formatting units with zero exponent"""
        element = SimpleUnitElement("", UnitSymbol.METER, 0.0)
        result = element.format_string(no_fraction=False)
        assert result == ("", "nominator")
    
    def test_decimal_precision_formatting(self):
        """Test formatting units with high precision decimals"""
        # Test with more than 3 decimal places (should be truncated, not rounded)
        element = SimpleUnitElement("", UnitSymbol.METER, 1.23456789)
        result = element.format_string(no_fraction=False)
        assert result == ("m^1.234", "nominator")
        
        # Test with trailing zeros (should be removed)
        element = SimpleUnitElement("", UnitSymbol.METER, 2.500)
        result = element.format_string(no_fraction=False)
        assert result == ("m^2.5", "nominator")


class TestSimpleUnitElementRoundTrip:
    """Test that parsing and formatting are consistent"""
    
    def test_round_trip_basic_units(self):
        """Test round-trip conversion for basic units"""
        test_cases = [
            ("m", "nominator"),
            ("g", "nominator"),
            ("s", "denominator"),
            ("km", "nominator"),
            ("kg", "nominator"),
            ("nV", "nominator"),
        ]
        
        for unit_str, nom_or_denom in test_cases:
            # Parse the unit
            element = SimpleUnitElement.parse_string(unit_str, nom_or_denom)
            
            # Format it back
            formatted, direction = element.format_string(no_fraction=False)
            
            # Check that we get back what we expect
            assert formatted == unit_str
            assert direction == nom_or_denom
    
    def test_round_trip_with_exponents(self):
        """Test round-trip conversion for units with exponents"""
        test_cases = [
            ("m^2", "nominator"),
            ("m^3", "nominator"),
            ("J^3", "denominator"),
            ("nV^4", "nominator"),
            ("kg^2", "nominator"),
        ]
        
        for unit_str, nom_or_denom in test_cases:
            # Parse the unit
            element = SimpleUnitElement.parse_string(unit_str, nom_or_denom)
            
            # Format it back
            formatted, direction = element.format_string(no_fraction=False)
            
            # Check that we get back what we expect
            assert formatted == unit_str
            assert direction == nom_or_denom
    
    def test_digit_suffix_formats_equivalence(self):
        """Test that digit suffix formats are equivalent to ^ formats"""
        equivalence_cases = [
            ("cm2", "cm^2"),
            ("m3", "m^3"),
            ("mm2", "mm^2"),
            ("kg4", "kg^4"),
            ("nV-1", "nV^-1"),
            ("m-2", "m^-2"),
            ("kg1.5", "kg^1.5"),
            ("g-0.5", "g^-0.5"),
        ]
        
        for digit_format, caret_format in equivalence_cases:
            # Parse both formats
            element1 = SimpleUnitElement.parse_string(digit_format, "nominator")
            element2 = SimpleUnitElement.parse_string(caret_format, "nominator")
            
            # They should be identical
            assert element1 == element2
            
            # Test with denominator too
            element3 = SimpleUnitElement.parse_string(digit_format, "denominator")
            element4 = SimpleUnitElement.parse_string(caret_format, "denominator")
            
            assert element3 == element4


class TestSimpleUnitElementProperties:
    """Test the properties of SimpleUnitElement"""
    
    def test_unit_dimension_property(self):
        """Test that unit_dimension property returns correct dimension"""
        element = SimpleUnitElement("", UnitSymbol.METER, 1.0)
        dimension = element.unit_dimension
        # Should be length dimension [0, 0, 1, 0, 0, 0, 0] (3rd position is length)
        assert dimension.dimension_exponents == (0, 0, 1, 0, 0, 0, 0)
    
    def test_canonical_factor_property(self):
        """Test that canonical_factor property calculates correctly"""
        # Test meter (no prefix)
        element = SimpleUnitElement("", UnitSymbol.METER, 1.0)
        assert element.canonical_factor == 1.0
        
        # Test kilometer (with prefix)
        element = SimpleUnitElement("k", UnitSymbol.METER, 1.0)
        assert element.canonical_factor == 1000.0
        
        # Test square kilometer
        element = SimpleUnitElement("k", UnitSymbol.METER, 2.0)
        assert element.canonical_factor == 1000000.0  # 1000^2
        
        # Test milligram (gram=0.001 kg, milli=0.001, so milligram=0.001*0.001=1e-06)
        element = SimpleUnitElement("m", UnitSymbol.GRAM, 1.0)
        assert element.canonical_factor == 1e-06
    
    def test_canonical_offset_property(self):
        """Test that canonical_offset property returns correct offset"""
        # Most units have zero offset
        element = SimpleUnitElement("", UnitSymbol.METER, 1.0)
        assert element.canonical_offset == 0.0
        
        # Test with a unit that might have offset (like temperature)
        element = SimpleUnitElement("", UnitSymbol.KELVIN, 1.0)
        assert element.canonical_offset == 0.0  # Kelvin has no offset
    
    def test_invalid_prefix_error(self):
        """Test that invalid prefix raises ValueError"""
        element = SimpleUnitElement("invalid", UnitSymbol.METER, 1.0)
        with pytest.raises(ValueError, match="Invalid prefix"):
            _ = element.canonical_factor


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 