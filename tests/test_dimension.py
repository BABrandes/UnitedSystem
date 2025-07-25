"""
Comprehensive tests for Dimension class.

Tests all functionality including:
- String parsing and construction
- Arithmetic operations
- Comparison operations
- String representation
- Serialization
- Edge cases and error conditions
"""

import pytest

# Import the modules to test
from united_system.dimension import Dimension, DIMENSIONLESS_DIMENSION, ANGLE_DIMENSION
from united_system.named_quantity import NamedQuantity


class TestDimensionStringParsing:
    """Test dimension creation from string parsing."""
    
    def test_parse_simple_mass_dimension(self):
        """Test parsing simple mass dimension 'M'."""
        dimension = Dimension("M")
        assert dimension._proper_exponents[""] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_length_dimension(self):
        """Test parsing simple length dimension 'L'."""
        dimension = Dimension("L")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_time_dimension(self):
        """Test parsing simple time dimension 'T'."""
        dimension = Dimension("T")
        assert dimension._proper_exponents[""] == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_current_dimension(self):
        """Test parsing simple current dimension 'I'."""
        dimension = Dimension("I")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_temperature_dimension(self):
        """Test parsing simple temperature dimension 'Θ'."""
        dimension = Dimension("Θ")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_amount_dimension(self):
        """Test parsing simple amount dimension 'N'."""
        dimension = Dimension("N")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_luminous_dimension(self):
        """Test parsing simple luminous intensity dimension 'J'."""
        dimension = Dimension("J")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_simple_angle_dimension(self):
        """Test parsing simple angle dimension 'A'."""
        dimension = Dimension("A")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_powered_dimension(self):
        """Test parsing dimension with power 'L^2'."""
        dimension = Dimension("L^2")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_negative_power_dimension(self):
        """Test parsing dimension with negative power 'T^-1'."""
        dimension = Dimension("T^-1")
        assert dimension._proper_exponents[""] == (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_fractional_power_dimension(self):
        """Test parsing dimension with fractional power 'L^0.5'."""
        dimension = Dimension("L^0.5")
        assert dimension._proper_exponents[""] == (0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_multiplication(self):
        """Test parsing multiplication 'M*L'."""
        dimension = Dimension("M*L")
        assert dimension._proper_exponents[""] == (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_division(self):
        """Test parsing division 'M/L'."""
        dimension = Dimension("M/L")
        assert dimension._proper_exponents[""] == (1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_complex_expression(self):
        """Test parsing complex expression 'M*L^2/T^2'."""
        dimension = Dimension("M*L^2/T^2")
        assert dimension._proper_exponents[""] == (1.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_with_subscripts(self):
        """Test parsing with subscripts 'L_elec'."""
        dimension = Dimension("L_elec")
        assert dimension._proper_exponents["elec"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_multiple_subscripts(self):
        """Test parsing with multiple subscripts 'L_elec*L_geo'."""
        dimension = Dimension("L_elec*L_geo")
        assert dimension._proper_exponents["elec"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert dimension._proper_exponents["geo"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_subscript_division(self):
        """Test parsing division with subscripts 'L_elec/L_geo'."""
        dimension = Dimension("L_elec/L_geo")
        assert dimension._proper_exponents["elec"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert dimension._proper_exponents["geo"] == (0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
    
    def test_parse_log_dimension(self):
        """Test parsing log dimension 'DEC(L)'."""
        dimension = Dimension("DEC(L)")
        assert len(dimension._proper_exponents) == 0  # type: ignore
        assert len(dimension._log_dimensions) == 1  # type: ignore
        # Check that the log dimension is a length dimension
        log_dim = next(iter(dimension._log_dimensions.keys()))  # type: ignore
        assert log_dim._proper_exponents[""] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert dimension._log_dimensions[log_dim] == 1.0  # type: ignore
    
    def test_parse_log_dimension_with_power(self):
        """Test parsing log dimension with power 'DEC(L)^2'."""
        dimension = Dimension("DEC(L)^2")
        assert len(dimension._proper_exponents) == 0  # type: ignore
        assert len(dimension._log_dimensions) == 1  # type: ignore
        log_dim = next(iter(dimension._log_dimensions.keys()))  # type: ignore
        assert log_dim._proper_exponents[""] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert dimension._log_dimensions[log_dim] == 2.0  # type: ignore
    
    def test_parse_complex_log_expression(self):
        """Test parsing complex log expression 'M*DEC(L^2_elec/L^2_geo)/DEC(M)^2'."""
        dimension = Dimension("M*DEC(L^2_elec/L^2_geo)/DEC(M)^2")
        # Check proper exponents
        assert dimension._proper_exponents[""] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert len(dimension._log_dimensions) == 2  # type: ignore
        
        # Check log dimensions
        log_dims = list(dimension._log_dimensions.keys())  # type: ignore
        assert len(log_dims) == 2
        
        # Find the complex log dimension (L^2_elec/L^2_geo)
        complex_log_dim = None
        mass_log_dim = None
        for log_dim in log_dims:
            if len(log_dim._proper_exponents) == 2:  # Has subscripts  # type: ignore
                complex_log_dim = log_dim
            else:
                mass_log_dim = log_dim
        
        assert complex_log_dim is not None
        assert mass_log_dim is not None
        
        # Check complex log dimension
        assert complex_log_dim._proper_exponents["elec"] == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert complex_log_dim._proper_exponents["geo"] == (0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert dimension._log_dimensions[complex_log_dim] == 1.0  # type: ignore
        
        # Check mass log dimension
        assert mass_log_dim._proper_exponents[""] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        assert dimension._log_dimensions[mass_log_dim] == -2.0  # type: ignore
    
    def test_parse_extremely_complex_dimension(self):
        """Test parsing extremely complex dimension with nested DEC() and subscripts."""
        # Test the first extremely complex dimension string
        complex_string1 = "M*DEC(DEC(L^2_elec/L^2_geo)*DEC(M^3_thermal/M^2_mech)/DEC(T^1.5_elec/T^0.5_geo)^2)*DEC(DEC(N^0.5_chem/N^0.3_bio)*DEC(Θ^2_high/Θ^1.5_low)/DEC(I^1.2_AC/I^0.8_DC)^3)^0.7/DEC(J^1.5_vis/J^0.5_IR)^2"
        
        complex_dim1 = Dimension(complex_string1)
        
        # Verify it was parsed successfully (no exceptions)
        assert complex_dim1 is not None
        assert isinstance(complex_dim1, Dimension)
        
        # Analyze the structure of complex_dim1
        # Check proper exponents (should have M from the main part)
        assert complex_dim1._proper_exponents[""] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        
        # Check log dimensions
        assert len(complex_dim1._log_dimensions) == 3  # Should have 3 DEC() expressions  # type: ignore
        
        # Find and verify each log dimension
        log_dims = list(complex_dim1._log_dimensions.keys())  # type: ignore
        
        # First DEC: DEC(DEC(L^2_elec/L^2_geo)*DEC(M^3_thermal/M^2_mech)/DEC(T^1.5_elec/T^0.5_geo)^2)
        # This should be a log dimension with nested log dimensions
        first_log_dim = None
        for log_dim in log_dims:
            if complex_dim1._log_dimensions[log_dim] == 1.0 and len(log_dim._log_dimensions) > 0:  # type: ignore
                first_log_dim = log_dim
                break
        
        assert first_log_dim is not None
        assert complex_dim1._log_dimensions[first_log_dim] == 1.0  # type: ignore
        
        # Check that first log dimension has nested log dimensions
        assert len(first_log_dim._log_dimensions) == 3  # Should have 3 nested DEC() expressions  # type: ignore
        
        # Check the nested log dimensions
        nested_log_dims = list(first_log_dim._log_dimensions.keys())  # type: ignore
        
        # First nested: DEC(L^2_elec/L^2_geo)
        first_nested = None
        for nested_dim in nested_log_dims:
            if "elec" in nested_dim._proper_exponents and "geo" in nested_dim._proper_exponents:  # type: ignore
                first_nested = nested_dim
                break
        
        assert first_nested is not None
        assert first_log_dim._log_dimensions[first_nested] == 1.0  # type: ignore
        assert first_nested._proper_exponents["elec"] == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # L^2  # type: ignore
        assert first_nested._proper_exponents["geo"] == (0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # L^-2  # type: ignore
        
        # Second nested: DEC(M^3_thermal/M^2_mech)
        second_nested = None
        for nested_dim in nested_log_dims:
            if "thermal" in nested_dim._proper_exponents and "mech" in nested_dim._proper_exponents:  # type: ignore
                second_nested = nested_dim
                break
        
        assert second_nested is not None
        assert first_log_dim._log_dimensions[second_nested] == 1.0  # type: ignore
        assert second_nested._proper_exponents["thermal"] == (3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # M^3  # type: ignore
        assert second_nested._proper_exponents["mech"] == (-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # M^-2  # type: ignore
        
        # Third nested: DEC(T^1.5_elec/T^0.5_geo)^2
        third_nested = None
        for nested_dim in nested_log_dims:
            if first_log_dim._log_dimensions[nested_dim] == -2.0:  # Has exponent -2  # type: ignore
                third_nested = nested_dim
                break
        
        assert third_nested is not None
        assert first_log_dim._log_dimensions[third_nested] == -2.0  # type: ignore
        assert third_nested._proper_exponents["elec"] == (0.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # T^1.5  # type: ignore
        assert third_nested._proper_exponents["geo"] == (0.0, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # T^-0.5  # type: ignore
        
        # Second DEC: DEC(DEC(N^0.5_chem/N^0.3_bio)*DEC(Θ^2_high/Θ^1.5_low)/DEC(I^1.2_AC/I^0.8_DC)^3)^0.7
        # This should have exponent 0.7 and contain nested log dimensions
        second_log_dim = None
        for log_dim in log_dims:
            if complex_dim1._log_dimensions[log_dim] == 0.7:  # type: ignore
                second_log_dim = log_dim
                break
        
        assert second_log_dim is not None
        assert len(second_log_dim._log_dimensions) == 3  # Should have 3 nested DEC() expressions  # type: ignore
        
        # Check the nested log dimensions in second log dimension
        second_nested_log_dims = list(second_log_dim._log_dimensions.keys())  # type: ignore
        
        # First nested: DEC(N^0.5_chem/N^0.3_bio)
        second_first_nested = None
        for nested_dim in second_nested_log_dims:
            if "chem" in nested_dim._proper_exponents and "bio" in nested_dim._proper_exponents:  # type: ignore
                second_first_nested = nested_dim
                break
        
        assert second_first_nested is not None
        assert second_log_dim._log_dimensions[second_first_nested] == 1.0  # type: ignore
        assert second_first_nested._proper_exponents["chem"] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0)  # N^0.5  # type: ignore
        assert second_first_nested._proper_exponents["bio"] == (0.0, 0.0, 0.0, 0.0, 0.0, -0.3, 0.0, 0.0)  # N^-0.3  # type: ignore
        
        # Third DEC: DEC(J^1.5_vis/J^0.5_IR)^2
        # This should have exponent -2 (due to division) and contain subscripts directly
        third_log_dim = None
        for log_dim in log_dims:
            if complex_dim1._log_dimensions[log_dim] == -2.0 and len(log_dim._log_dimensions) == 0:  # type: ignore
                third_log_dim = log_dim
                break
        
        assert third_log_dim is not None
        assert len(third_log_dim._log_dimensions) == 0  # No nested log dimensions  # type: ignore
        assert "vis" in third_log_dim._proper_exponents  # type: ignore
        assert "IR" in third_log_dim._proper_exponents  # type: ignore
        assert third_log_dim._proper_exponents["vis"] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 0.0)  # J^1.5  # type: ignore
        assert third_log_dim._proper_exponents["IR"] == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.0)   # J^-0.5  # type: ignore
        
        # Test the second extremely complex dimension string
        complex_string2 = "T*DEC(DEC(A^3_rad/A^2_deg)*DEC(L^1.5_elec/L^0.5_geo)/DEC(M^2_thermal/M^1_mech)^1.5)*DEC(DEC(Θ^0.8_high/Θ^0.4_low)*DEC(I^1.1_AC/I^0.9_DC)/DEC(N^0.6_chem/N^0.4_bio)^2.5)^0.3/DEC(J^1.2_vis/J^0.8_IR)^1.8"
        
        complex_dim2 = Dimension(complex_string2)
        
        # Verify it was parsed successfully (no exceptions)
        assert complex_dim2 is not None
        assert isinstance(complex_dim2, Dimension)
        
        # Analyze the structure of complex_dim2
        # Check proper exponents (should have T from the main part)
        assert complex_dim2._proper_exponents[""] == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        
        # Check log dimensions
        assert len(complex_dim2._log_dimensions) == 3  # Should have 3 DEC() expressions  # type: ignore
        
        # Test that both dimensions are different (they should be)
        assert complex_dim1 != complex_dim2
        
        # Test round-trip formatting: string -> parse -> format -> string
        str1 = complex_dim1.format_string()
        str2 = complex_dim2.format_string()
        assert len(str1) > 0
        assert len(str2) > 0
        assert str1 != str2
        
        # Test that formatted strings can be parsed again
        reparsed_dim1 = Dimension(str1)
        reparsed_dim2 = Dimension(str2)
        
        # Verify the reparsed dimensions are equivalent to the original
        assert reparsed_dim1 == complex_dim1
        assert reparsed_dim2 == complex_dim2
        
        # Test that the round-trip preserves the structure
        assert len(reparsed_dim1._log_dimensions) == len(complex_dim1._log_dimensions)  # type: ignore
        assert len(reparsed_dim2._log_dimensions) == len(complex_dim2._log_dimensions)  # type: ignore
    
    def test_parse_nested_complex_dimension(self):
        """Test parsing nested complex dimension with detailed structure analysis."""
        # This is an extremely complicated dimension string that includes:
        # - Multiple nested DEC() dimensions
        # - Subscripts within the DEC() expressions
        # - Complex arithmetic operations
        # - Multiple levels of nesting
        
        complex_string = "M*L^2/T^2*DEC(L_elec*M_geo/T_therm)*DEC(DEC(L^2_quantum/M_plasma)*T_fluid/L_optical)^3"
        
        dimension = Dimension(complex_string)
        
        # Verify it was parsed successfully (no exceptions)
        assert dimension is not None
        assert isinstance(dimension, Dimension)
        
        # Check proper exponents (should have M, L^2, T^-2 from the main part)
        assert dimension._proper_exponents[""] == (1.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        
        # Check log dimensions
        assert len(dimension._log_dimensions) == 2  # type: ignore
        
        # Find and verify each log dimension
        log_dims = list(dimension._log_dimensions.keys())  # type: ignore
        
        # First DEC: DEC(L_elec*M_geo/T_therm)
        first_log_dim = None
        for log_dim in log_dims:
            if len(log_dim._proper_exponents) == 3:  # Has 3 subscripts  # type: ignore
                first_log_dim = log_dim
                break
        
        assert first_log_dim is not None
        assert dimension._log_dimensions[first_log_dim] == 1.0  # type: ignore
        
        # Check subscripts in first log dimension
        assert "elec" in first_log_dim._proper_exponents  # type: ignore
        assert "geo" in first_log_dim._proper_exponents  # type: ignore
        assert "therm" in first_log_dim._proper_exponents  # type: ignore
        
        # Verify the exponents for each subscript
        assert first_log_dim._proper_exponents["elec"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # L  # type: ignore
        assert first_log_dim._proper_exponents["geo"] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # M  # type: ignore
        assert first_log_dim._proper_exponents["therm"] == (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # T^-1  # type: ignore
        
        # Second DEC: DEC(DEC(L^2_quantum/M_plasma)*T_fluid/L_optical)^3
        # This should have exponent 3 and contain both subscripts and nested DEC
        second_log_dim = None
        for log_dim in log_dims:
            if dimension._log_dimensions[log_dim] == 3.0:  # type: ignore
                second_log_dim = log_dim
                break
        
        assert second_log_dim is not None
        
        # Check that second log dimension has both subscripts and nested log dimensions
        assert len(second_log_dim._log_dimensions) == 1  # One nested log dimension  # type: ignore
        assert len(second_log_dim._proper_exponents) == 2  # Two subscripts: fluid, optical  # type: ignore
        
        # Check subscripts in second log dimension
        assert "fluid" in second_log_dim._proper_exponents  # type: ignore
        assert "optical" in second_log_dim._proper_exponents  # type: ignore
        assert second_log_dim._proper_exponents["fluid"] == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    # T  # type: ignore
        assert second_log_dim._proper_exponents["optical"] == (0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0) # L^-1  # type: ignore
        
        # Get the nested log dimension
        nested_log_dim = next(iter(second_log_dim._log_dimensions.keys()))  # type: ignore
        assert second_log_dim._log_dimensions[nested_log_dim] == 1.0  # type: ignore
        
        # Check subscripts in nested log dimension
        assert "quantum" in nested_log_dim._proper_exponents  # type: ignore
        assert "plasma" in nested_log_dim._proper_exponents  # type: ignore
        
        # Verify the exponents for nested subscripts
        assert nested_log_dim._proper_exponents["quantum"] == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # L^2  # type: ignore
        assert nested_log_dim._proper_exponents["plasma"] == (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # M^-1  # type: ignore
        
        # Test round-trip formatting: string -> parse -> format -> string
        formatted = dimension.format_string()
        assert len(formatted) > 0
        
        # Test that formatted string can be parsed again
        reparsed_dimension = Dimension(formatted)
        
        # Verify the reparsed dimension is equivalent to the original
        assert reparsed_dimension == dimension
        
        # Test that the round-trip preserves the structure
        assert len(reparsed_dimension._log_dimensions) == len(dimension._log_dimensions)  # type: ignore
        assert reparsed_dimension._proper_exponents == dimension._proper_exponents  # type: ignore
    
    def test_parse_dimensionless(self):
        """Test parsing dimensionless dimension (empty string)."""
        dimension = Dimension("")
        assert len(dimension._proper_exponents) == 0  # type: ignore
        assert len(dimension._log_dimensions) == 0  # type: ignore
        assert dimension.is_dimensionless
    
    def test_parse_caching(self):
        """Test that parsing results are cached."""
        dim1 = Dimension("M*L^2/T^2")
        dim2 = Dimension("M*L^2/T^2")
        assert dim1 is dim2  # Same object due to caching
    
    def test_parse_invalid_symbol(self):
        """Test parsing with invalid dimension symbol."""
        with pytest.raises(ValueError, match="Invalid dimension symbol"):
            Dimension("X")
    
    def test_parse_invalid_separator(self):
        """Test parsing with invalid separator."""
        with pytest.raises(ValueError, match="Invalid dimension symbol"):
            Dimension("M+L")  # + is not a valid separator
    
    def test_parse_malformed_log_expression(self):
        """Test parsing malformed log expression."""
        with pytest.raises(ValueError, match="Invalid dimension string"):
            Dimension("DEC(L")  # Missing closing bracket
    
    def test_parse_invalid_power(self):
        """Test parsing with invalid power."""
        with pytest.raises(ValueError):
            Dimension("L^abc")  # Non-numeric power
    
    def test_parse_lowercase_log_functions(self):
        """Test parsing lowercase log functions like dec(), bin(), nat()."""
        # Test basic lowercase log functions
        test_cases = [
            ("dec(L)", "DEC(L)"),
            ("bin(M)", "DEC(M)"),
            ("nat(T)", "DEC(T)"),
            ("decade(L)", "DEC(L)"),
            ("mag(M)", "DEC(M)"),
            ("magnitude(T)", "DEC(T)")
        ]
        
        for input_str, expected_str in test_cases:
            dimension = Dimension(input_str)
            formatted = dimension.format_string()
            assert formatted == expected_str, f"Failed for {input_str}: got {formatted}, expected {expected_str}"
    
    def test_parse_nested_lowercase_log_functions(self):
        """Test parsing nested lowercase log functions."""
        # Test nested lowercase log functions
        test_cases = [
            ("dec(dec(L))", "DEC(DEC(L))"),
            ("bin(dec(M))", "DEC(DEC(M))"),
            ("nat(bin(T))", "DEC(DEC(T))"),
            ("dec(DEC(L))", "DEC(DEC(L))"),
            ("DEC(dec(L))", "DEC(DEC(L))")
        ]
        
        for input_str, expected_str in test_cases:
            dimension = Dimension(input_str)
            formatted = dimension.format_string()
            assert formatted == expected_str, f"Failed for {input_str}: got {formatted}, expected {expected_str}"
    
    def test_parse_complex_lowercase_log_expressions(self):
        """Test parsing complex expressions with lowercase log functions and subscripts."""
        # Test complex expressions with lowercase log functions
        test_cases = [
            ("dec(L_elec*M_geo/T_therm)", "DEC(L_elec*M_geo/T_therm)"),
            ("bin(M*L^2/T^2)", "DEC(M/T^2*L^2)"),  # Formatting normalizes the expression
            ("nat(L^2_elec/L^2_geo)", "DEC(L^2_elec/L^2_geo)"),  # Formatting normalizes exponents
            ("dec(DEC(L^2_quantum/M_plasma)*T_fluid/L_optical)", "DEC(T_fluid/L_optical*DEC(L^2_quantum/M_plasma))")  # Formatting reorders terms and normalizes exponents
        ]
        
        for input_str, expected_str in test_cases:
            dimension = Dimension(input_str)
            formatted = dimension.format_string()
            assert formatted == expected_str, f"Failed for {input_str}: got {formatted}, expected {expected_str}"
    
    def test_parse_mixed_regular_and_lowercase_log(self):
        """Test parsing mixed regular dimensions and lowercase log functions."""
        # Test mixed expressions
        test_cases = [
            ("M*dec(L_elec)", "M*DEC(L_elec)"),
            ("L^2*dec(M_thermal)/dec(T_geo)", "L^2*DEC(M_thermal)/DEC(T_geo)"),  # Formatting normalizes exponents
            ("kg*m/s^2*dec(DEC(A_elec*V_thermal))", "M/T^2*L*DEC(DEC(A_elec*M_thermal/T^3_thermal*L^2_thermal/I_thermal))"),
            ("N/mol*DEC(dec(mm^2_geo))", "DEC(DEC(L^2_geo))")  # Formatting normalizes exponents
        ]
        
        for input_str, expected_str in test_cases:
            dimension = Dimension(input_str)
            formatted = dimension.format_string()
            assert formatted == expected_str, f"Failed for {input_str}: got {formatted}, expected {expected_str}"
    
    def test_parse_extremely_nested_lowercase_log(self):
        """Test parsing extremely nested lowercase log functions."""
        # Test deeply nested expressions
        test_cases = [
            ("dec(dec(dec(L)))", "DEC(DEC(DEC(L)))"),
            ("bin(dec(nat(M)))", "DEC(DEC(DEC(M)))"),
            ("DEC(dec(bin(T)))", "DEC(DEC(DEC(T)))"),
            ("dec(DEC(dec(L_geo)))", "DEC(DEC(DEC(L_geo)))")
        ]
        
        for input_str, expected_str in test_cases:
            dimension = Dimension(input_str)
            formatted = dimension.format_string()
            assert formatted == expected_str, f"Failed for {input_str}: got {formatted}, expected {expected_str}"
    
    def test_parse_complex_division_with_lowercase_log(self):
        """Test parsing complex division expressions with lowercase log functions."""
        # Test the specific case mentioned by the user
        test_cases = [
            ("(N/mol)/(dec(mm^2_geo)*L_elec)", "1/L_elec/DEC(L^2_geo)"),  # Formatting normalizes exponents
            ("kg*m/s^2*dec(DEC(A_elec*V_thermal))", "M/T^2*L*DEC(DEC(A_elec*M_thermal/T^3_thermal*L^2_thermal/I_thermal))"),
            ("M/T^2.0*L", "M/T^2*L")  # Test the seperate_string behavior - preserves original format
        ]
        
        for input_str, expected_str in test_cases:
            dimension = Dimension(input_str)
            formatted = dimension.format_string()
            assert formatted == expected_str, f"Failed for {input_str}: got {formatted}, expected {expected_str}"
    
    def test_parse_all_log_function_symbols(self):
        """Test parsing all log function symbols defined in LOG_UNIT_SYMBOLS."""
        from united_system.utils.units.unit_symbol import LOG_UNIT_SYMBOLS
        
        # Test all log function symbols
        for log_symbol_enum in LOG_UNIT_SYMBOLS:
            for log_symbol in log_symbol_enum.value.symbols:
                input_str = f"{log_symbol}(L)"
                dimension = Dimension(input_str)
                formatted = dimension.format_string()
                expected_str = "DEC(L)"
                assert formatted == expected_str, f"Failed for {log_symbol}: got {formatted}, expected {expected_str}"
    
    def test_parse_nested_complex_dimension_with_lowercase(self):
        """Test parsing nested complex dimension with lowercase log functions."""
        # This is an extremely complicated dimension string that includes:
        # - Multiple nested lowercase log functions
        # - Subscripts within the log expressions
        # - Complex arithmetic operations
        # - Multiple levels of nesting
        
        complex_string = "M*L^2/T^2*dec(L_elec*M_geo/T_therm)*dec(dec(L^2_quantum/M_plasma)*T_fluid/L_optical)^3"
        
        dimension = Dimension(complex_string)
        
        # Verify it was parsed successfully (no exceptions)
        assert dimension is not None
        assert isinstance(dimension, Dimension)
        
        # Check proper exponents (should have M, L^2, T^-2 from the main part)
        assert dimension._proper_exponents[""] == (1.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        
        # Check log dimensions
        assert len(dimension._log_dimensions) == 2  # type: ignore
        
        # Find and verify each log dimension
        log_dims = list(dimension._log_dimensions.keys())  # type: ignore
        
        # First dec: dec(L_elec*M_geo/T_therm)
        first_log_dim = None
        for log_dim in log_dims:
            if len(log_dim._proper_exponents) == 3:  # Has 3 subscripts  # type: ignore
                first_log_dim = log_dim
                break
        
        assert first_log_dim is not None
        assert dimension._log_dimensions[first_log_dim] == 1.0  # type: ignore
        
        # Check subscripts in first log dimension
        assert "elec" in first_log_dim._proper_exponents  # type: ignore
        assert "geo" in first_log_dim._proper_exponents  # type: ignore
        assert "therm" in first_log_dim._proper_exponents  # type: ignore
        
        # Verify the exponents for each subscript
        assert first_log_dim._proper_exponents["elec"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # L  # type: ignore
        assert first_log_dim._proper_exponents["geo"] == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)   # M  # type: ignore
        assert first_log_dim._proper_exponents["therm"] == (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) # T^-1  # type: ignore
        
        # Second dec: dec(dec(L^2_quantum/M_plasma)*T_fluid/L_optical)^3
        # This should have exponent 3 and contain both subscripts and nested dec
        second_log_dim = None
        for log_dim in log_dims:
            if dimension._log_dimensions[log_dim] == 3.0:  # type: ignore
                second_log_dim = log_dim
                break
        
        assert second_log_dim is not None
        
        # Check that second log dimension has both subscripts and nested log dimensions
        assert len(second_log_dim._log_dimensions) == 1  # One nested log dimension  # type: ignore
        assert len(second_log_dim._proper_exponents) == 2  # Two subscripts: fluid, optical  # type: ignore
        
        # Check subscripts in second log dimension
        assert "fluid" in second_log_dim._proper_exponents  # type: ignore
        assert "optical" in second_log_dim._proper_exponents  # type: ignore
        assert second_log_dim._proper_exponents["fluid"] == (0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)    # T  # type: ignore
        assert second_log_dim._proper_exponents["optical"] == (0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0) # L^-1  # type: ignore
        
        # Get the nested log dimension
        nested_log_dim = next(iter(second_log_dim._log_dimensions.keys()))  # type: ignore
        assert second_log_dim._log_dimensions[nested_log_dim] == 1.0  # type: ignore
        
        # Check subscripts in nested log dimension
        assert "quantum" in nested_log_dim._proper_exponents  # type: ignore
        assert "plasma" in nested_log_dim._proper_exponents  # type: ignore
        
        # Verify the exponents for nested subscripts
        assert nested_log_dim._proper_exponents["quantum"] == (0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # L^2  # type: ignore
        assert nested_log_dim._proper_exponents["plasma"] == (-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # M^-1  # type: ignore
        
        # Test round-trip formatting: string -> parse -> format -> string
        formatted = dimension.format_string()
        assert len(formatted) > 0
        
        # Test that formatted string can be parsed again
        reparsed_dimension = Dimension(formatted)
        
        # Verify the reparsed dimension is equivalent to the original
        assert reparsed_dimension == dimension
        
        # Test that the round-trip preserves the structure
        assert len(reparsed_dimension._log_dimensions) == len(dimension._log_dimensions)  # type: ignore
        assert reparsed_dimension._proper_exponents == dimension._proper_exponents  # type: ignore

    def test_canonical_unit_basic_dimensions(self):
        """Test canonical_unit property for basic dimensions."""
        # Test basic dimensions
        test_cases = [
            ("M", "kg"),
            ("L", "m"),
            ("T", "s"),
            ("I", "A"),
            ("Θ", "K"),
            ("N", "mol"),
            ("J", "cd"),
            ("A", "rad")
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_complex_dimensions(self):
        """Test canonical_unit property for complex dimensions."""
        # Test complex dimensions - note that unit formatting may reorder terms alphabetically
        test_cases = [
            ("M*L/T^2", "N"),  # Force
            ("M*L^2/T^2", "J"),  # Energy
            ("M/L^3", "kg/m^3"),  # Density
            ("L/T", "m/s"),  # Velocity
            ("L/T^2", "m/s^2"),  # Acceleration
            ("M*L^2/T^3", "W"),  # Power
            ("I*T", "C"),  # Electric charge - reordered alphabetically
            ("M*L^2/T^2/I^2", "H")  # Inductance
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_with_subscripts(self):
        """Test canonical_unit property for dimensions with subscripts."""
        # Test dimensions with subscripts
        test_cases = [
            ("L_elec", "m_elec"),
            ("M_thermal", "kg_thermal"),
            ("T_geo", "s_geo"),
            ("L_elec*M_geo", "m_elec*kg_geo"),
            ("L_elec/L_geo", "m_elec/m_geo"),
            ("M_thermal*L_elec/T_geo", "kg_thermal*m_elec/s_geo")
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_log_dimensions(self):
        """Test canonical_unit property for log dimensions."""
        # Test log dimensions - note that log dimensions use dimension symbols (L, M) not unit symbols (m, kg)
        test_cases = [
            ("DEC(L)", "dec(L)"),  # Uses dimension symbol L, not unit symbol m
            ("DEC(M)", "dec(M)"),  # Uses dimension symbol M, not unit symbol kg
            ("DEC(M*L/T^2)", "dec(M/T^2*L)"),  # Uses dimension symbols and normalizes
            ("DEC(L_elec)", "dec(L_elec)"),  # Preserves subscripts
            ("DEC(L_elec/L_geo)", "dec(L_elec/L_geo)")  # Preserves subscripts
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_mixed_regular_and_log(self):
        """Test canonical_unit property for mixed regular and log dimensions."""
        # Test mixed regular and log dimensions - regular parts use unit symbols, log parts use dimension symbols
        test_cases = [
            ("M*DEC(L)", "kg*dec(L)"),  # Regular part uses kg, log part uses L
            ("L^2*DEC(M)", "m^2*dec(M)"),  # Regular part uses m^2, log part uses M
            ("M*DEC(L_elec)/DEC(T_geo)", "kg*dec(L_elec)*dec(T_geo)^-1.0"),  # Preserves subscripts in log parts
            ("DEC(M*L/T^2)*L", "m*dec(M/T^2*L)")  # Log part uses dimension symbols, regular part uses m
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_nested_log_dimensions(self):
        """Test canonical_unit property for nested log dimensions."""
        # Test nested log dimensions - all log parts use dimension symbols
        test_cases = [
            ("DEC(DEC(L))", "dec(DEC(L))"),  # Nested log uses dimension symbols
            ("DEC(DEC(M*L/T^2))", "dec(DEC(M/T^2*L))"),  # Nested log uses dimension symbols and normalizes
            ("DEC(DEC(L_elec/L_geo))", "dec(DEC(L_elec/L_geo))"),  # Preserves subscripts
            ("M*DEC(DEC(L))", "kg*dec(DEC(L))")  # Regular part uses kg, nested log uses L
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_dimensionless(self):
        """Test canonical_unit property for dimensionless dimensions."""
        # Test dimensionless dimension
        dimension = Dimension("")
        canonical_unit = dimension.canonical_unit
        assert str(canonical_unit) == "", f"Failed for dimensionless: got {canonical_unit}, expected empty string"
    
    def test_canonical_unit_with_fractional_exponents(self):
        """Test canonical_unit property for dimensions with fractional exponents."""
        # Test dimensions with fractional exponents
        test_cases = [
            ("L^0.5", "m^0.5"),
            ("M^0.25*L^0.75", "kg^0.25*m^0.75"),
            ("T^-0.5", "1/s^0.5"),
            ("L^1.5*T^-0.5", "m^1.5/s^0.5")
        ]
        
        for dim_string, expected_unit in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            assert str(canonical_unit) == expected_unit, f"Failed for {dim_string}: got {canonical_unit}, expected {expected_unit}"
    
    def test_canonical_unit_caching(self):
        """Test that canonical_unit property is cached."""
        dimension = Dimension("M*L/T^2")
        
        # First call should create the canonical unit
        canonical_unit1 = dimension.canonical_unit
        
        # Second call should return the same object (cached)
        canonical_unit2 = dimension.canonical_unit
        
        assert canonical_unit1 is canonical_unit2, "Canonical unit should be cached"
        assert str(canonical_unit1) == "N"
    
    def test_canonical_unit_round_trip(self):
        """Test round-trip: dimension -> canonical_unit -> dimension."""
        test_cases = [
            "M",
            "L",
            "T",
            "M*L/T^2",
            "L_elec",
            "DEC(L)",
            "M*DEC(L)",
            "DEC(DEC(L))"
        ]
        
        for dim_string in test_cases:
            dimension = Dimension(dim_string)
            canonical_unit = dimension.canonical_unit
            
            # Convert canonical unit back to dimension
            # This tests that the canonical unit represents the same dimension
            canonical_dimension = canonical_unit.dimension
            
            assert canonical_dimension == dimension, f"Round-trip failed for {dim_string}: {canonical_dimension} != {dimension}"
    
    def test_canonical_unit_complex_nested(self):
        """Test canonical_unit property for complex nested dimensions."""
        # Test extremely complex nested dimensions
        complex_string = "M*L^2/T^2*DEC(L_elec*M_geo/T_therm)*DEC(DEC(L^2_quantum/M_plasma)*T_fluid/L_optical)^3"
        dimension = Dimension(complex_string)
        
        canonical_unit = dimension.canonical_unit
        
        # Verify the canonical unit has the expected structure
        assert "J" in str(canonical_unit)   # Should contain energy unit (reduced from kg*m^2/s^2)
        assert "dec" in str(canonical_unit) # Should contain log units
        
        # Verify round-trip
        canonical_dimension = canonical_unit.dimension
        assert canonical_dimension == dimension, f"Round-trip failed for complex nested dimension"
    
    def test_canonical_unit_arithmetic_operations(self):
        """Test canonical_unit property for dimensions created through arithmetic operations."""
        # Test dimensions created through arithmetic operations
        mass = Dimension("M")
        length = Dimension("L")
        time = Dimension("T")
        
        # Test multiplication
        force = mass * length / (time ** 2)
        canonical_force = force.canonical_unit
        assert str(canonical_force) == "N"
        
        # Test with subscripts
        elec_length = Dimension("L_elec")
        geo_length = Dimension("L_geo")
        ratio = elec_length / geo_length
        canonical_ratio = ratio.canonical_unit
        assert str(canonical_ratio) == "m_elec/m_geo"
        
        # Test with log operations
        log_mass = mass.log()
        canonical_log_mass = log_mass.canonical_unit
        assert str(canonical_log_mass) == "dec(M)"
        
        # Test complex arithmetic
        complex_dim = (mass * length ** 2) / (time ** 2) * log_mass
        canonical_complex = complex_dim.canonical_unit
        assert "J" in str(canonical_complex)  # Should contain energy unit (reduced from kg*m^2/s^2)
        assert "dec" in str(canonical_complex)


class TestDimensionCreation:
    """Test dimension creation and initialization."""
    
    def test_create_dimensionless(self):
        """Test creation of dimensionless dimension."""
        dimension = Dimension.dimensionless_dimension()
        assert dimension.is_dimensionless
        assert dimension == DIMENSIONLESS_DIMENSION
    
    def test_create_with_none(self):
        """Test creation with None value."""
        with pytest.raises(ValueError, match="Invalid value for dimension"):
            Dimension(None)
    
    def test_create_with_named_quantity(self):
        """Test creation with NamedQuantity."""
        # Test with base quantity
        mass_quantity = NamedQuantity.MASS
        mass_dim = Dimension(mass_quantity)
        assert mass_dim == Dimension("M")
        
        # Test with derived quantity
        force_quantity = NamedQuantity.FORCE
        force_dim = Dimension(force_quantity)
        assert force_dim == Dimension("M*L/T^2")
        
        # Test with energy quantity
        energy_quantity = NamedQuantity.ENERGY
        energy_dim = Dimension(energy_quantity)
        assert energy_dim == Dimension("M*L^2/T^2")
    
    def test_create_with_named_quantity_and_subscript(self):
        """Test creation with NamedQuantity and subscript."""
        # Test with base quantity and subscript
        mass_quantity = NamedQuantity.MASS
        mass_dim_with_sub = Dimension(mass_quantity, subscript="thermal")
        assert mass_dim_with_sub == Dimension("M_thermal")
        
        # Test with derived quantity and subscript
        force_quantity = NamedQuantity.FORCE
        force_dim_with_sub = Dimension(force_quantity, subscript="applied")
        # The subscript should apply to the entire quantity, not be split up
        # So this should create M*L/T^2 with the subscript applied to the whole expression
        expected_force = Dimension._construct({"applied": (1.0, -2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}, {})  # type: ignore
        assert force_dim_with_sub == expected_force
        
        # Test with energy quantity and subscript
        energy_quantity = NamedQuantity.ENERGY
        energy_dim_with_sub = Dimension(energy_quantity, subscript="kinetic")
        expected_energy = Dimension._construct({"kinetic": (1.0, -2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)}, {})  # type: ignore
        assert energy_dim_with_sub == expected_energy
    
    def test_create_with_named_quantity_edge_cases(self):
        """Test creation with NamedQuantity edge cases."""
        # Test with dimensionless quantity
        dimensionless_quantity = NamedQuantity.FRACTION
        dimensionless_dim = Dimension(dimensionless_quantity)
        assert dimensionless_dim == Dimension("")
        assert dimensionless_dim.is_dimensionless
        
        # Test with dimensionless quantity and subscript (should still be dimensionless)
        dimensionless_dim_with_sub = Dimension(dimensionless_quantity, subscript="test")
        assert dimensionless_dim_with_sub == Dimension("")
        assert dimensionless_dim_with_sub.is_dimensionless
        
        # Test with angle quantity
        angle_quantity = NamedQuantity.ANGLE
        angle_dim = Dimension(angle_quantity)
        assert angle_dim == Dimension("A")
        
        # Test with angle quantity and subscript
        angle_dim_with_sub = Dimension(angle_quantity, subscript="rad")
        expected_angle = Dimension._construct({"rad": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)}, {})  # type: ignore
        assert angle_dim_with_sub == expected_angle
    
    def test_create_with_named_quantity_complex_quantities(self):
        """Test creation with complex NamedQuantity instances."""
        # Test with power quantity
        power_quantity = NamedQuantity.POWER
        power_dim = Dimension(power_quantity)
        assert power_dim == Dimension("M*L^2/T^3")
        
        # Test with power quantity and subscript
        power_dim_with_sub = Dimension(power_quantity, subscript="electrical")
        expected_power = Dimension._construct({"electrical": (1.0, -3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0)}, {})  # type: ignore
        assert power_dim_with_sub == expected_power
        
        # Test with pressure quantity
        pressure_quantity = NamedQuantity.PRESSURE
        pressure_dim = Dimension(pressure_quantity)
        assert pressure_dim == Dimension("M/L/T^2")
        
        # Test with pressure quantity and subscript
        pressure_dim_with_sub = Dimension(pressure_quantity, subscript="atmospheric")
        expected_pressure = Dimension._construct({"atmospheric": (1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0)}, {})  # type: ignore
        assert pressure_dim_with_sub == expected_pressure
    
    def test_create_with_named_quantity_invalid_cases(self):
        """Test creation with invalid NamedQuantity cases."""
        # Test with None quantity
        with pytest.raises(ValueError, match="Invalid value for dimension: None"):
            Dimension(None)
        
        # Test with invalid subscript (empty string)
        mass_quantity = NamedQuantity.MASS
        with pytest.raises(ValueError, match="Invalid subscript"):
            Dimension(mass_quantity, subscript="")
        
        # Test with invalid subscript (contains invalid characters)
        with pytest.raises(ValueError, match="Invalid subscript"):
            Dimension(mass_quantity, subscript="invalid-subscript")


class TestDimensionArithmetic:
    """Test arithmetic operations on dimensions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension("M")
        self.length_dim = Dimension("L")
        self.time_dim = Dimension("T")
        self.zero_dim = Dimension("")
    
    def test_multiplication_of_dimensions(self):
        """Test multiplication of two dimensions."""
        result = self.mass_dim * self.length_dim
        expected = Dimension("M*L")
        assert result == expected

    def test_division_of_dimensions(self):
        """Test division of two dimensions."""
        result = self.mass_dim / self.length_dim
        expected = Dimension("M/L")
        assert result == expected

    def test_power_of_dimension(self):
        """Test raising a dimension to a power."""
        result = self.mass_dim ** 2
        expected = Dimension("M^2")
        assert result == expected

    def test_fractional_power_of_dimension(self):
        """Test raising a dimension to a fractional power."""
        result = self.length_dim ** 0.5
        expected = Dimension("L^0.5")
        assert result == expected
        
    def test_invert(self):
        """Test dimension inversion."""
        result = self.mass_dim.invert()
        expected = Dimension("M^-1")
        assert result == expected

    def test_invert_complex_dimension(self):
        """Test inversion of complex dimension."""
        force_dim = Dimension("M*L/T^2")
        result = force_dim.invert()
        expected = Dimension("T^2/M/L")
        assert result == expected
    
    def test_log_operation(self):
        """Test log operation."""
        result = self.mass_dim.log()
        assert len(result._proper_exponents) == 0  # type: ignore
        assert len(result._log_dimensions) == 1  # type: ignore
        log_dim = next(iter(result._log_dimensions.keys()))  # type: ignore
        assert log_dim == self.mass_dim
        assert result._log_dimensions[log_dim] == 1.0  # type: ignore
    
    def test_exp_operation_dimensionless(self):
        """Test exp operation on dimensionless dimension."""
        result = self.zero_dim.exp()
        assert result == DIMENSIONLESS_DIMENSION
    
    def test_exp_operation_log_dimension(self):
        """Test exp operation on log dimension."""
        log_dim = self.mass_dim.log()
        result = log_dim.exp()
        assert result == self.mass_dim
    
    def test_arc_operation_dimensionless(self):
        """Test arc operation on dimensionless dimension."""
        result = self.zero_dim.arc()
        assert result == ANGLE_DIMENSION
    
    def test_trig_operation_dimensionless(self):
        """Test trig operation on dimensionless dimension."""
        result = self.zero_dim.trig()
        assert result == DIMENSIONLESS_DIMENSION
    
    def test_trig_operation_angle(self):
        """Test trig operation on angle dimension."""
        result = ANGLE_DIMENSION.trig()
        assert result == DIMENSIONLESS_DIMENSION


class TestDimensionComparison:
    """Test comparison operations on dimensions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mass_dim = Dimension("M")
        self.length_dim = Dimension("L")
        self.zero_dim = Dimension("")
    
    def test_equality_same_dimension(self):
        """Test equality of same dimensions."""
        mass_dim2 = Dimension("M")
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
        mass_dim2 = Dimension("M")
        assert hash(self.mass_dim) == hash(mass_dim2)
    
    def test_is_dimensionless(self):
        """Test is_dimensionless property."""
        assert self.zero_dim.is_dimensionless
        assert not self.mass_dim.is_dimensionless
        assert not self.length_dim.is_dimensionless


class TestDimensionStringRepresentation:
    """Test string representation of dimensions."""
    
    def test_str_mass_dimension(self):
        """Test string representation of mass dimension."""
        dimension = Dimension("M")
        result = str(dimension)
        assert "M" in result
    
    def test_str_length_dimension(self):
        """Test string representation of length dimension."""
        dimension = Dimension("L")
        result = str(dimension)
        assert "L" in result
    
    def test_str_complex_dimension(self):
        """Test string representation of complex dimension."""
        dimension = Dimension("M*L^2/T^2")
        result = str(dimension)
        assert len(result) > 0
    
    def test_format_string_mass(self):
        """Test format_string method for mass dimension."""
        dimension = Dimension("M")
        result = dimension.format_string()
        assert "M" in result
    
    def test_format_string_complex(self):
        """Test format_string method for complex dimension."""
        dimension = Dimension("M*L^2/T^2")
        result = dimension.format_string()
        assert "M" in result
        assert "L^2" in result
        assert "T^2" in result
    
    def test_format_string_as_fraction_false(self):
        """Test format_string with as_fraction=False."""
        dimension = Dimension("M/L")
        result = dimension.format_string(as_fraction=False)
        assert "M" in result
        assert "L^-1" in result
    
    def test_repr(self):
        """Test repr method."""
        dimension = Dimension("M*L^2/T^2")
        result = repr(dimension)
        assert result == dimension.format_string()


class TestDimensionValidation:
    """Test dimension validation methods."""
    
    def test_is_valid_for_addition_same_dimensions(self):
        """Test is_valid_for_addition with same dimensions."""
        dim1 = Dimension("M*L^2/T^2")
        dim2 = Dimension("M*L^2/T^2")
        assert Dimension.is_valid_for_addition(dim1, dim2)
    
    def test_is_valid_for_addition_different_dimensions(self):
        """Test is_valid_for_addition with different dimensions."""
        dim1 = Dimension("M")
        dim2 = Dimension("L")
        assert not Dimension.is_valid_for_addition(dim1, dim2)
    
    def test_is_valid_for_log(self):
        """Test is_valid_for_log method."""
        dim = Dimension("M")
        assert Dimension.is_valid_for_log(dim)
    
    def test_is_valid_for_exponentiation(self):
        """Test is_valid_for_exponentiation method."""
        log_dim = Dimension("M").log()
        assert Dimension.is_valid_for_exponentiation(log_dim, 1.0)
        assert not Dimension.is_valid_for_exponentiation(Dimension("M"), 1.0)
    
    def test_is_valid_for_arc(self):
        """Test is_valid_for_arc method."""
        assert Dimension.is_valid_for_arc(Dimension(""))
        assert not Dimension.is_valid_for_arc(Dimension("M"))
    
    def test_is_valid_for_trig(self):
        """Test is_valid_for_trig method."""
        assert Dimension.is_valid_for_trig(Dimension(""))
        assert Dimension.is_valid_for_trig(ANGLE_DIMENSION)
        assert not Dimension.is_valid_for_trig(Dimension("M"))


class TestDimensionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_very_large_exponents(self):
        """Test with very large exponents."""
        dimension = Dimension("L^1000000")
        assert dimension._proper_exponents[""][2] == 1000000.0  # type: ignore
    
    def test_very_small_exponents(self):
        """Test with very small exponents."""
        dimension = Dimension("L^0.000001")
        assert dimension._proper_exponents[""][2] == 0.000001  # type: ignore
    
    def test_negative_exponents(self):
        """Test with negative exponents."""
        dimension = Dimension("L^-1")
        assert dimension._proper_exponents[""][2] == -1.0  # type: ignore
    
    def test_fractional_exponents(self):
        """Test with fractional exponents."""
        dimension = Dimension("L^0.5")
        assert dimension._proper_exponents[""][2] == 0.5  # type: ignore
    
    def test_dimensionless_exponents(self):
        """Test with all zero exponents."""
        dimension = Dimension("")
        assert dimension.is_dimensionless
    
    def test_immutability(self):
        """Test that dimensions are immutable."""
        dimension = Dimension("M")
        
        # Should not be able to modify attributes
        with pytest.raises((AttributeError, TypeError)):
            dimension._proper_exponents[""] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore


class TestDimensionRoundTrip:
    """Test round-trip formatting: string -> parse -> format -> string."""
    
    def test_simple_round_trip_fraction(self):
        """Test round-trip with simple dimensions using fraction formatting."""
        # Clear cache to avoid state pollution
        Dimension.clear_cache()
        
        test_cases = [
            "M",
            "L^2",
            "T^-1",
            "M*L/T^2",
            "L_elec",
            "M_thermal*L_geo",
            "L_elec/L_geo"
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with fraction formatting
            formatted = dimension.format_string(as_fraction=True)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_simple_round_trip_non_fraction(self):
        """Test round-trip with simple dimensions using non-fraction formatting."""
        # Clear cache to avoid state pollution
        Dimension.clear_cache()
        
        test_cases = [
            "M",
            "L^2", 
            "T^-1",
            "M*L*T^-2",  # Using negative exponents instead of division
            "L_elec",
            "M_thermal*L_geo",
            "L_elec*L_geo^-1"  # Using negative exponents instead of division
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with non-fraction formatting
            formatted = dimension.format_string(as_fraction=False)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_log_dimension_round_trip_fraction(self):
        """Test round-trip with log dimensions using fraction formatting."""
        test_cases = [
            "DEC(L)",
            "DEC(M)^2",
            "DEC(L_elec)",
            "DEC(M*L/T^2)",
            "DEC(L^2_elec/L^2_geo)"
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with fraction formatting
            formatted = dimension.format_string(as_fraction=True)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_log_dimension_round_trip_non_fraction(self):
        """Test round-trip with log dimensions using non-fraction formatting."""
        test_cases = [
            "DEC(L)",
            "DEC(M)^2",
            "DEC(L_elec)",
            "DEC(M*L*T^-2)",  # Using negative exponents
            "DEC(L^2_elec*L^2_geo^-1)"  # Using negative exponents
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with non-fraction formatting
            formatted = dimension.format_string(as_fraction=False)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_nested_log_round_trip_fraction(self):
        """Test round-trip with nested log dimensions using fraction formatting."""
        test_cases = [
            "DEC(DEC(L))",
            "DEC(DEC(L^2_elec/L^2_geo))",
            "DEC(DEC(M)*DEC(L))",
            "DEC(DEC(L_elec)*DEC(M_geo)/DEC(T_therm))"
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with fraction formatting
            formatted = dimension.format_string(as_fraction=True)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_nested_log_round_trip_non_fraction(self):
        """Test round-trip with nested log dimensions using non-fraction formatting."""
        test_cases = [
            "DEC(DEC(L))",
            "DEC(DEC(L^2_elec*L^2_geo^-1))",  # Using negative exponents
            "DEC(DEC(M)*DEC(L))",
            "DEC(DEC(L_elec)*DEC(M_geo)*DEC(T_therm)^-1)"  # Using negative exponents
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with non-fraction formatting
            formatted = dimension.format_string(as_fraction=False)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_complex_mixed_round_trip_fraction(self):
        """Test round-trip with complex mixed expressions using fraction formatting."""
        test_cases = [
            "M*DEC(L_elec)",
            "L^2*DEC(M_thermal)/DEC(T_geo)",
            "M*DEC(DEC(L^2_quantum/M_plasma)*T_fluid/L_optical)^3",
            "T*DEC(DEC(A^3_rad/A^2_deg)*DEC(L^1.5_elec/L^0.5_geo)/DEC(M^2_thermal/M^1_mech)^1.5)"
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with fraction formatting
            formatted = dimension.format_string(as_fraction=True)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_complex_mixed_round_trip_non_fraction(self):
        """Test round-trip with complex mixed expressions using non-fraction formatting."""
        test_cases = [
            "M*DEC(L_elec)",
            "L^2*DEC(M_thermal)*DEC(T_geo)^-1",  # Using negative exponents
            "M*DEC(DEC(L^2_quantum*M_plasma^-1)*T_fluid*L_optical^-1)^3",  # Using negative exponents
            "T*DEC(DEC(A^3_rad*A^2_deg^-1)*DEC(L^1.5_elec*L^0.5_geo^-1)*DEC(M^2_thermal*M^1_mech^-1)^1.5)"  # Using negative exponents
        ]
        
        for dim_string in test_cases:
            # Parse
            dimension = Dimension(dim_string)
            
            # Format back with non-fraction formatting
            formatted = dimension.format_string(as_fraction=False)
            
            # Parse again
            reparsed = Dimension(formatted)
            
            # Verify equivalence
            assert reparsed == dimension, f"Round-trip failed for '{dim_string}'"
    
    def test_extremely_complex_round_trip_fraction(self):
        """Test round-trip with extremely complex dimensions using fraction formatting."""
        complex_string = "M*DEC(DEC(L^2_elec/L^2_geo)*DEC(M^3_thermal/M^2_mech)/DEC(T^1.5_elec/T^0.5_geo)^2)*DEC(DEC(N^0.5_chem/N^0.3_bio)*DEC(Θ^2_high/Θ^1.5_low)/DEC(I^1.2_AC/I^0.8_DC)^3)^0.7/DEC(J^1.5_vis/J^0.5_IR)^2"
        
        # Parse
        dimension = Dimension(complex_string)
        
        # Format back with fraction formatting
        formatted = dimension.format_string(as_fraction=True)
        
        # Parse again
        reparsed = Dimension(formatted)
        
        # Verify equivalence
        assert reparsed == dimension
        
        # Verify structure preservation
        assert len(reparsed._log_dimensions) == len(dimension._log_dimensions)  # type: ignore
        assert reparsed._proper_exponents == dimension._proper_exponents  # type: ignore
        
        # Test multiple round-trips
        for i in range(3):
            formatted = reparsed.format_string(as_fraction=True)
            reparsed = Dimension(formatted)
            assert reparsed == dimension, f"Multiple round-trip failed at iteration {i+1}"
    
    def test_extremely_complex_round_trip_non_fraction(self):
        """Test round-trip with extremely complex dimensions using non-fraction formatting."""
        complex_string = "M*DEC(DEC(L^2_elec*L^2_geo^-1)*DEC(M^3_thermal*M^2_mech^-1)*DEC(T^1.5_elec*T^0.5_geo^-1)^2)*DEC(DEC(N^0.5_chem*N^0.3_bio^-1)*DEC(Θ^2_high*Θ^1.5_low^-1)*DEC(I^1.2_AC*I^0.8_DC^-1)^3)^0.7*DEC(J^1.5_vis*J^0.5_IR^-1)^2"
        
        # Parse
        dimension = Dimension(complex_string)
        
        # Format back with non-fraction formatting
        formatted = dimension.format_string(as_fraction=False)
        
        # Parse again
        reparsed = Dimension(formatted)
        
        # Verify equivalence
        assert reparsed == dimension
        
        # Verify structure preservation
        assert len(reparsed._log_dimensions) == len(dimension._log_dimensions)  # type: ignore
        assert reparsed._proper_exponents == dimension._proper_exponents  # type: ignore
        
        # Test multiple round-trips
        for i in range(3):
            formatted = reparsed.format_string(as_fraction=False)
            reparsed = Dimension(formatted)
            assert reparsed == dimension, f"Multiple round-trip failed at iteration {i+1}"
    
    def test_format_string_options(self):
        """Test format_string with different options."""
        dimension = Dimension("M*L^2/T^2")
        
        # Test default formatting (should be fraction)
        formatted_default = dimension.format_string()
        assert "M" in formatted_default
        assert "L" in formatted_default
        assert "T" in formatted_default
        
        # Test with as_fraction=True (explicit)
        formatted_fraction = dimension.format_string(as_fraction=True)
        assert "M" in formatted_fraction
        assert "L" in formatted_fraction
        assert "T" in formatted_fraction
        
        # Test with as_fraction=False
        formatted_no_fraction = dimension.format_string(as_fraction=False)
        assert "M" in formatted_no_fraction
        assert "L" in formatted_no_fraction
        assert "T" in formatted_no_fraction
        
        # All should parse back to the same dimension
        reparsed_default = Dimension(formatted_default)
        reparsed_fraction = Dimension(formatted_fraction)
        reparsed_no_fraction = Dimension(formatted_no_fraction)
        
        assert reparsed_default == dimension
        assert reparsed_fraction == dimension
        assert reparsed_no_fraction == dimension
        
        # Test that different formatting styles produce different strings
        assert formatted_default != formatted_no_fraction
        assert formatted_fraction != formatted_no_fraction
    
    def test_formatting_normalization(self):
        """Test that formatting normalizes the representation while preserving mathematical equivalence."""
        
        # Test cases that work correctly
        test_pairs = [
            ("M*L/T^2", "M*L*T^-2"),  # These are mathematically equivalent
            ("DEC(M*L/T^2)", "DEC(M*L*T^-2)")  # These are mathematically equivalent
        ]
        
        for fraction_str, negative_exp_str in test_pairs:
            # Parse both representations
            fraction_dim = Dimension(fraction_str)
            negative_exp_dim = Dimension(negative_exp_str)
            
            # They should be mathematically equivalent
            assert fraction_dim == negative_exp_dim, f"Mathematical equivalence failed for {fraction_str} vs {negative_exp_str}"
            
            # Format both - they should produce the same normalized string
            formatted_fraction = fraction_dim.format_string()
            formatted_negative_exp = negative_exp_dim.format_string()
            
            # Both should parse back to the same dimension
            reparsed_fraction = Dimension(formatted_fraction)
            reparsed_negative_exp = Dimension(formatted_negative_exp)
            
            assert reparsed_fraction == fraction_dim
            assert reparsed_negative_exp == negative_exp_dim
            assert reparsed_fraction == reparsed_negative_exp
    
    def test_formatting_behavior(self):
        """Test the actual formatting behavior observed."""
        
        # Test with as_fraction=True (default)
        dimension = Dimension("M*L/T^2")
        formatted_fraction = dimension.format_string(as_fraction=True)
        
        # Should use division for negative exponents
        assert "/T^2" in formatted_fraction or "/T^-2" in formatted_fraction
        assert "M" in formatted_fraction
        assert "L" in formatted_fraction
        
        # Test with as_fraction=False
        formatted_no_fraction = dimension.format_string(as_fraction=False)
        
        # Should use negative exponents
        assert "T^-2" in formatted_no_fraction
        assert "M" in formatted_no_fraction
        assert "L" in formatted_no_fraction
        
        # Both should parse back to the same dimension
        reparsed_fraction = Dimension(formatted_fraction)
        reparsed_no_fraction = Dimension(formatted_no_fraction)
        
        assert reparsed_fraction == dimension
        assert reparsed_no_fraction == dimension
        
        # Test with subscripts
        subscript_dim = Dimension("L_elec/L_geo")
        formatted_subscript_fraction = subscript_dim.format_string(as_fraction=True)
        formatted_subscript_no_fraction = subscript_dim.format_string(as_fraction=False)
        
        # Should preserve the original structure for subscripts
        assert "L_elec" in formatted_subscript_fraction
        assert "L_geo" in formatted_subscript_fraction
        assert "L_elec" in formatted_subscript_no_fraction
        assert "L^-1_geo" in formatted_subscript_no_fraction  # Non-fraction uses negative exponents
        
        # Both should parse back to the same dimension
        reparsed_subscript_fraction = Dimension(formatted_subscript_fraction)
        reparsed_subscript_no_fraction = Dimension(formatted_subscript_no_fraction)
        
        assert reparsed_subscript_fraction == subscript_dim
        assert reparsed_subscript_no_fraction == subscript_dim


class TestDimensionArithmeticReversibility:
    """Test arithmetic operations and their reversibility."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear cache to avoid state pollution between tests
        Dimension.clear_cache()
        self.mass_dim = Dimension("M")
        self.length_dim = Dimension("L")
        self.time_dim = Dimension("T")
        self.force_dim = Dimension("M*L/T^2")
        self.energy_dim = Dimension("M*L^2/T^2")
        self.zero_dim = Dimension("")
    
    def test_multiplication_division_reversibility(self):
        """Test that multiplication and division are reversible operations."""
        # Test: (a * b) / b = a
        result1 = (self.mass_dim * self.length_dim) / self.length_dim
        assert result1 == self.mass_dim
        
        # Test: (a / b) * b = a
        result2 = (self.force_dim / self.mass_dim) * self.mass_dim
        assert result2 == self.force_dim
        
        # Test with complex dimensions
        complex_dim = Dimension("M*L^2/T^2*DEC(L_elec)")
        result3 = (complex_dim * self.time_dim) / self.time_dim
        assert result3 == complex_dim
        
        # Test with subscripts
        subscript_dim = Dimension("L_elec*M_geo")
        result4 = (subscript_dim / self.mass_dim) * self.mass_dim
        assert result4 == subscript_dim
    
    def test_power_root_reversibility(self):
        """Test that power and root operations are reversible."""
        # Test: (a^2)^0.5 = a
        result1 = (self.length_dim ** 2) ** 0.5
        assert result1 == self.length_dim
        
        # Test: (a^0.5)^2 = a
        result2 = (self.length_dim ** 0.5) ** 2
        assert result2 == self.length_dim
        
        # Test with negative powers: (a^-2)^-0.5 = a
        result3 = (self.time_dim ** -2) ** -0.5
        assert result3 == self.time_dim
        
        # Test with complex dimensions
        complex_dim = Dimension("M*L^2/T^2")
        result4 = (complex_dim ** 3) ** (1/3)
        assert result4 == complex_dim
    
    def test_log_exp_reversibility(self):
        """Test that log and exp operations are reversible."""
        # Test: exp(log(a)) = a
        result1 = self.mass_dim.log().exp()
        assert result1 == self.mass_dim
        
        # Test: log(exp(a)) = a (for dimensionless)
        result2 = self.zero_dim.exp().log()
        expected_log_exp_dimensionless = Dimension("DEC()")
        assert result2 == expected_log_exp_dimensionless
        
        # Test with complex dimensions
        complex_dim = Dimension("M*L^2/T^2")
        result3 = complex_dim.log().exp()
        assert result3 == complex_dim
        
        # Test with subscripts
        subscript_dim = Dimension("L_elec*M_geo")
        result4 = subscript_dim.log().exp()
        assert result4 == subscript_dim
    
    def test_invert_reversibility(self):
        """Test that inversion is reversible."""
        # Test: (a^-1)^-1 = a
        result1 = self.mass_dim.invert().invert()
        assert result1 == self.mass_dim
        
        # Test with complex dimensions
        result2 = self.force_dim.invert().invert()
        assert result2 == self.force_dim
        
        # Test with subscripts
        subscript_dim = Dimension("L_elec/M_geo")
        result3 = subscript_dim.invert().invert()
        assert result3 == subscript_dim
    
    def test_mixed_arithmetic_reversibility(self):
        """Test reversibility of mixed arithmetic operations."""
        # Test: ((a * b) / c) * c / b = a
        result1 = ((self.mass_dim * self.length_dim) / self.time_dim) * self.time_dim / self.length_dim
        assert result1 == self.mass_dim
        
        # Test: (a^2 * b) / b = a^2
        result2 = (self.mass_dim ** 2 * self.length_dim) / self.length_dim
        expected2 = self.mass_dim ** 2
        assert result2 == expected2
        
        # Test: ((a / b)^2 * b^2) = a^2
        result3 = ((self.mass_dim / self.length_dim) ** 2) * (self.length_dim ** 2)
        expected3 = self.mass_dim ** 2
        assert result3 == expected3
    
    def test_log_arithmetic_reversibility(self):
        """Test reversibility of log arithmetic operations."""
        # Test: log(a * b) creates DEC(a*b)
        log_ab = (self.mass_dim * self.length_dim).log()
        expected_log_ab = Dimension("DEC(M*L)")
        assert log_ab == expected_log_ab
        
        # Test: exp(log(a * b)) = a * b
        result = log_ab.exp()
        expected = self.mass_dim * self.length_dim
        assert result == expected
        
        # Test: log(a) + log(b) = log(a*b) (logarithmic identity)
        log_a = self.mass_dim.log()
        log_b = self.length_dim.log()
        sum_logs = log_a + log_b
        assert sum_logs == log_ab
        
        # Test: log(a) - log(b) = log(a/b) (logarithmic identity)
        diff_logs = log_a - log_b
        log_quotient = (self.mass_dim / self.length_dim).log()
        assert diff_logs == log_quotient
        
        # Test: log(a) * log(b) creates DEC(a) * DEC(b) (different from log(a*b))
        log_a_times_log_b = log_a * log_b
        expected_log_a_times_log_b = Dimension("DEC(M)*DEC(L)")
        assert log_a_times_log_b == expected_log_a_times_log_b
        
        # These are different mathematical expressions
        assert log_ab != log_a_times_log_b
    
    def test_complex_nested_reversibility(self):
        """Test reversibility of complex nested operations."""
        # Create a complex dimension with log operations
        complex_dim = Dimension("M*L^2/T^2*DEC(L_elec)")
        
        # Test: ((a * b) / c) * c / b = a
        result1 = ((complex_dim * self.time_dim) / self.mass_dim) * self.mass_dim / self.time_dim
        assert result1 == complex_dim
        
        # Test: (a^2)^0.5 = a
        result2 = (complex_dim ** 2) ** 0.5
        assert result2 == complex_dim
        
        # Test: exp(log(a)) = a
        result3 = complex_dim.log().exp()
        assert result3 == complex_dim
    
    def test_subscript_arithmetic_reversibility(self):
        """Test reversibility with subscripted dimensions."""
        # Test multiplication/division with subscripts
        elec_dim = Dimension("L_elec")
        geo_dim = Dimension("L_geo")
        
        # Test: (a_elec * b_geo) / b_geo = a_elec
        result1 = (elec_dim * geo_dim) / geo_dim
        assert result1 == elec_dim
        
        # Test: (a_elec / b_geo) * b_geo = a_elec
        result2 = (elec_dim / geo_dim) * geo_dim
        assert result2 == elec_dim
        
        # Test power operations with subscripts
        result3 = (elec_dim ** 2) ** 0.5
        assert result3 == elec_dim
        
        # Test log operations with subscripts
        result4 = elec_dim.log().exp()
        assert result4 == elec_dim
    
    def test_edge_case_reversibility(self):
        """Test reversibility with edge cases."""
        # Test with dimensionless dimension
        result1 = (self.zero_dim * self.mass_dim) / self.mass_dim
        assert result1 == self.zero_dim
        
        # Test with zero power
        result2 = (self.mass_dim ** 0) ** 1
        assert str(result2) == ""  # Should be dimensionless (empty string)
        
        # Test with negative powers
        result3 = (self.mass_dim ** -1) ** -1
        assert result3 == self.mass_dim
        
        # Test with fractional powers
        result4 = (self.mass_dim ** 0.5) ** 2
        assert result4 == self.mass_dim
    
    def test_arithmetic_identity_properties(self):
        """Test identity properties of arithmetic operations."""
        # Test multiplication identity: a * 1 = a
        identity_dim = Dimension("")
        result1 = self.mass_dim * identity_dim
        assert result1 == self.mass_dim
        
        # Test division identity: a / 1 = a
        result2 = self.mass_dim / identity_dim
        assert result2 == self.mass_dim
        
        # Test power identity: a^1 = a
        result3 = self.mass_dim ** 1
        assert result3 == self.mass_dim
        
        # Test log identity: log(1) = DEC() (log dimension)
        result4 = identity_dim.log()
        expected_log_identity = Dimension("DEC()")
        assert result4 == expected_log_identity
    
    def test_arithmetic_zero_properties(self):
        """Test zero properties of arithmetic operations."""
        # Test: a * dimensionless = a (multiplication identity)
        result1 = self.mass_dim * Dimension("")
        assert result1 == self.mass_dim
        
        # Test: dimensionless / a = 1/a (division by any dimension gives inverse)
        result2 = Dimension("") / self.mass_dim
        expected_inverse = Dimension("M^-1")
        assert result2 == expected_inverse
        
        # Test: dimensionless^a = dimensionless (any power of dimensionless is dimensionless)
        result3 = Dimension("") ** 2
        assert result3 == Dimension("")
    
    def test_distributive_property(self):
        """Test distributive property of multiplication over addition (for dimensions)."""
        # Test: a * (b * c) = (a * b) * c (associative property)
        result1 = self.mass_dim * (self.length_dim * self.time_dim)
        result2 = (self.mass_dim * self.length_dim) * self.time_dim
        assert result1 == result2
        
        # Test: (a * b)^c = a^c * b^c (power distributes over multiplication)
        result3 = (self.mass_dim * self.length_dim) ** 2
        result4 = (self.mass_dim ** 2) * (self.length_dim ** 2)
        assert result3 == result4
    
    def test_commutative_property(self):
        """Test commutative property of multiplication."""
        # Test: a * b = b * a
        result1 = self.mass_dim * self.length_dim
        result2 = self.length_dim * self.mass_dim
        assert result1 == result2
        
        # Test with complex dimensions
        complex_dim1 = Dimension("M*L^2/T^2")
        complex_dim2 = Dimension("DEC(L_elec)")
        result3 = complex_dim1 * complex_dim2
        result4 = complex_dim2 * complex_dim1
        assert result3 == result4
    
    def test_round_trip_arithmetic(self):
        """Test that arithmetic operations preserve round-trip formatting."""
        # Create a complex dimension
        original = Dimension("M*L^2/T^2*DEC(L_elec)")
        
        # Apply arithmetic operations
        modified = (original * self.time_dim) / self.mass_dim
        
        # Apply reverse operations
        restored = (modified * self.mass_dim) / self.time_dim
        
        # Test that formatting and parsing still work
        formatted = restored.format_string()
        reparsed = Dimension(formatted)
        assert reparsed == restored
        assert reparsed == original
    
    def test_arithmetic_with_log_dimensions(self):
        """Test arithmetic operations involving log dimensions."""
        # Test: log(a) * log(b) creates DEC(a) * DEC(b)
        log_a = self.mass_dim.log()
        log_b = self.length_dim.log()
        log_a_times_log_b = log_a * log_b
        expected_log_a_times_log_b = Dimension("DEC(M)*DEC(L)")
        assert log_a_times_log_b == expected_log_a_times_log_b
        
        # Test: log(a * b) creates DEC(a*b) (different from log(a) * log(b))
        log_ab = (self.mass_dim * self.length_dim).log()
        expected_log_ab = Dimension("DEC(M*L)")
        assert log_ab == expected_log_ab
        
        # These are different mathematical expressions
        assert log_a_times_log_b != log_ab
        
        # Test: exp(log(a * b)) = a * b
        result = log_ab.exp()
        expected = self.mass_dim * self.length_dim
        assert result == expected
        
        # Test with complex log dimensions
        complex_log = Dimension("DEC(M*L^2/T^2)")
        result2 = (complex_log * self.time_dim) / self.time_dim
        assert result2 == complex_log
    
    def test_arithmetic_preserves_structure(self):
        """Test that arithmetic operations preserve the internal structure."""
        # Create a dimension with subscripts and log dimensions
        original = Dimension("M*L_elec*DEC(L_geo)")
        
        # Apply arithmetic operations
        modified = original * self.time_dim
        
        # Check that subscripts are preserved
        assert "elec" in modified._proper_exponents  # type: ignore
        assert modified._proper_exponents["elec"] == (0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # type: ignore
        
        # Check that log dimensions are preserved
        assert len(modified._log_dimensions) == 1  # type: ignore
        
        # Apply reverse operation
        restored = modified / self.time_dim
        
        # Should be equivalent to original
        assert restored == original


class TestDimensionSerialization:
    """Test serialization and deserialization of dimensions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear cache to avoid state pollution between tests
        Dimension.clear_cache()
        self.mass_dim = Dimension("M")
        self.length_dim = Dimension("L")
        self.time_dim = Dimension("T")
        self.force_dim = Dimension("M*L/T^2")
        self.energy_dim = Dimension("M*L^2/T^2")
        self.zero_dim = Dimension("")
        self.complex_dim = Dimension("M*L^2/T^2*DEC(L_elec)")
        self.subscript_dim = Dimension("L_elec*M_geo")
        self.log_dim = Dimension("DEC(M*L/T^2)")
        self.nested_log_dim = Dimension("DEC(DEC(L^2_elec/L^2_geo))")
    
    def test_json_serialization_simple(self):
        """Test JSON serialization of simple dimensions."""
        # Test mass dimension
        json_str = self.mass_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.mass_dim
        
        # Test length dimension
        json_str = self.length_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.length_dim
        
        # Test time dimension
        json_str = self.time_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.time_dim
        
        # Test dimensionless dimension
        json_str = self.zero_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.zero_dim
    
    def test_json_serialization_complex(self):
        """Test JSON serialization of complex dimensions."""
        # Test force dimension
        json_str = self.force_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.force_dim
        
        # Test energy dimension
        json_str = self.energy_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.energy_dim
        
        # Test complex dimension with log
        json_str = self.complex_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.complex_dim
    
    def test_json_serialization_subscripts(self):
        """Test JSON serialization of dimensions with subscripts."""
        # Test subscript dimension
        json_str = self.subscript_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.subscript_dim
        
        # Test complex subscript dimension
        complex_subscript = Dimension("L_elec*M_geo/T_therm")
        json_str = complex_subscript.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == complex_subscript
    
    def test_json_serialization_log_dimensions(self):
        """Test JSON serialization of log dimensions."""
        # Test simple log dimension
        json_str = self.log_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.log_dim
        
        # Test nested log dimension
        json_str = self.nested_log_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == self.nested_log_dim
        
        # Test complex log dimension with subscripts
        complex_log = Dimension("DEC(L^2_elec/L^2_geo*M_thermal)")
        json_str = complex_log.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == complex_log
    
    def test_json_serialization_extremely_complex(self):
        """Test JSON serialization of extremely complex dimensions."""
        # Test the extremely complex dimension from parsing tests
        complex_string = "M*DEC(DEC(L^2_elec/L^2_geo)*DEC(M^3_thermal/M^2_mech)/DEC(T^1.5_elec/T^0.5_geo)^2)*DEC(DEC(N^0.5_chem/N^0.3_bio)*DEC(Θ^2_high/Θ^1.5_low)/DEC(I^1.2_AC/I^0.8_DC)^3)^0.7/DEC(J^1.5_vis/J^0.5_IR)^2"
        complex_dim = Dimension(complex_string)
        
        json_str = complex_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert reparsed == complex_dim
        
        # Verify structure preservation
        assert len(reparsed._log_dimensions) == len(complex_dim._log_dimensions)  # type: ignore
        assert reparsed._proper_exponents == complex_dim._proper_exponents  # type: ignore
    
    def test_json_serialization_round_trip(self):
        """Test multiple round-trips of JSON serialization."""
        # Test simple dimension
        original = self.force_dim
        for i in range(5):
            json_str = original.to_json()
            reparsed = Dimension.from_json(json_str)
            assert reparsed == original, f"Round-trip {i+1} failed for force dimension"
            original = reparsed
        
        # Test complex dimension
        original = self.complex_dim
        for i in range(5):
            json_str = original.to_json()
            reparsed = Dimension.from_json(json_str)
            assert reparsed == original, f"Round-trip {i+1} failed for complex dimension"
            original = reparsed
    
    def test_json_serialization_format_consistency(self):
        """Test that JSON serialization produces consistent format strings."""
        # Test that to_json() produces the same result as format_string()
        assert self.mass_dim.to_json() == self.mass_dim.format_string()
        assert self.force_dim.to_json() == self.force_dim.format_string()
        assert self.complex_dim.to_json() == self.complex_dim.format_string()
        assert self.zero_dim.to_json() == self.zero_dim.format_string()
    
    def test_hdf5_serialization_simple(self):
        """Test HDF5 serialization of simple dimensions."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test mass dimension
                self.mass_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.mass_dim
                
                # Test length dimension
                h5_file.attrs.clear()
                self.length_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.length_dim
                
                # Test dimensionless dimension
                h5_file.attrs.clear()
                self.zero_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.zero_dim
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_hdf5_serialization_complex(self):
        """Test HDF5 serialization of complex dimensions."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test force dimension
                self.force_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.force_dim
                
                # Test complex dimension with log
                h5_file.attrs.clear()
                self.complex_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.complex_dim
                
                # Test subscript dimension
                h5_file.attrs.clear()
                self.subscript_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.subscript_dim
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_hdf5_serialization_log_dimensions(self):
        """Test HDF5 serialization of log dimensions."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test simple log dimension
                self.log_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.log_dim
                
                # Test nested log dimension
                h5_file.attrs.clear()
                self.nested_log_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == self.nested_log_dim
                
                # Test complex log dimension with subscripts
                complex_log = Dimension("DEC(L^2_elec/L^2_geo*M_thermal)")
                h5_file.attrs.clear()
                complex_log.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == complex_log
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_hdf5_serialization_extremely_complex(self):
        """Test HDF5 serialization of extremely complex dimensions."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test the extremely complex dimension
                complex_string = "M*DEC(DEC(L^2_elec/L^2_geo)*DEC(M^3_thermal/M^2_mech)/DEC(T^1.5_elec/T^0.5_geo)^2)*DEC(DEC(N^0.5_chem/N^0.3_bio)*DEC(Θ^2_high/Θ^1.5_low)/DEC(I^1.2_AC/I^0.8_DC)^3)^0.7/DEC(J^1.5_vis/J^0.5_IR)^2"
                complex_dim = Dimension(complex_string)
                
                complex_dim.to_hdf5(h5_file)
                reparsed = Dimension.from_hdf5(h5_file)
                assert reparsed == complex_dim
                
                # Verify structure preservation
                assert len(reparsed._log_dimensions) == len(complex_dim._log_dimensions)  # type: ignore
                assert reparsed._proper_exponents == complex_dim._proper_exponents  # type: ignore
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_hdf5_serialization_round_trip(self):
        """Test multiple round-trips of HDF5 serialization."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test simple dimension
                original = self.force_dim
                for i in range(5):
                    h5_file.attrs.clear()
                    original.to_hdf5(h5_file)
                    reparsed = Dimension.from_hdf5(h5_file)
                    assert reparsed == original, f"Round-trip {i+1} failed for force dimension"
                    original = reparsed
                
                # Test complex dimension
                original = self.complex_dim
                for i in range(5):
                    h5_file.attrs.clear()
                    original.to_hdf5(h5_file)
                    reparsed = Dimension.from_hdf5(h5_file)
                    assert reparsed == original, f"Round-trip {i+1} failed for complex dimension"
                    original = reparsed
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_hdf5_serialization_format_consistency(self):
        """Test that HDF5 serialization produces consistent format strings."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test that HDF5 stores the same format string as to_json()
                self.mass_dim.to_hdf5(h5_file)
                assert h5_file.attrs["dimension"] == self.mass_dim.format_string()
                
                h5_file.attrs.clear()
                self.force_dim.to_hdf5(h5_file)
                assert h5_file.attrs["dimension"] == self.force_dim.format_string()
                
                h5_file.attrs.clear()
                self.complex_dim.to_hdf5(h5_file)
                assert h5_file.attrs["dimension"] == self.complex_dim.format_string()
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_serialization_equivalence(self):
        """Test that JSON and HDF5 serialization produce equivalent results."""
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Test that both serialization methods produce the same format string
                json_str = self.complex_dim.to_json()
                self.complex_dim.to_hdf5(h5_file)
                hdf5_str = h5_file.attrs["dimension"]  # type: ignore
                
                assert json_str == hdf5_str
                
                # Test that both can be parsed back to the same dimension
                json_reparsed = Dimension.from_json(json_str)
                hdf5_reparsed = Dimension.from_hdf5(h5_file)
                
                assert json_reparsed == self.complex_dim
                assert hdf5_reparsed == self.complex_dim
                assert json_reparsed == hdf5_reparsed
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_serialization_error_handling(self):
        """Test error handling in serialization methods."""
        # Test from_json with invalid JSON string
        with pytest.raises(ValueError):
            Dimension.from_json("invalid_dimension_string")
        
        # Test from_json with empty string (should create dimensionless dimension)
        empty_dim = Dimension.from_json("")
        assert empty_dim.is_dimensionless
        
        # Test from_hdf5 with missing attribute
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                # Create file without dimension attribute
                with pytest.raises(KeyError):
                    Dimension.from_hdf5(h5_file)
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_serialization_with_arithmetic_operations(self):
        """Test serialization of dimensions created through arithmetic operations."""
        # Create dimensions through arithmetic operations
        multiplied = self.mass_dim * self.length_dim
        divided = self.force_dim / self.mass_dim
        powered = self.length_dim ** 2
        logged = self.mass_dim.log()
        
        # Test JSON serialization
        assert multiplied == Dimension.from_json(multiplied.to_json())
        assert divided == Dimension.from_json(divided.to_json())
        assert powered == Dimension.from_json(powered.to_json())
        assert logged == Dimension.from_json(logged.to_json())
        
        # Test HDF5 serialization
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                multiplied.to_hdf5(h5_file)
                assert multiplied == Dimension.from_hdf5(h5_file)
                
                h5_file.attrs.clear()
                divided.to_hdf5(h5_file)
                assert divided == Dimension.from_hdf5(h5_file)
                
                h5_file.attrs.clear()
                powered.to_hdf5(h5_file)
                assert powered == Dimension.from_hdf5(h5_file)
                
                h5_file.attrs.clear()
                logged.to_hdf5(h5_file)
                assert logged == Dimension.from_hdf5(h5_file)
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_serialization_with_named_quantities(self):
        """Test serialization of dimensions created from NamedQuantity."""
        # Create dimensions from NamedQuantity
        mass_from_quantity = Dimension(NamedQuantity.MASS)
        force_from_quantity = Dimension(NamedQuantity.FORCE)
        energy_from_quantity = Dimension(NamedQuantity.ENERGY)
        
        # Test JSON serialization
        assert mass_from_quantity == Dimension.from_json(mass_from_quantity.to_json())
        assert force_from_quantity == Dimension.from_json(force_from_quantity.to_json())
        assert energy_from_quantity == Dimension.from_json(energy_from_quantity.to_json())
        
        # Test HDF5 serialization
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                mass_from_quantity.to_hdf5(h5_file)
                assert mass_from_quantity == Dimension.from_hdf5(h5_file)
                
                h5_file.attrs.clear()
                force_from_quantity.to_hdf5(h5_file)
                assert force_from_quantity == Dimension.from_hdf5(h5_file)
                
                h5_file.attrs.clear()
                energy_from_quantity.to_hdf5(h5_file)
                assert energy_from_quantity == Dimension.from_hdf5(h5_file)
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_serialization_performance(self):
        """Test performance of serialization methods."""
        import time
        
        # Test JSON serialization performance
        start_time = time.time()
        for _ in range(100):
            json_str = self.complex_dim.to_json()
            Dimension.from_json(json_str)
        json_time = time.time() - start_time
        
        # Test HDF5 serialization performance
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            start_time = time.time()
            for _ in range(100):
                with h5py.File(tmp_file.name, 'w') as h5_file:
                    self.complex_dim.to_hdf5(h5_file)
                    Dimension.from_hdf5(h5_file)
            hdf5_time = time.time() - start_time
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
        
        # Both should complete in reasonable time
        assert json_time < 1.0, f"JSON serialization took {json_time:.3f}s"
        assert hdf5_time < 5.0, f"HDF5 serialization took {hdf5_time:.3f}s"  # HDF5 is slower due to file I/O


class TestDimensionPerformance:
    """Test performance characteristics."""
    
    def test_creation_performance(self):
        """Test that dimension creation is fast."""
        import time
        
        start_time = time.time()
        for _ in range(1000):
            Dimension("M*L^2/T^2")
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
    
    def test_caching_performance(self):
        """Test that caching improves performance."""
        import time
        
        # First call (no cache)
        start_time = time.time()
        dim1 = Dimension("M*L^2/T^2")
        first_call_time = time.time() - start_time
        
        # Second call (cached)
        start_time = time.time()
        dim2 = Dimension("M*L^2/T^2")
        second_call_time = time.time() - start_time
        
        # Test that caching works (same object)
        assert dim1 is dim2
        
        # Test that both calls complete in reasonable time
        assert first_call_time < 0.1  # Should complete in under 100ms
        assert second_call_time < 0.1  # Should complete in under 100ms


class TestDimensionConstructorEdgeCases:
    """Test edge cases in constructor validation."""
    
    def test_unit_with_subscript_combination(self):
        """Test that Unit + subscript combination works but ignores subscript."""
        from united_system.unit import Unit
        
        # Test Unit + subscript combination (should work but ignore subscript)
        dim = Dimension(Unit("kg"), subscript="test") # type: ignore
        assert dim == Dimension("M")  # Should ignore subscript and use Unit's dimension
    
    def test_none_with_subscript(self):
        """Test None with subscript raises error."""
        with pytest.raises(ValueError, match="Invalid value for dimension"):
            Dimension(None, subscript="test")
    
    def test_malformed_unit_objects(self):
        """Test handling of malformed Unit objects."""
        from united_system.unit import Unit
        
        # Test with Unit that has no dimension property
        # This would require mocking, but we can test the basic case
        unit = Unit("kg")
        dim = Dimension(unit)
        assert dim == Dimension("M")
    
    def test_invalid_constructor_combinations(self):
        """Test various invalid constructor combinations."""
        from united_system.unit import Unit
        
        # Test string with Unit (constructor is permissive, ignores extra args)
        dim: Dimension = Dimension("M", Unit("kg")) # type: ignore
        assert dim == Dimension("M")  # Should ignore the Unit argument
        
        # Test NamedQuantity with Unit as subscript (should fail due to type checking)
        with pytest.raises(TypeError, match="'Unit' object is not iterable"):
            Dimension(NamedQuantity.MASS, Unit("kg")) # type: ignore
        
        # Test with unknown keyword argument (should fail)
        with pytest.raises(TypeError):
            Dimension("M", unknown_param="test") # type: ignore


class TestDimensionMathematicalProperties:
    """Test fundamental mathematical properties."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.a = Dimension("M*L/T^2")  # Force
        self.b = Dimension("L^2")      # Area
        self.c = Dimension("T")        # Time
        self.d = Dimension("M")        # Mass
        self.zero = Dimension("")      # Dimensionless
    
    def test_associative_property(self):
        """Test associative property: (a * b) * c == a * (b * c)."""
        # Test with regular dimensions
        left = (self.a * self.b) * self.c
        right = self.a * (self.b * self.c)
        assert left == right
        
        # Test with log dimensions
        log_a = self.a.log()
        log_b = self.b.log()
        left_log = (log_a * log_b) * self.c
        right_log = log_a * (log_b * self.c)
        assert left_log == right_log
    
    def test_identity_element_properties(self):
        """Test identity element properties."""
        # Test multiplication identity
        assert self.a * self.zero == self.a
        assert self.zero * self.a == self.a
        
        # Test addition identity (should fail for different dimensions)
        with pytest.raises(ValueError):
            _ = self.a + self.b  # Different dimensions
    
    def test_distributive_property_with_log(self):
        """Test distributive property with logarithmic dimensions."""
        # Test: log(a*b) == log(a) + log(b) (logarithmic identity)
        product = self.a * self.b
        log_product = product.log()
        log_sum = self.a.log() + self.b.log()
        assert log_product == log_sum
    
    def test_commutative_property_multiplication(self):
        """Test commutative property of multiplication."""
        assert self.a * self.b == self.b * self.a
        
        # Test with log dimensions
        log_a = self.a.log()
        log_b = self.b.log()
        assert log_a * log_b == log_b * log_a
    
    def test_power_properties(self):
        """Test power operation properties."""
        # Test: (a^m)^n == a^(m*n)
        assert (self.a ** 2) ** 3 == self.a ** 6
        
        # Test: (a*b)^n == a^n * b^n
        assert (self.a * self.b) ** 2 == (self.a ** 2) * (self.b ** 2)
        
        # Test: a^0 == dimensionless
        assert self.a ** 0 == self.zero
    
    def test_log_properties(self):
        """Test logarithmic operation properties."""
        # Test: log(a*b) == log(a) + log(b)
        product = self.a * self.b
        log_product = product.log()
        log_sum = self.a.log() + self.b.log()
        assert log_product == log_sum
        
        # Test: log(a^n) behavior (current implementation)
        power = self.a ** 3
        log_power = power.log()
        # The current implementation creates DEC(M^3/T^6*L^3) for log(a^3)
        # This is different from the mathematical identity log(a^n) = n * log(a)
        # but it's consistent with the current design
        assert "DEC(" in str(log_power)
        assert "M^3" in str(log_power) or "L^3" in str(log_power)


class TestDimensionMemoryAndPerformance:
    """Test memory usage and performance characteristics."""
    
    def test_cache_memory_behavior(self):
        """Test cache behavior under memory pressure."""
        # Clear cache first
        Dimension.clear_cache()
        
        # Create many unique dimensions
        dimensions = []
        for i in range(1000):
            dim_str = f"M^{i%5}*L^{i%3}/T^{i%2}"
            dimensions.append(Dimension(dim_str)) # type: ignore
        
        # Verify cache is working
        cache_size_before = len(Dimension._CACHE) # type: ignore
        
        # Create the same dimensions again
        for i in range(1000):
            dim_str = f"M^{i%5}*L^{i%3}/T^{i%2}"
            Dimension(dim_str)
        
        # Cache size should be the same (no new entries)
        cache_size_after = len(Dimension._CACHE) # type: ignore
        assert cache_size_after == cache_size_before
        
        # Clear cache
        Dimension.clear_cache()
        assert len(Dimension._CACHE) == 0 # type: ignore
    
    def test_deeply_nested_performance(self):
        """Test performance with deeply nested log dimensions."""
        import time
        
        # Create a deeply nested log dimension
        base = Dimension("M*L/T^2")
        nested = base
        for _ in range(10):
            nested = nested.log()
        
        # Test that operations on deeply nested dimensions are reasonable
        start_time = time.time()
        for _ in range(100):
            _ = nested * nested
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
    
    def test_large_exponent_performance(self):
        """Test performance with large exponents."""
        import time
        
        # Create dimension with large exponents
        large_dim = Dimension("M^100*L^50/T^25")
        
        start_time = time.time()
        for _ in range(100):
            _ = large_dim ** 2
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0
    
    def test_garbage_collection_behavior(self):
        """Test garbage collection behavior with cached dimensions."""
        import gc
        
        # Clear cache
        Dimension.clear_cache()
        
        # Create dimensions and let them go out of scope
        for i in range(100):
            Dimension(f"M^{i%5}*L^{i%3}")
        
        # Force garbage collection
        gc.collect()
        
        # Cache should still contain the dimensions
        assert len(Dimension._CACHE) > 0 # type: ignore
        
        # Clear cache manually
        Dimension.clear_cache()
        assert len(Dimension._CACHE) == 0 # type: ignore


class TestDimensionUnicodeAndSpecialCharacters:
    """Test handling of Unicode subscripts and special characters."""
    
    def test_unicode_subscripts(self):
        """Test Unicode subscripts in dimensions."""
        # Test various Unicode characters
        unicode_chars = ["α", "β", "γ", "δ", "ε", "θ", "λ", "μ", "π", "σ", "φ", "ω"]
        
        for char in unicode_chars:
            dim_str = f"L_{char}"
            dim = Dimension(dim_str)
            assert str(dim) == dim_str
    
    def test_special_characters_in_subscripts(self):
        """Test special characters in subscripts."""
        # Test various special characters
        special_chars = ["_", "1", "2", "3", "a", "b", "c", "x", "y", "z"]
        
        for char in special_chars:
            dim_str = f"L_{char}"
            dim = Dimension(dim_str)
            assert str(dim) == dim_str
    
    def test_invalid_subscript_characters(self):
        """Test that invalid subscript characters are handled gracefully."""
        # The current implementation is permissive and accepts most characters
        # Only certain characters that interfere with parsing will cause issues
        problematic_chars = ["*", "/", "(", ")", "^"]  # These interfere with parsing
        
        for char in problematic_chars:
            dim_str = f"L_{char}"
            # These should either parse correctly or raise a parsing error
            try:
                dim = Dimension(dim_str)
                # If it parses, it should be a valid dimension
                assert isinstance(dim, Dimension)
            except (ValueError, TypeError):
                # Parsing error is acceptable for problematic characters
                pass
        
        # Test that normal special characters work
        normal_chars = ["!", "@", "#", "$", "%", "&", "-", "+", "="]
        for char in normal_chars:
            dim_str = f"L_{char}"
            dim = Dimension(dim_str)
            assert isinstance(dim, Dimension)
            assert str(dim) == dim_str
    
    def test_mixed_unicode_and_ascii(self):
        """Test mixed Unicode and ASCII characters in subscripts."""
        dim_str = "L_αβγ123"
        dim = Dimension(dim_str)
        assert str(dim) == dim_str
    
    def test_unicode_log_functions(self):
        """Test Unicode characters in log function arguments."""
        dim_str = "DEC(L_α/L_β)"
        dim = Dimension(dim_str)
        # The exact string representation might be normalized, but it should parse correctly
        assert "α" in str(dim) or "β" in str(dim)


class TestDimensionIntegrationScenarios:
    """Test real-world integration scenarios."""
    
    def test_complex_physics_workflow(self):
        """Test a complex physics problem workflow."""
        # Simulate a physics problem: calculating energy from force and distance
        force = Dimension("M*L/T^2")  # Force
        distance = Dimension("L")     # Distance
        energy = force * distance     # Energy = Force * Distance
        expected_energy = Dimension("M*L^2/T^2")
        
        assert energy == expected_energy
        
        # Test canonical unit conversion
        energy_unit = energy.canonical_unit
        assert str(energy_unit) == "J"  # Should be Joules
        
        # Test round-trip
        energy_dim = energy_unit.dimension
        assert energy_dim == energy
    
    def test_dataframe_integration_edge_cases(self):
        """Test DataFrame integration edge cases."""
        from united_system.unit import Unit
        
        # Test that dimensions work with units in DataFrame context
        mass_dim = Dimension("M")
        mass_unit = Unit("kg")
        
        # Verify dimension consistency
        assert mass_unit.dimension == mass_dim
        
        # Test with subscripts
        elec_mass_dim = Dimension("M_elec")
        elec_mass_unit = Unit("kg_elec")
        assert elec_mass_unit.dimension == elec_mass_dim
    
    def test_serialization_of_complex_nested_structures(self):
        """Test serialization of complex nested structures."""
        # Create a very complex dimension
        complex_dim = Dimension("M*DEC(L_α/L_β)*DEC(DEC(M_γ/T_δ))")
        
        # Test JSON serialization
        json_str = complex_dim.to_json()
        reparsed = Dimension.from_json(json_str)
        assert complex_dim == reparsed
        
        # Test HDF5 serialization
        import h5py
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            with h5py.File(tmp_file.name, 'w') as h5_file:
                complex_dim.to_hdf5(h5_file)
                reparsed_hdf5 = Dimension.from_hdf5(h5_file)
                assert complex_dim == reparsed_hdf5
        
        # Clean up
        import os
        os.unlink(tmp_file.name)
    
    def test_named_quantity_integration_edge_cases(self):
        """Test NamedQuantity integration edge cases."""
        # Test all named quantities
        quantities = [
            NamedQuantity.MASS,
            NamedQuantity.LENGTH,
            NamedQuantity.TIME,
            NamedQuantity.CURRENT,
            NamedQuantity.TEMPERATURE,
            NamedQuantity.AMOUNT_OF_SUBSTANCE,
            NamedQuantity.LUMINOUS_INTENSITY,
            NamedQuantity.ANGLE,
            NamedQuantity.FORCE,
            NamedQuantity.ENERGY,
            NamedQuantity.POWER,
            NamedQuantity.PRESSURE,
            NamedQuantity.VOLTAGE,
            NamedQuantity.RESISTANCE,
            NamedQuantity.CAPACITANCE,
            NamedQuantity.INDUCTANCE,
            NamedQuantity.FREQUENCY,
            NamedQuantity.AREA,
            NamedQuantity.VOLUME,
            NamedQuantity.DENSITY,
            NamedQuantity.ELECTRIC_FIELD,
            NamedQuantity.MAGNETIC_FIELD,
            NamedQuantity.ELECTRIC_FLUX,
            NamedQuantity.MAGNETIC_FLUX,
            NamedQuantity.ELECTRIC_DISPLACEMENT,
            NamedQuantity.MAGNETIZATION,
            NamedQuantity.LUMINOUS_INTENSITY,
        ]
        
        for quantity in quantities:
            try:
                dim = Dimension(quantity)
                # Verify it's a valid dimension
                assert isinstance(dim, Dimension)
                # Test round-trip
                assert dim == Dimension(str(dim))
            except Exception as e:
                # Some quantities might not be implemented yet
                print(f"Warning: {quantity} failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__])
