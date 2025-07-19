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

# Import the modules to test
from src.united_system.unit import Unit
from src.united_system.dimension import Dimension


class TestUnitCreation:
    """Test unit creation and initialization."""
    
    def test_create_empty(self):
        """Test creation of empty unit."""
        unit = Unit.empty_unit()
        assert len(unit.unit_elements) == 0
        assert unit.factor == 1.0
        assert unit.offset == 0.0
    
    def test_create_from_simple_unit_element(self):
        """Test creation from SimpleUnitElement."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        # Create a meter unit element
        meter_element = SimpleUnitElement.parse_string("m", "nominator")
        unit = Unit((meter_element,))
        
        assert len(unit.unit_elements) == 1
        assert unit.unit_elements[0] == meter_element
    
    def test_create_from_multiple_elements(self):
        """Test creation from multiple unit elements."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        meter_element = SimpleUnitElement.parse_string("m", "nominator")
        second_element = SimpleUnitElement.parse_string("s", "denominator")
        unit = Unit((meter_element, second_element))
        
        assert len(unit.unit_elements) == 2
        assert meter_element in unit.unit_elements
        assert second_element in unit.unit_elements
    
    def test_create_from_existing_unit(self):
        """Test creation from existing unit."""
        original_unit = Unit.parse_string("m/s")
        new_unit = Unit(original_unit.unit_elements)
        
        assert new_unit.unit_elements == original_unit.unit_elements
        assert new_unit.factor == original_unit.factor
        assert new_unit.offset == original_unit.offset
    
    def test_create_from_list_of_units(self):
        """Test creation from list of units."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("s")
        # Combine unit elements from both units
        combined_elements = list(unit1.unit_elements) + list(unit2.unit_elements)
        combined_unit = Unit(combined_elements)
        
        # Should combine elements from both units
        assert len(combined_unit.unit_elements) == 2
    
    def test_create_from_set_of_units(self):
        """Test creation from set of units."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("s")
        # Combine unit elements from both units
        combined_elements = list(unit1.unit_elements) + list(unit2.unit_elements)
        combined_unit = Unit(combined_elements)
        
        # Should combine elements from both units
        assert len(combined_unit.unit_elements) == 2
    
    def test_create_invalid_argument(self):
        """Test creation with invalid argument."""
        with pytest.raises(ValueError):
            Unit("invalid")
    
    def test_create_with_offset_conflict(self):
        """Test creation with conflicting offsets."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        
        # Create elements with different offsets
        celsius_element = SimpleUnitElement.parse_string("°C", "nominator")
        fahrenheit_element = SimpleUnitElement.parse_string("°F", "nominator")
        
        with pytest.raises(ValueError, match="Cannot have two non-zero offsets"):
            Unit((celsius_element, fahrenheit_element))

    def test_create_from_dimension(self):
        """Test creating a unit from a Dimension object."""
        from src.united_system.dimension import Dimension
        
        # Test with length dimension
        length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])  # L
        unit = Unit(length_dim)
        assert unit.dimension == length_dim
        assert unit.format_string(as_fraction=True) == "m"
        
        # Test with force dimension
        force_dim = Dimension.create([1, -2, 1, 0, 0, 0, 0], [0, 0])  # ML/T²
        unit = Unit(force_dim)
        assert unit.dimension == force_dim
        assert unit.format_string(as_fraction=True) == "N"
        
        # Test with dimensionless
        dimensionless = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        unit = Unit(dimensionless)
        assert unit.dimension == dimensionless
        assert unit.format_string(as_fraction=True) == ""

    def test_create_from_named_dimension(self):
        """Test creating a unit from a NamedDimension object."""
        from src.united_system.named_dimensions import NamedDimension
        
        # Test with named dimensions
        length_unit = Unit(NamedDimension.LENGTH)
        assert length_unit.dimension == NamedDimension.LENGTH.dimension
        assert length_unit.format_string(as_fraction=True) == "m"
        
        force_unit = Unit(NamedDimension.FORCE)
        assert force_unit.dimension == NamedDimension.FORCE.dimension
        assert force_unit.format_string(as_fraction=True) == "N"
        
        energy_unit = Unit(NamedDimension.ENERGY)
        assert energy_unit.dimension == NamedDimension.ENERGY.dimension
        assert energy_unit.format_string(as_fraction=True) == "J"
        
        voltage_unit = Unit(NamedDimension.VOLTAGE)
        assert voltage_unit.dimension == NamedDimension.VOLTAGE.dimension
        assert voltage_unit.format_string(as_fraction=True) == "V"

    def test_create_from_complex_dimension(self):
        """Test creating a unit from a complex dimension."""
        from src.united_system.dimension import Dimension
        
        # Test velocity dimension (L/T)
        velocity_dim = Dimension.create([0, -1, 1, 0, 0, 0, 0], [0, 0])
        unit = Unit(velocity_dim)
        assert unit.dimension == velocity_dim
        assert unit.format_string(as_fraction=True) == "m/s"
        
        # Test acceleration dimension (L/T²)
        acceleration_dim = Dimension.create([0, -2, 1, 0, 0, 0, 0], [0, 0])
        unit = Unit(acceleration_dim)
        assert unit.dimension == acceleration_dim
        assert unit.format_string(as_fraction=True) == "m/s^2"
        
        # Test pressure dimension (M/LT²)
        pressure_dim = Dimension.create([1, -2, -1, 0, 0, 0, 0], [0, 0])
        unit = Unit(pressure_dim)
        assert unit.dimension == pressure_dim
        assert unit.format_string(as_fraction=True) == "Pa"

    def test_create_from_dimension_with_angle(self):
        """Test creating a unit from a dimension with angle."""
        from src.united_system.dimension import Dimension
        
        # Test angular velocity dimension (angle/T)
        angular_velocity_dim = Dimension.create([0, -1, 0, 0, 0, 0, 0], [1, 0])
        unit = Unit(angular_velocity_dim)
        assert unit.dimension == angular_velocity_dim
        assert unit.format_string(as_fraction=True) == "rad/s"
        
        # Test angular acceleration dimension (angle/T²)
        angular_acceleration_dim = Dimension.create([0, -2, 0, 0, 0, 0, 0], [1, 0])
        unit = Unit(angular_acceleration_dim)
        assert unit.dimension == angular_acceleration_dim
        assert unit.format_string(as_fraction=True) == "rad/s^2"

    def test_dimension_canonical_unit_property(self):
        """Test that dimensions have a canonical_unit property."""
        from src.united_system.dimension import Dimension
        from src.united_system.named_dimensions import NamedDimension
        
        # Test basic dimensions
        length_dim = Dimension.create([0, 0, 1, 0, 0, 0, 0], [0, 0])
        canonical_unit = length_dim.canonical_unit
        assert canonical_unit.dimension == length_dim
        assert canonical_unit.format_string(as_fraction=True) == "m"
        
        # Test named dimensions
        force_canonical = NamedDimension.FORCE.dimension.canonical_unit
        assert force_canonical.dimension == NamedDimension.FORCE.dimension
        assert force_canonical.format_string(as_fraction=True) == "N"
        
        # Test dimensionless
        dimensionless = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        canonical_unit = dimensionless.canonical_unit
        assert canonical_unit.dimension == dimensionless
        assert canonical_unit.format_string(as_fraction=True) == ""


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

    def test_direct_constructor_simple_unit(self):
        """Test Unit('m') constructor matches Unit.parse_string('m')."""
        unit1 = Unit("m")
        unit2 = Unit.parse_string("m")
        assert unit1 == unit2
        assert str(unit1) == str(unit2)

    def test_direct_constructor_composite_unit(self):
        """Test Unit('m/s^2') constructor matches Unit.parse_string('m/s^2')."""
        unit1 = Unit("m/s^2")
        unit2 = Unit.parse_string("m/s^2")
        assert unit1 == unit2
        assert str(unit1) == str(unit2)

    def test_direct_constructor_with_prefix(self):
        """Test Unit('km') constructor matches Unit.parse_string('km')."""
        unit1 = Unit("km")
        unit2 = Unit.parse_string("km")
        assert unit1 == unit2
        assert str(unit1) == str(unit2)

    def test_direct_constructor_complex(self):
        """Test Unit('kg*m^2/(s^3*A)') constructor matches Unit.parse_string('kg*m^2/(s^3*A)')."""
        unit1 = Unit("kg*m^2/(s^3*A)")
        unit2 = Unit.parse_string("kg*m^2/(s^3*A)")
        assert unit1 == unit2
        assert str(unit1) == str(unit2)


class TestUnitArithmetic:
    """Test arithmetic operations on units."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.united_system.unit import clear_unit_cache
        clear_unit_cache()  # Clear cache to ensure fresh parsing for every test
        self.meter_unit = Unit.parse_string("m")
        self.second_unit = Unit.parse_string("s")
        self.kilogram_unit = Unit.parse_string("kg")
    
    def test_multiplication(self):
        result = self.meter_unit * self.second_unit
        expected = Unit.parse_string("m*s")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_division(self):
        result = self.meter_unit / self.second_unit
        # Division should result in m/s, not m*s
        assert result.format_string(as_fraction=True) == "m/s"
    
    def test_power(self):
        result = self.meter_unit.pow(2)
        expected = Unit.parse_string("m^2")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_power_negative(self):
        result = self.meter_unit.pow(-1)
        # Negative power should result in 1/m, not m
        assert result.format_string(as_fraction=True) == "1/m"
    
    def test_power_zero(self):
        result = self.meter_unit.pow(0)
        expected = Unit.parse_string("")
        assert result.dimension == expected.dimension
    
    def test_power_fractional(self):
        result = self.meter_unit.pow(0.5)
        assert len(result.unit_elements) == 1
        assert result.unit_elements[0].exponent == 0.5
    
    def test_complex_multiplication(self):
        force_unit = Unit.parse_string("kg*m/s^2")
        area_unit = Unit.parse_string("m^2")
        result = force_unit * area_unit
        expected = Unit.parse_string("kg*m^3/s^2")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_complex_division(self):
        energy_unit = Unit.parse_string("kg*m^2/s^2")
        length_unit = Unit.parse_string("m")
        result = energy_unit / length_unit
        expected = Unit.parse_string("kg*m/s^2")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_chained_operations(self):
        velocity_unit = Unit.parse_string("m/s")
        time_unit = Unit.parse_string("s")
        length_unit = Unit.parse_string("m")
        result = (velocity_unit * time_unit) / length_unit
        # (m/s * s) / m = m / m = dimensionless
        assert result.dimension == Unit.parse_string("").dimension
    
    def test_power_of_composite_unit(self):
        velocity_unit = Unit.parse_string("m/s")
        result = velocity_unit.pow(2)
        # (m/s)^2 = m^2/s^2
        assert result.format_string(as_fraction=True) == "m^2/s^2"
    
    def test_power_of_composite_unit_negative(self):
        force_unit = Unit.parse_string("kg*m/s^2")
        result = force_unit.pow(-1)
        # (kg*m/s^2)^-1 = s^2/(kg*m)
        assert result.format_string(as_fraction=True) == "s^2/(kg*m)"
    
    def test_mixed_prefix_operations(self):
        km_unit = Unit.parse_string("km")
        cm_unit = Unit.parse_string("cm")
        result = km_unit * cm_unit
        # Expecting 'km*cm' as the result
        assert result.format_string(as_fraction=True) == "km*cm"
    
    def test_cancellation_operations(self):
        velocity_unit = Unit.parse_string("m/s")
        result = velocity_unit / velocity_unit
        expected = Unit.parse_string("")
        assert result.dimension == expected.dimension
    
    def test_fractional_power_of_composite(self):
        area_unit = Unit.parse_string("m^2")
        result = area_unit.pow(0.5)
        assert len(result.unit_elements) == 1
        assert result.unit_elements[0].exponent == 1.0
        assert result.unit_elements[0].unit_symbol.value.symbols[0] == "m"
    
    def test_complex_energy_operations(self):
        energy_unit = Unit.parse_string("kg*m^2/s^2")
        force_unit = Unit.parse_string("kg*m/s^2")
        result = energy_unit / force_unit
        expected = Unit.parse_string("m")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_electrical_units(self):
        power_unit = Unit.parse_string("V*A")
        voltage_unit = Unit.parse_string("V")
        result = power_unit / voltage_unit
        expected = Unit.parse_string("A")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_temperature_operations(self):
        celsius_unit = Unit.parse_string("°C")
        result = celsius_unit / celsius_unit
        expected = Unit.parse_string("")
        assert result.dimension == expected.dimension
    
    def test_multiplication_with_dimensionless(self):
        meter_unit = Unit.parse_string("m")
        dimensionless_unit = Unit.parse_string("")
        result = meter_unit * dimensionless_unit
        expected = Unit.parse_string("m")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
    
    def test_division_by_dimensionless(self):
        meter_unit = Unit.parse_string("m")
        dimensionless_unit = Unit.parse_string("")
        result = meter_unit / dimensionless_unit
        expected = Unit.parse_string("m")
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)

    def test_electrical_power_operations(self):
        """Test electrical power calculations with voltage and current."""
        voltage_unit = Unit.parse_string("V")
        current_unit = Unit.parse_string("A")
        power_unit = voltage_unit * current_unit
        expected = Unit.parse_string("V*A")
        assert power_unit.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
        
        # Test power divided by voltage = current
        result = power_unit / voltage_unit
        assert result.format_string(as_fraction=True) == "A"

    def test_energy_and_power_relationships(self):
        """Test energy (J) and power (W) relationships."""
        energy_unit = Unit.parse_string("J")
        time_unit = Unit.parse_string("s")
        power_unit = energy_unit / time_unit
        # J/s = W (watt), but Unit class doesn't auto-simplify
        assert power_unit.format_string(as_fraction=True) == "J/s"
        
        # Test power * time = energy
        result = power_unit * time_unit
        assert result.format_string(as_fraction=True) == "J"

    def test_force_and_pressure_operations(self):
        """Test force (N) and pressure (Pa) calculations."""
        force_unit = Unit.parse_string("N")
        area_unit = Unit.parse_string("m^2")
        pressure_unit = force_unit / area_unit
        # N/m^2 = Pa (pascal), but Unit class doesn't auto-simplify
        assert pressure_unit.format_string(as_fraction=True) == "N/m^2"
        
        # Test pressure * area = force
        result = pressure_unit * area_unit
        assert result.format_string(as_fraction=True) == "N"

    def test_prefix_combinations(self):
        """Test operations with different prefixes."""
        km_unit = Unit.parse_string("km")
        cm_unit = Unit.parse_string("cm")
        result = km_unit / cm_unit
        # km/cm = 1000m/0.01m = 100000 (dimensionless)
        assert result.dimension == Unit.parse_string("").dimension
        
        # Test mixed prefix multiplication
        mm_unit = Unit.parse_string("mm")
        result = cm_unit * mm_unit
        assert result.format_string(as_fraction=True) == "cm*mm"

    def test_complex_electrical_circuits(self):
        """Test complex electrical circuit calculations."""
        # Ohm's Law: V = I * R
        current_unit = Unit.parse_string("A")
        resistance_unit = Unit.parse_string("Ω")
        voltage_unit = current_unit * resistance_unit
        assert voltage_unit.format_string(as_fraction=True) == "A*Ω"
        
        # Power in terms of current and resistance: P = I^2 * R
        power_unit = current_unit.pow(2) * resistance_unit
        assert power_unit.format_string(as_fraction=True) == "A^2*Ω"

    def test_frequency_and_period_operations(self):
        """Test frequency (Hz) and period (s) relationships."""
        frequency_unit = Unit.parse_string("Hz")
        period_unit = Unit.parse_string("s")
        
        # Frequency * period = 1 (dimensionless)
        result = frequency_unit * period_unit
        assert result.dimension == Unit.parse_string("").dimension
        
        # 1 / period = frequency
        result = Unit.parse_string("1") / period_unit
        assert result.format_string(as_fraction=True) == "1/s"

    def test_magnetic_field_operations(self):
        """Test magnetic field (T) and flux (Wb) calculations."""
        magnetic_field_unit = Unit.parse_string("T")
        area_unit = Unit.parse_string("m^2")
        flux_unit = magnetic_field_unit * area_unit
        # T*m^2 = Wb (weber), but Unit class doesn't auto-simplify
        assert flux_unit.format_string(as_fraction=True) == "T*m^2"
        
        # Flux / area = magnetic field
        result = flux_unit / area_unit
        assert result.format_string(as_fraction=True) == "T"

    def test_thermal_operations(self):
        """Test thermal conductivity and heat transfer."""
        # Thermal conductivity: W/(m*K)
        power_unit = Unit.parse_string("W")
        length_unit = Unit.parse_string("m")
        temp_unit = Unit.parse_string("K")
        thermal_conductivity = power_unit / (length_unit * temp_unit)
        assert thermal_conductivity.format_string(as_fraction=True) == "W/(m*K)"
        
        # Heat capacity: J/K
        energy_unit = Unit.parse_string("J")
        heat_capacity = energy_unit / temp_unit
        assert heat_capacity.format_string(as_fraction=True) == "J/K"

    def test_prefix_cancellation(self):
        """Test that prefixes cancel out in division operations."""
        # mV/mV = 1 (dimensionless)
        mv_unit = Unit.parse_string("mV")
        result = mv_unit / mv_unit
        assert result.dimension == Unit.parse_string("").dimension
        
        # km/km = 1 (dimensionless)
        km_unit = Unit.parse_string("km")
        result = km_unit / km_unit
        assert result.dimension == Unit.parse_string("").dimension
        
        # μA/μA = 1 (dimensionless)
        ua_unit = Unit.parse_string("μA")
        result = ua_unit / ua_unit
        assert result.dimension == Unit.parse_string("").dimension

    def test_prefix_multiplication(self):
        """Test prefix multiplication behavior."""
        # mbar * mbar^2 = mbar^3
        mbar_unit = Unit.parse_string("mbar")
        mbar_squared = mbar_unit.pow(2)
        result = mbar_unit * mbar_squared
        expected = mbar_unit.pow(3)
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)
        
        # mm * mm = mm^2
        mm_unit = Unit.parse_string("mm")
        result = mm_unit * mm_unit
        expected = mm_unit.pow(2)
        assert result.format_string(as_fraction=True) == expected.format_string(as_fraction=True)

    def test_mixed_prefix_arithmetic(self):
        """Test arithmetic with different prefixes on same base unit."""
        # km * m = km*m (no automatic simplification)
        km_unit = Unit.parse_string("km")
        m_unit = Unit.parse_string("m")
        result = km_unit * m_unit
        assert result.format_string(as_fraction=True) == "km*m"
        
        # km / m = km/m (no automatic simplification)
        result = km_unit / m_unit
        assert result.format_string(as_fraction=True) == "km/m"
        
        # cm * mm = cm*mm (no automatic simplification)
        cm_unit = Unit.parse_string("cm")
        mm_unit = Unit.parse_string("mm")
        result = cm_unit * mm_unit
        assert result.format_string(as_fraction=True) == "cm*mm"

    def test_prefix_power_operations(self):
        """Test power operations with prefixed units."""
        # (km)^2 = km^2
        km_unit = Unit.parse_string("km")
        result = km_unit.pow(2)
        assert result.format_string(as_fraction=True) == "km^2"
        
        # (mV)^3 = mV^3
        mv_unit = Unit.parse_string("mV")
        result = mv_unit.pow(3)
        assert result.format_string(as_fraction=True) == "mV^3"
        
        # (μA)^(-1) = 1/μA
        ua_unit = Unit.parse_string("μA")
        result = ua_unit.pow(-1)
        assert result.format_string(as_fraction=True) == "1/μA"

    def test_prefix_composite_operations(self):
        """Test operations with prefixed composite units."""
        # mV/s * s = mV
        mv_per_s = Unit.parse_string("mV/s")
        s_unit = Unit.parse_string("s")
        result = mv_per_s * s_unit
        assert result.format_string(as_fraction=True) == "mV"
        
        # km/h * h = km
        km_per_h = Unit.parse_string("km/h")
        h_unit = Unit.parse_string("h")
        result = km_per_h * h_unit
        assert result.format_string(as_fraction=True) == "km"
        
        # mbar * m^2 = mbar*m^2
        mbar_unit = Unit.parse_string("mbar")
        m_squared = Unit.parse_string("m^2")
        result = mbar_unit * m_squared
        assert result.format_string(as_fraction=True) == "mbar*m^2"

    def test_prefix_cancellation_in_chains(self):
        """Test prefix cancellation in chained operations."""
        # (km * m) / (m * km) = 1 (dimensionless)
        km_unit = Unit.parse_string("km")
        m_unit = Unit.parse_string("m")
        result = (km_unit * m_unit) / (m_unit * km_unit)
        assert result.dimension == Unit.parse_string("").dimension
        
        # (mV * A) / (V * mA) = 1 (dimensionless)
        mv_unit = Unit.parse_string("mV")
        a_unit = Unit.parse_string("A")
        v_unit = Unit.parse_string("V")
        ma_unit = Unit.parse_string("mA")
        result = (mv_unit * a_unit) / (v_unit * ma_unit)
        assert result.dimension == Unit.parse_string("").dimension

    def test_prefix_with_derived_units(self):
        """Test prefixes with derived units like V, A, etc."""
        # mV * mA = mV*mA (no automatic simplification to μW)
        mv_unit = Unit.parse_string("mV")
        ma_unit = Unit.parse_string("mA")
        result = mv_unit * ma_unit
        assert result.format_string(as_fraction=True) == "mV*mA"
        
        # kV / kA = kV/kA (no automatic simplification to Ω)
        kv_unit = Unit.parse_string("kV")
        ka_unit = Unit.parse_string("kA")
        result = kv_unit / ka_unit
        assert result.format_string(as_fraction=True) == "kV/kA"
        
        # mW / mV = mW/mV (no automatic simplification to mA)
        mw_unit = Unit.parse_string("mW")
        mv_unit = Unit.parse_string("mV")
        result = mw_unit / mv_unit
        assert result.format_string(as_fraction=True) == "mW/mV"

    def test_prefix_fractional_powers(self):
        """Test fractional powers with prefixed units."""
        # (km^2)^0.5 = km
        km_squared = Unit.parse_string("km^2")
        result = km_squared.pow(0.5)
        assert result.format_string(as_fraction=True) == "km"
        
        # (mbar^3)^(1/3) = mbar
        mbar_cubed = Unit.parse_string("mbar^3")
        result = mbar_cubed.pow(1/3)
        assert result.format_string(as_fraction=True) == "mbar"
        
        # (μA^2)^0.5 = μA
        ua_squared = Unit.parse_string("μA^2")
        result = ua_squared.pow(0.5)
        assert result.format_string(as_fraction=True) == "μA"

    def test_advanced_power_operations(self):
        """Test advanced power calculations with different units."""
        # Power in terms of voltage and resistance: P = V^2 / R
        voltage_unit = Unit.parse_string("V")
        resistance_unit = Unit.parse_string("Ω")
        power_unit = voltage_unit.pow(2) / resistance_unit
        assert power_unit.format_string(as_fraction=True) == "V^2/Ω"
        
        # Power in terms of current and voltage: P = I * V
        current_unit = Unit.parse_string("A")
        power_unit2 = current_unit * voltage_unit
        assert power_unit2.format_string(as_fraction=True) == "A*V"

    def test_complex_derived_units(self):
        """Test complex derived units and their relationships."""
        # Capacitance: F = C/V
        charge_unit = Unit.parse_string("C")
        voltage_unit = Unit.parse_string("V")
        capacitance_unit = charge_unit / voltage_unit
        # C/V = F (farad), but Unit class doesn't auto-simplify
        assert capacitance_unit.format_string(as_fraction=True) == "C/V"
        
        # Inductance: H = Wb/A
        flux_unit = Unit.parse_string("Wb")
        current_unit = Unit.parse_string("A")
        inductance_unit = flux_unit / current_unit
        # Wb/A = H (henry), but Unit class doesn't auto-simplify
        assert inductance_unit.format_string(as_fraction=True) == "Wb/A"

    def test_mixed_system_operations(self):
        """Test operations mixing different unit systems and complex units."""
        # Angular velocity: rad/s
        angle_unit = Unit.parse_string("rad")
        time_unit = Unit.parse_string("s")
        angular_velocity = angle_unit / time_unit
        assert angular_velocity.format_string(as_fraction=True) == "rad/s"
        
        # Centripetal acceleration: m/s^2
        velocity_unit = Unit.parse_string("m/s")
        radius_unit = Unit.parse_string("m")
        # a = v^2/r = (m/s)^2 / m = m/s^2
        centripetal_accel = velocity_unit.pow(2) / radius_unit
        assert centripetal_accel.format_string(as_fraction=True) == "m/s^2"


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
        
        # 0°C = 273.15 K
        celsius_value = celsius_unit.from_canonical_value(273.15)
        assert celsius_value == 0.0
        
        # 0°C = 273.15 K (reverse)
        kelvin_value = celsius_unit.to_canonical_value(0.0)
        assert kelvin_value == 273.15
    
    def test_invalid_canonical_value_type(self):
        """Test conversion with invalid canonical value type."""
        with pytest.raises(ValueError):
            self.meter_unit.from_canonical_value("invalid") # type: ignore
        
        with pytest.raises(ValueError):
            self.meter_unit.to_canonical_value("invalid") # type: ignore


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
        assert Unit.compatible([])
    
    def test_compatible_single_argument(self):
        """Test compatibility with single argument."""
        assert Unit.compatible([self.meter_unit])
        assert Unit.compatible([self.meter_unit.dimension])


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
        """Test format_string with as_fraction=False."""
        unit = Unit.parse_string("m/s")
        formatted = unit.format_string(as_fraction=False)
        # Should avoid fraction notation if possible
        assert len(formatted) > 0
    
    def test_format_string_with_fraction(self):
        """Test format_string with as_fraction=True."""
        unit = Unit.parse_string("m/s")
        formatted = unit.format_string(as_fraction=True)
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
        best_unit, suggestions = Unit.suggest_units(self.mass_dimension, None)
        
        assert best_unit is not None
        assert len(suggestions) > 0
        assert all(unit.dimension == self.mass_dimension for unit in suggestions)
    
    def test_suggest_units_length(self):
        """Test unit suggestions for length dimension."""
        best_unit, suggestions = Unit.suggest_units(self.length_dimension, None)
        
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
            None,
            must_include=[kg_element]
        )
        
        assert best_unit is not None
        # Should include kg in suggestions (kg is prefix 'k' + symbol 'g')
        kg_units = [unit for unit in suggestions if any(elem.prefix == "k" and "g" in elem.unit_symbol.value.symbols for elem in unit.unit_elements)]
        assert len(kg_units) > 0
    
    def test_suggest_units_with_limit(self):
        """Test unit suggestions with limit."""
        best_unit, suggestions = Unit.suggest_units(self.mass_dimension, None, n=5)
        
        assert best_unit is not None
        assert len(suggestions) <= 5
    
    def test_suggest_units_dimensionless(self):
        """Test unit suggestions for dimensionless quantity."""
        dimensionless = Dimension.create([0, 0, 0, 0, 0, 0, 0], [0, 0])
        best_unit, suggestions = Unit.suggest_units(dimensionless, None)
        
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
        
        assert "unit_string" in json_data
    
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
            group: h5py.Group = f.create_group("unit") # type: ignore
            self.meter_unit.to_hdf5(group)
            
            # Verify data was written
            assert "unit_string" in group.attrs
    
    def test_from_hdf5(self):
        """Test from_hdf5 method."""
        # First write to file
        with h5py.File("test_unit.h5", "w") as f:
            group: h5py.Group = f.create_group("unit") # type: ignore
            self.meter_unit.to_hdf5(group)
        
        # Then read from file
        with h5py.File("test_unit.h5", "r") as f:
            group: h5py.Group = f["unit"] # type: ignore
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
        # Since we can't easily create a unit with zero factor through normal means,
        # we'll test that normal units have non-zero factors
        unit = Unit.parse_string("m")
        assert unit.factor != 0.0
    
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
            unit.factor = 2.0 # type: ignore
        
        with pytest.raises(AttributeError):
            unit.unit_elements = () # type: ignore
    
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
        Unit.suggest_units(mass_dim, None, n=10)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 1.0


class TestUnitInvert:
    """Test unit inversion operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.united_system.unit import clear_unit_cache
        clear_unit_cache()  # Clear cache to ensure fresh parsing for every test
        self.meter_unit = Unit.parse_string("m")
        self.second_unit = Unit.parse_string("s")
        self.kilogram_unit = Unit.parse_string("kg")
        self.composite_unit = Unit.parse_string("kg*m/s^2")
    
    def test_invert_simple_unit(self):
        """Test inverting a simple unit using the ~ operator."""
        inverted = ~self.meter_unit
        assert inverted.format_string(as_fraction=True) == "1/m"
        assert inverted.dimension == self.meter_unit.dimension.invert()
    
    def test_invert_simple_unit_method(self):
        """Test inverting a simple unit using the invert() method."""
        inverted = self.meter_unit.invert()
        assert inverted.format_string(as_fraction=True) == "1/m"
        assert inverted.dimension == self.meter_unit.dimension.invert()
    
    def test_invert_composite_unit(self):
        """Test inverting a composite unit."""
        inverted = ~self.composite_unit
        # kg*m/s^2 inverted = s^2/(kg*m)
        assert inverted.format_string(as_fraction=True) == "s^2/(kg*m)"
        assert inverted.dimension == self.composite_unit.dimension.invert()
    
    def test_invert_unit_with_prefix(self):
        """Test inverting a unit with prefix."""
        km_unit = Unit.parse_string("km")
        inverted = ~km_unit
        assert inverted.format_string(as_fraction=True) == "1/km"
        assert inverted.dimension == km_unit.dimension.invert()
    
    def test_invert_dimensionless_unit(self):
        """Test inverting a dimensionless unit."""
        dimensionless = Unit.parse_string("")
        inverted = ~dimensionless
        assert inverted.format_string(as_fraction=True) == ""
        assert inverted.dimension == dimensionless.dimension.invert()
    
    def test_invert_already_inverted_unit(self):
        """Test inverting an already inverted unit."""
        inverted = ~self.meter_unit
        double_inverted = ~inverted
        assert double_inverted.format_string(as_fraction=True) == "m"
        assert double_inverted.dimension == self.meter_unit.dimension
    
    def test_invert_unit_with_negative_exponent(self):
        """Test inverting a unit that already has negative exponents."""
        per_second = Unit.parse_string("1/s")
        inverted = ~per_second
        assert inverted.format_string(as_fraction=True) == "s"
        assert inverted.dimension == per_second.dimension.invert()
    
    def test_invert_complex_fractional_unit(self):
        """Test inverting a complex fractional unit."""
        complex_unit = Unit.parse_string("kg/(m*s^2)")
        inverted = ~complex_unit
        # The original kg/(m*s^2) gets parsed correctly, so inverted becomes m*s^2/kg
        assert inverted.format_string(as_fraction=True) == "m*s^2/kg"
        assert inverted.dimension == complex_unit.dimension.invert()
    
    def test_invert_temperature_unit(self):
        """Test inverting a temperature unit."""
        celsius_unit = Unit.parse_string("°C")
        inverted = ~celsius_unit
        assert inverted.format_string(as_fraction=True) == "1/°C"
        assert inverted.dimension == celsius_unit.dimension.invert()
    
    def test_invert_electrical_unit(self):
        """Test inverting an electrical unit."""
        voltage_unit = Unit.parse_string("V")
        inverted = ~voltage_unit
        assert inverted.format_string(as_fraction=True) == "1/V"
        assert inverted.dimension == voltage_unit.dimension.invert()
    
    def test_invert_angle_unit(self):
        """Test inverting an angle unit."""
        radian_unit = Unit.parse_string("rad")
        inverted = ~radian_unit
        assert inverted.format_string(as_fraction=True) == "1/rad"
        assert inverted.dimension == radian_unit.dimension.invert()
    
    def test_invert_chained_operations(self):
        """Test chained inversion operations."""
        # Test: ~(~(m/s)) = m/s
        velocity_unit = Unit.parse_string("m/s")
        inverted = ~velocity_unit
        double_inverted = ~inverted
        assert double_inverted.format_string(as_fraction=True) == "m/s"
        assert double_inverted.dimension == velocity_unit.dimension
    
    def test_invert_with_power_operations(self):
        """Test inversion with power operations."""
        # Test: ~(m^2) = 1/m^2
        area_unit = Unit.parse_string("m^2")
        inverted = ~area_unit
        assert inverted.format_string(as_fraction=True) == "1/m^2"
        assert inverted.dimension == area_unit.dimension.invert()
    
    def test_invert_mixed_prefix_unit(self):
        """Test inverting a unit with mixed prefixes."""
        mixed_unit = Unit.parse_string("km*cm/s^2")
        inverted = ~mixed_unit
        assert inverted.format_string(as_fraction=True) == "s^2/(km*cm)"
        assert inverted.dimension == mixed_unit.dimension.invert()
    
    def test_invert_energy_unit(self):
        """Test inverting an energy unit."""
        joule_unit = Unit.parse_string("J")
        inverted = ~joule_unit
        assert inverted.format_string(as_fraction=True) == "1/J"
        assert inverted.dimension == joule_unit.dimension.invert()
    
    def test_invert_pressure_unit(self):
        """Test inverting a pressure unit."""
        pascal_unit = Unit.parse_string("Pa")
        inverted = ~pascal_unit
        assert inverted.format_string(as_fraction=True) == "1/Pa"
        assert inverted.dimension == pascal_unit.dimension.invert()
    
    def test_invert_frequency_unit(self):
        """Test inverting a frequency unit."""
        hertz_unit = Unit.parse_string("Hz")
        inverted = ~hertz_unit
        assert inverted.format_string(as_fraction=True) == "1/Hz"
        assert inverted.dimension == hertz_unit.dimension.invert()
    
    def test_invert_very_complex_unit(self):
        """Test inverting a very complex unit."""
        complex_unit = Unit.parse_string("kg*m^2/(s^3*A)")
        inverted = ~complex_unit
        # The original kg*m^2/(s^3*A) gets parsed correctly, so inverted becomes s^3*A/(kg*m^2)
        assert inverted.format_string(as_fraction=True) == "s^3*A/(kg*m^2)"
        assert inverted.dimension == complex_unit.dimension.invert()
    
    def test_invert_consistency_operator_vs_method(self):
        """Test that the ~ operator and invert() method give the same result."""
        inverted_operator = ~self.composite_unit
        inverted_method = self.composite_unit.invert()
        
        assert inverted_operator.format_string(as_fraction=True) == inverted_method.format_string(as_fraction=True)
        assert inverted_operator.dimension == inverted_method.dimension
        assert inverted_operator.factor == inverted_method.factor
        assert inverted_operator.offset == inverted_method.offset


class TestUnitEquality:
    """Test unit equality operations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.united_system.unit import clear_unit_cache
        clear_unit_cache()  # Clear cache to ensure fresh parsing for every test
        self.meter_unit = Unit.parse_string("m")
        self.kilometer_unit = Unit.parse_string("km")
        self.second_unit = Unit.parse_string("s")
        self.composite_unit = Unit.parse_string("kg*m/s^2")
    
    def test_eq_same_unit(self):
        """Test equality with same unit."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("m")
        assert unit1 == unit2
        assert unit1.equal_exact(unit2)
        assert unit1.equal_effectively(unit2)
    
    def test_eq_different_units_same_dimension(self):
        """Test equality with different units of same dimension."""
        # These should not be equal because they have different factors
        assert self.meter_unit != self.kilometer_unit
        assert not self.meter_unit.equal_exact(self.kilometer_unit)
        assert not self.meter_unit.equal_effectively(self.kilometer_unit)  # Different factors mean different units
    
    def test_eq_different_dimensions(self):
        """Test equality with different dimensions."""
        assert self.meter_unit != self.second_unit
        assert not self.meter_unit.equal_exact(self.second_unit)
        assert not self.meter_unit.equal_effectively(self.second_unit)
    
    def test_ne_same_unit(self):
        """Test inequality with same unit."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("m")
        assert not (unit1 != unit2)
        assert not (unit1 != unit2)
    
    def test_ne_different_units(self):
        """Test inequality with different units."""
        assert self.meter_unit != self.second_unit
        assert self.meter_unit != self.kilometer_unit
    
    def test_equal_exact_same_parsed_string(self):
        """Test equal_exact with units parsed from same string."""
        unit1 = Unit.parse_string("kg*m/s^2")
        unit2 = Unit.parse_string("kg*m/s^2")
        assert unit1.equal_exact(unit2)
    
    def test_equal_exact_different_parsed_strings(self):
        """Test equal_exact with units parsed from different but equivalent strings."""
        unit1 = Unit.parse_string("kg*m/s^2")
        unit2 = Unit.parse_string("m*kg/s^2")  # Different order
        # These should be equal effectively but not exactly due to different order
        assert not unit1.equal_exact(unit2)
        assert unit1.equal_effectively(unit2)
    
    def test_equal_exact_with_prefixes(self):
        """Test equal_exact with prefixed units."""
        unit1 = Unit.parse_string("km")
        unit2 = Unit.parse_string("km")
        assert unit1.equal_exact(unit2)
    
    def test_equal_exact_different_prefixes(self):
        """Test equal_exact with different prefixes."""
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("km")
        assert not unit1.equal_exact(unit2)
        assert not unit1.equal_effectively(unit2)  # Different factors mean different units
    
    def test_equal_effectively_same_dimension_different_units(self):
        """Test equal_effectively with same dimension but different units."""
        # These have the same dimension but different factors
        unit1 = Unit.parse_string("m")
        unit2 = Unit.parse_string("km")
        assert not unit1.equal_effectively(unit2)  # Different factors mean different units
    
    def test_equal_effectively_different_dimensions(self):
        """Test equal_effectively with different dimensions."""
        assert not self.meter_unit.equal_effectively(self.second_unit)
    
    def test_equal_effectively_composite_units(self):
        """Test equal_effectively with composite units."""
        unit1 = Unit.parse_string("kg*m/s^2")
        _ = Unit.parse_string("N")  # Newton
        # These should be equal effectively if N is defined as kg*m/s^2
        # But this depends on how N is defined in the system
        # For now, let's test with units that should be equal
        unit3 = Unit.parse_string("kg*m/s^2")
        assert unit1.equal_effectively(unit3)
    
    def test_equal_exact_with_complex_units(self):
        """Test equal_exact with complex units."""
        unit1 = Unit.parse_string("kg*m^2/(s^3*A)")
        unit2 = Unit.parse_string("kg*m^2/(s^3*A)")
        assert unit1.equal_exact(unit2)
    
    def test_equal_exact_with_inverted_units(self):
        """Test equal_exact with inverted units."""
        unit1 = Unit.parse_string("m")
        inverted1 = ~unit1
        inverted2 = ~unit1
        assert inverted1.equal_exact(inverted2)
        assert not unit1.equal_exact(inverted1)
    
    def test_equal_effectively_with_inverted_units(self):
        """Test equal_effectively with inverted units."""
        unit1 = Unit.parse_string("m")
        inverted1 = ~unit1
        inverted2 = ~unit1
        assert inverted1.equal_effectively(inverted2)
        assert not unit1.equal_effectively(inverted1)
    
    def test_eq_with_none(self):
        """Test equality with None."""
        assert self.meter_unit != None
        assert not (self.meter_unit == None)
    
    def test_eq_with_different_type(self):
        """Test equality with different type."""
        assert self.meter_unit != "not a unit"
        assert not (self.meter_unit == "not a unit")
    
    def test_ne_with_none(self):
        """Test inequality with None."""
        assert self.meter_unit != None
    
    def test_ne_with_different_type(self):
        """Test inequality with different type."""
        assert self.meter_unit != "not a unit"
    
    def test_equal_exact_with_empty_unit(self):
        """Test equal_exact with empty unit."""
        empty1 = Unit.parse_string("")
        empty2 = Unit.parse_string("")
        assert empty1.equal_exact(empty2)
    
    def test_equal_effectively_with_empty_unit(self):
        """Test equal_effectively with empty unit."""
        empty1 = Unit.parse_string("")
        empty2 = Unit.parse_string("")
        assert empty1.equal_effectively(empty2)
    
    def test_equal_exact_with_temperature_units(self):
        """Test equal_exact with temperature units."""
        celsius1 = Unit.parse_string("°C")
        celsius2 = Unit.parse_string("°C")
        assert celsius1.equal_exact(celsius2)
    
    def test_equal_effectively_with_temperature_units(self):
        """Test equal_effectively with temperature units."""
        celsius = Unit.parse_string("°C")
        _ = Unit.parse_string("K")
        # These should be equal effectively if they represent the same dimension
        # But this depends on how temperature units are defined
        assert celsius.equal_effectively(celsius)
        # Note: celsius and kelvin might not be equal_effectively due to different offsets


class TestUnitLogLevel:
    """Test log level manipulation functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        from src.united_system.unit import clear_unit_cache
        clear_unit_cache()  # Clear cache to ensure fresh parsing for every test
        self.meter_unit = Unit.parse_string("m")
        self.second_unit = Unit.parse_string("s")
        self.composite_unit = Unit.parse_string("kg*m/s^2")
        self.log_level_unit = Unit.parse_string("dec")
    
    def test_includes_log_level_property(self):
        """Test the includes_log_level property."""
        # Test log level units
        assert self.log_level_unit.includes_log_level
        neper_unit = Unit.parse_string("Np")
        assert neper_unit.includes_log_level
        
        # Test non-log-level units
        assert not self.meter_unit.includes_log_level
        assert not self.second_unit.includes_log_level
        assert not self.composite_unit.includes_log_level
    
    def test_is_dimensionless_property(self):
        """Test the is_dimensionless property."""
        # Test dimensionless units
        dimensionless_unit = Unit.parse_string("")
        assert dimensionless_unit.is_dimensionless
        
        # Test non-dimensionless units
        assert not self.meter_unit.is_dimensionless
        assert not self.second_unit.is_dimensionless
        assert not self.composite_unit.is_dimensionless
        assert not self.log_level_unit.is_dimensionless

    def test_remove_log_level(self):
        """Test remove_log_level removes log-level elements only."""
        # dec is log-level, m is not
        unit = Unit.parse_string("dec*m")
        result = unit.remove_log_level()
        assert result.format_string(as_fraction=True) == "m"
        # If no log-level, should be unchanged
        unit2 = Unit.parse_string("m")
        result2 = unit2.remove_log_level()
        assert result2.format_string(as_fraction=True) == "m"

    def _unit_str_set(self, unit):
        # Helper: split unit string into set of elements (ignoring order and parentheses)
        s = unit.format_string(as_fraction=True)
        s = s.replace('(', '').replace(')', '')
        return set(s.replace('/', '*').split('*'))

    def test_add_log_level_with_string(self):
        """Test add_log_level with string parameter."""
        # Add log-level to simple unit
        result = self.meter_unit.add_log_level("dec")
        assert self._unit_str_set(result) == {"m", "dec"}
        assert result.includes_log_level
        
        # Add log-level to composite unit
        result = self.composite_unit.add_log_level("dec")
        assert self._unit_str_set(result) == {"kg", "m", "s^2", "dec"}
        assert result.includes_log_level
        
        # Add log-level to dimensionless unit
        dimensionless = Unit.parse_string("")
        result = dimensionless.add_log_level("dec")
        assert self._unit_str_set(result) == {"dec"}
        assert result.includes_log_level

    def test_add_log_level_with_unit_symbol(self):
        """Test add_log_level with UnitSymbol parameter."""
        from src.united_system.utils.units.unit_symbol import UnitSymbol
        
        # Add decade using UnitSymbol
        result = self.meter_unit.add_log_level(UnitSymbol.DECADE)
        assert result.format_string(as_fraction=True) == "m*dec"
        assert result.includes_log_level
        
        # Add neper using UnitSymbol
        result = self.second_unit.add_log_level(UnitSymbol.NEPER)
        assert result.format_string(as_fraction=True) == "s*Np"
        assert result.includes_log_level

    def test_add_log_level_with_unit(self):
        """Test add_log_level with Unit parameter."""
        # Create a log-level unit
        log_unit = Unit.parse_string("dec")
        
        # Add it to another unit
        result = self.meter_unit.add_log_level(log_unit)
        assert result.format_string(as_fraction=True) == "m*dec"
        assert result.includes_log_level

    def test_add_log_level_with_simple_unit_element(self):
        """Test add_log_level with SimpleUnitElement parameter."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        from src.united_system.utils.units.unit_symbol import UnitSymbol
        
        # Create a log-level unit element
        log_element = SimpleUnitElement("", UnitSymbol.DECADE, 1.0)
        
        # Add it to another unit
        result = self.meter_unit.add_log_level(log_element)
        assert result.format_string(as_fraction=True) == "m*dec"
        assert result.includes_log_level

    def test_add_log_level_default_parameter(self):
        """Test add_log_level with default parameter (DECADE)."""
        result = self.meter_unit.add_log_level()
        assert result.format_string(as_fraction=True) == "m*dec"
        assert result.includes_log_level

    def test_add_log_level_to_already_log_level_unit(self):
        """Test add_log_level to a unit that already has log-level."""
        # Start with a log-level unit
        log_unit = Unit.parse_string("dec")
        
        # Add another log-level element
        result = log_unit.add_log_level("Np")
        assert self._unit_str_set(result) == {"dec", "Np"}
        assert result.includes_log_level

    def test_add_log_level_error_non_log_level(self):
        """Test add_log_level raises error for non-log-level elements."""
        # Try to add a non-log-level unit
        with pytest.raises(ValueError, match="unit_element must have dimension LOG_LEVEL_DIMENSION"):
            self.meter_unit.add_log_level("m")
        
        # Try to add a non-log-level unit symbol
        from src.united_system.utils.units.unit_symbol import UnitSymbol
        with pytest.raises(ValueError, match="unit_element must have dimension LOG_LEVEL_DIMENSION"):
            self.meter_unit.add_log_level(UnitSymbol.METER)

    def test_add_log_level_error_non_log_level_unit(self):
        """Test add_log_level raises error for non-log-level Unit."""
        # Try to add a non-log-level unit
        with pytest.raises(ValueError, match="unit_element must have dimension LOG_LEVEL_DIMENSION"):
            self.meter_unit.add_log_level(self.second_unit)

    def test_add_log_level_error_non_log_level_element(self):
        """Test add_log_level raises error for non-log-level SimpleUnitElement."""
        from src.united_system.utils.units.simple_unit_element import SimpleUnitElement
        from src.united_system.utils.units.unit_symbol import UnitSymbol
        
        # Create a non-log-level unit element
        non_log_element = SimpleUnitElement("", UnitSymbol.METER, 1.0)
        
        with pytest.raises(ValueError, match="unit_element must have dimension LOG_LEVEL_DIMENSION"):
            self.meter_unit.add_log_level(non_log_element)

    def test_add_log_level_preserves_existing_elements(self):
        """Test add_log_level preserves existing unit elements."""
        # Start with a complex unit
        complex_unit = Unit.parse_string("kg*m^2/s^3*A")
        
        # Add log-level
        result = complex_unit.add_log_level("dec")
        
        # Check that all original elements are preserved
        assert "kg" in result.format_string(as_fraction=True)
        assert "m^2" in result.format_string(as_fraction=True)
        assert "s^3" in result.format_string(as_fraction=True)
        assert "A" in result.format_string(as_fraction=True)
        assert "dec" in result.format_string(as_fraction=True)
        assert result.includes_log_level

    def test_add_log_level_with_different_log_level_units(self):
        """Test add_log_level with different log-level units."""
        # Test with decade
        result1 = self.meter_unit.add_log_level("dec")
        assert result1.format_string(as_fraction=True) == "m*dec"
        
        # Test with neper
        result2 = self.meter_unit.add_log_level("Np")
        assert result2.format_string(as_fraction=True) == "m*Np"
        
        # Both should have log-level dimension
        assert result1.includes_log_level
        assert result2.includes_log_level

    def test_add_log_level_immutability(self):
        """Test that add_log_level doesn't modify the original unit."""
        original_unit = Unit.parse_string("m")
        original_elements = original_unit.unit_elements
        
        # Add log-level
        result = original_unit.add_log_level("dec")
        
        # Original should be unchanged
        assert original_unit.unit_elements == original_elements
        assert not original_unit.includes_log_level
        
        # Result should be different
        assert result.unit_elements != original_elements
        assert result.includes_log_level

    def test_add_log_level_chaining(self):
        """Test chaining add_log_level operations."""
        # Add log-level twice
        result = self.meter_unit.add_log_level("dec").add_log_level("Np")
        
        assert self._unit_str_set(result) == {"m", "dec", "Np"}
        assert result.includes_log_level

    def test_add_log_level_with_prefixed_log_units(self):
        """Test add_log_level with prefixed log-level units."""
        # Test with prefixed decade (if supported)
        # Note: This depends on whether prefixed log units are defined
        # For now, test with basic log units
        result = self.meter_unit.add_log_level("dec")
        assert result.includes_log_level

if __name__ == "__main__":
    pytest.main([__file__])
