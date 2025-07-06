#!/usr/bin/env python3

import sys
sys.path.insert(0, 'src')

from united_system.units.simple.simple_unit import SimpleUnit
from united_system.units.simple.simple_dimension import SimpleDimension

def test_suggest_units():
    """Test the suggest_units method with various examples"""
    
    print("Testing suggest_units method...")
    print("=" * 80)
    
    # Test cases with different dimensions and values
    test_cases = [
        # Length tests
        ("Length: 0.001 meters (1 millimeter)", (0, 0, 1, 0, 0, 0, 0), (0, 0), 0.001),
        ("Length: 1000 meters (1 kilometer)", (0, 0, 1, 0, 0, 0, 0), (0, 0), 1000.0),
        ("Length: 0.000001 meters (1 micrometer)", (0, 0, 1, 0, 0, 0, 0), (0, 0), 0.000001),
        ("Length: 0.3 meters (30 centimeters)", (0, 0, 1, 0, 0, 0, 0), (0, 0), 0.3),
        
        # Time tests
        ("Time: 0.01 seconds (10 milliseconds)", (0, 1, 0, 0, 0, 0, 0), (0, 0), 0.01),
        ("Time: 3600 seconds (1 hour)", (0, 1, 0, 0, 0, 0, 0), (0, 0), 3600.0),
        ("Time: 0.000001 seconds (1 microsecond)", (0, 1, 0, 0, 0, 0, 0), (0, 0), 0.000001),
        
        # Mass tests
        ("Mass: 0.001 kg (1 gram)", (1, 0, 0, 0, 0, 0, 0), (0, 0), 0.001),
        ("Mass: 1000 kg (1 ton)", (1, 0, 0, 0, 0, 0, 0), (0, 0), 1000.0),
        ("Mass: 0.000001 kg (1 milligram)", (1, 0, 0, 0, 0, 0, 0), (0, 0), 0.000001),
        
        # Voltage tests
        ("Voltage: 0.5 volts (500 millivolts)", (1, -3, 2, -1, 0, 0, 0), (0, 0), 0.5),
        ("Voltage: 0.001 volts (1 millivolt)", (1, -3, 2, -1, 0, 0, 0), (0, 0), 0.001),
        ("Voltage: 1000 volts (1 kilovolt)", (1, -3, 2, -1, 0, 0, 0), (0, 0), 1000.0),
        
        # Frequency tests
        ("Frequency: 1000 Hz (1 kilohertz)", (0, -1, 0, 0, 0, 0, 0), (0, 0), 1000.0),
        ("Frequency: 1000000 Hz (1 megahertz)", (0, -1, 0, 0, 0, 0, 0), (0, 0), 1000000.0),
        ("Frequency: 0.001 Hz (1 millihertz)", (0, -1, 0, 0, 0, 0, 0), (0, 0), 0.001),
        
        # Current tests
        ("Current: 0.001 amperes (1 milliampere)", (0, 0, 0, 1, 0, 0, 0), (0, 0), 0.001),
        ("Current: 0.000001 amperes (1 microampere)", (0, 0, 0, 1, 0, 0, 0), (0, 0), 0.000001),
        
        # Force tests
        ("Force: 0.001 newtons (1 millinewton)", (1, -2, 1, 0, 0, 0, 0), (0, 0), 0.001),
        ("Force: 1000 newtons (1 kilonewton)", (1, -2, 1, 0, 0, 0, 0), (0, 0), 1000.0),
        
        # Energy tests
        ("Energy: 0.001 joules (1 millijoule)", (1, -2, 2, 0, 0, 0, 0), (0, 0), 0.001),
        ("Energy: 1000 joules (1 kilojoule)", (1, -2, 2, 0, 0, 0, 0), (0, 0), 1000.0),
        
        # Compound units - more complex dimensional analysis
        ("Voltage scan rate: 1 V/s", (1, -4, 2, -1, 0, 0, 0), (0, 0), 1.0),
        ("Voltage scan rate: 0.001 V/s (1 mV/s)", (1, -4, 2, -1, 0, 0, 0), (0, 0), 0.001),
        ("Voltage scan rate: 1000 V/s (1 kV/s)", (1, -4, 2, -1, 0, 0, 0), (0, 0), 1000.0),
        
        ("Charge: 1 A*s (1 coulomb)", (0, 1, 0, 1, 0, 0, 0), (0, 0), 1.0),
        ("Charge: 0.001 A*s (1 millicoulomb)", (0, 1, 0, 1, 0, 0, 0), (0, 0), 0.001),
        ("Charge: 3600 A*s (1 ampere-hour)", (0, 1, 0, 1, 0, 0, 0), (0, 0), 3600.0),
        
        ("Energy: 1 W*s (1 joule)", (1, -2, 2, 0, 0, 0, 0), (0, 0), 1.0),
        ("Energy: 3600 W*s (1 watt-hour)", (1, -2, 2, 0, 0, 0, 0), (0, 0), 3600.0),
        ("Energy: 0.001 W*s (1 millijoule)", (1, -2, 2, 0, 0, 0, 0), (0, 0), 0.001),
        
        ("Velocity: 10 m/s", (0, -1, 1, 0, 0, 0, 0), (0, 0), 10.0),
        ("Velocity: 0.001 m/s (1 mm/s)", (0, -1, 1, 0, 0, 0, 0), (0, 0), 0.001),
        ("Velocity: 1000 m/s (1 km/s)", (0, -1, 1, 0, 0, 0, 0), (0, 0), 1000.0),
        
        ("Acceleration: 9.81 m/s^2", (0, -2, 1, 0, 0, 0, 0), (0, 0), 9.81),
        ("Acceleration: 0.001 m/s^2", (0, -2, 1, 0, 0, 0, 0), (0, 0), 0.001),
        ("Acceleration: 100 m/s^2", (0, -2, 1, 0, 0, 0, 0), (0, 0), 100.0),
        
        ("Area: 1 m^2", (0, 0, 2, 0, 0, 0, 0), (0, 0), 1.0),
        ("Area: 0.01 m^2 (1 square decimeter)", (0, 0, 2, 0, 0, 0, 0), (0, 0), 0.01),
        ("Area: 10000 m^2 (1 hectare)", (0, 0, 2, 0, 0, 0, 0), (0, 0), 10000.0),
        
        ("Volume: 1 m^3", (0, 0, 3, 0, 0, 0, 0), (0, 0), 1.0),
        ("Volume: 0.001 m^3 (1 liter)", (0, 0, 3, 0, 0, 0, 0), (0, 0), 0.001),
        ("Volume: 1000 m^3", (0, 0, 3, 0, 0, 0, 0), (0, 0), 1000.0),
        
        ("Power density: 1 W/m^2", (1, -3, 0, 0, 0, 0, 0), (0, 0), 1.0),
        ("Power density: 0.001 W/m^2", (1, -3, 0, 0, 0, 0, 0), (0, 0), 0.001),
        ("Power density: 1000 W/m^2", (1, -3, 0, 0, 0, 0, 0), (0, 0), 1000.0),
        
        ("Current density: 1 A/m^2", (0, 0, -2, 1, 0, 0, 0), (0, 0), 1.0),
        ("Current density: 0.001 A/m^2", (0, 0, -2, 1, 0, 0, 0), (0, 0), 0.001),
        ("Current density: 1000 A/m^2", (0, 0, -2, 1, 0, 0, 0), (0, 0), 1000.0),
        
        ("Complex unit: 1 m^2*kg/s^5*A", (1, -5, 2, 1, 0, 0, 0), (0, 0), 1.0),
        ("Complex unit: 0.001 m^2*kg/s^5*A", (1, -5, 2, 1, 0, 0, 0), (0, 0), 0.001),
        ("Complex unit: 1000 m^2*kg/s^5*A", (1, -5, 2, 1, 0, 0, 0), (0, 0), 1000.0),
        
        ("Torque: 1 N*m", (1, -2, 2, 0, 0, 0, 0), (0, 0), 1.0),
        ("Torque: 0.001 N*m (1 millinewton-meter)", (1, -2, 2, 0, 0, 0, 0), (0, 0), 0.001),
        ("Torque: 1000 N*m", (1, -2, 2, 0, 0, 0, 0), (0, 0), 1000.0),
        
        ("Moment of inertia: 1 kg*m^2", (1, 0, 2, 0, 0, 0, 0), (0, 0), 1.0),
        ("Moment of inertia: 0.001 kg*m^2", (1, 0, 2, 0, 0, 0, 0), (0, 0), 0.001),
        ("Moment of inertia: 1000 kg*m^2", (1, 0, 2, 0, 0, 0, 0), (0, 0), 1000.0),
        
        ("Electric field: 1 V/m", (1, -3, 1, -1, 0, 0, 0), (0, 0), 1.0),
        ("Electric field: 0.001 V/m", (1, -3, 1, -1, 0, 0, 0), (0, 0), 0.001),
        ("Electric field: 1000 V/m", (1, -3, 1, -1, 0, 0, 0), (0, 0), 1000.0),
        
        ("Magnetic field: 1 T (tesla)", (1, -2, 0, -1, 0, 0, 0), (0, 0), 1.0),
        ("Magnetic field: 0.001 T (1 millitesla)", (1, -2, 0, -1, 0, 0, 0), (0, 0), 0.001),
        ("Magnetic field: 0.000001 T (1 microtesla)", (1, -2, 0, -1, 0, 0, 0), (0, 0), 0.000001),
        
        ("Viscosity: 1 Pa*s", (1, -1, -1, 0, 0, 0, 0), (0, 0), 1.0),
        ("Viscosity: 0.001 Pa*s", (1, -1, -1, 0, 0, 0, 0), (0, 0), 0.001),
        ("Viscosity: 1000 Pa*s", (1, -1, -1, 0, 0, 0, 0), (0, 0), 1000.0),
        
        ("Angular velocity: 1 rad/s", (0, -1, 0, 0, 0, 0, 0), (0, 1), 1.0),
        ("Angular velocity: 0.1 rad/s", (0, -1, 0, 0, 0, 0, 0), (0, 1), 0.1),
        ("Angular velocity: 100 rad/s", (0, -1, 0, 0, 0, 0, 0), (0, 1), 100.0),
        
        ("Angular acceleration: 1 rad/s^2", (0, -2, 0, 0, 0, 0, 0), (0, 1), 1.0),
        ("Angular acceleration: 0.01 rad/s^2", (0, -2, 0, 0, 0, 0, 0), (0, 1), 0.01),
        ("Angular acceleration: 100 rad/s^2", (0, -2, 0, 0, 0, 0, 0), (0, 1), 100.0),
        
        # No canonical value
        ("No canonical value (Length dimension)", (0, 0, 1, 0, 0, 0, 0), (0, 0), None),
    ]
    
    for i, (description, dim_exp, pseudo_dim_exp, canonical_value) in enumerate(test_cases, 1):
        print(f"\n{i:2d}. {description}")
        print("-" * 60)
        
        dimension = SimpleDimension(dim_exp, pseudo_dim_exp)
        
        try:
            best_unit, alternatives = SimpleUnit.suggest_units(dimension, canonical_value, n=10)
            
            if canonical_value is not None:
                best_value = best_unit.from_canonical_value(canonical_value)
                print(f"   Best unit: {best_unit.format_string(no_fraction=False):>10} = {best_value:>12g}")
                print(f"   Top 10 alternatives:")
                for j, unit in enumerate(alternatives[:10]):
                    value = unit.from_canonical_value(canonical_value)
                    score = SimpleUnit._calculate_value_score(f"{value:g}", unit.format_string(no_fraction=False))
                    print(f"     {j+1:2d}. {unit.format_string(no_fraction=False):>10} = {value:>12g} (score: {score:>6.2f})")
            else:
                print(f"   Best unit: {best_unit.format_string(no_fraction=False):>10}")
                print(f"   Number of alternatives: {len(alternatives)}")
                
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Tests completed!")

if __name__ == "__main__":
    test_suggest_units() 