#!/usr/bin/env python3
"""
Example demonstrating the user-friendly RealScalar wrapper.

This shows how easy it is to work with physical quantities using the 
UnitedSystem package through the RealScalar wrapper class.
"""

from src.united_system import RealUnitedScalar

def main():
    print("=== UNITED SYSTEM - REAL SCALAR EXAMPLE ===")
    print()
    
    # Create scalars with simple string constructors
    print("1. Creating physical quantities:")
    distance = RealUnitedScalar("5 m")
    time = RealUnitedScalar("2.5 s")
    voltage = RealUnitedScalar("3.14 V")
    current = RealUnitedScalar("0.5 A")
    mass = RealUnitedScalar("10 kg")
    
    print(f"   Distance: {distance}")
    print(f"   Time: {time}")
    print(f"   Voltage: {voltage}")
    print(f"   Current: {current}")
    print(f"   Mass: {mass}")
    print()
    
    # Create scalars with separate value and unit
    print("2. Alternative constructor:")
    temperature = RealUnitedScalar(298.15, "K")
    pressure = RealUnitedScalar(1.013e5, "Pa")
    print(f"   Temperature: {temperature}")
    print(f"   Pressure: {pressure}")
    print()
    
    # Natural arithmetic operations
    print("3. Natural arithmetic operations:")
    velocity = distance / time
    power = voltage * current
    energy = power * time
    force = mass * velocity / time  # F = ma
    
    print(f"   velocity = distance / time = {distance} / {time} = {velocity}")
    print(f"   power = voltage * current = {voltage} * {current} = {power}")
    print(f"   energy = power * time = {power} * {time} = {energy}")
    print(f"   force = mass * acceleration = {mass} * {velocity}/{time} = {force}")
    print()
    
    # Mixed operations with numbers
    print("4. Mixed operations with numbers:")
    double_distance = distance * 2
    half_time = time / 2.0
    doubled_voltage = 2 * voltage
    
    print(f"   double distance = {distance} * 2 = {double_distance}")
    print(f"   half time = {time} / 2.0 = {half_time}")
    print(f"   doubled voltage = 2 * {voltage} = {doubled_voltage}")
    print()
    
    # Division by zero preserves units
    print("5. Division by zero preserves units:")
    infinite_velocity = distance / 0
    zero_time = RealUnitedScalar("0 s")
    infinite_acceleration = velocity / zero_time
    
    print(f"   infinite velocity = {distance} / 0 = {infinite_velocity}")
    print(f"   infinite acceleration = {velocity} / {zero_time} = {infinite_acceleration}")
    print()
    
    # Properties and checks
    print("6. Properties and utility methods:")
    print(f"   velocity.value = {velocity.value}")
    print(f"   velocity.is_finite() = {velocity.is_finite()}")
    print(f"   infinite_velocity.is_infinite() = {infinite_velocity.is_infinite()}")
    print(f"   velocity.is_nan() = {velocity.is_nan()}")
    print()
    
    # Compatibility checking
    print("7. Compatibility checking:")
    another_distance = RealUnitedScalar("10 km")
    print(f"   distance compatible with another_distance: {distance.compatible_with(another_distance)}")
    print(f"   distance compatible with velocity: {distance.compatible_with(velocity)}")
    print()
    
    # Equality
    print("8. Equality:")
    same_distance = RealUnitedScalar("5 m")
    different_distance = RealUnitedScalar("5 km")
    print(f"   distance == same_distance: {distance == same_distance}")
    print(f"   distance == different_distance: {distance == different_distance}")
    print()
    
    # Physics calculations
    print("9. Real physics calculations:")
    
    # Ohm's law: V = I * R
    resistance = voltage / current
    print(f"   Ohm's law: resistance = {voltage} / {current} = {resistance}")
    
    # Kinetic energy: KE = 0.5 * m * v²
    kinetic_energy = 0.5 * mass * (velocity ** 2)
    print(f"   Kinetic energy = 0.5 * {mass} * {velocity}² = {kinetic_energy}")
    
    # Work: W = F * d
    work = force * distance
    print(f"   Work = {force} * {distance} = {work}")
    
    print()
    print("✅ ALL OPERATIONS SUCCESSFUL!")
    print()
    print("🎯 Key Benefits:")
    print("   • Simple, intuitive constructors")
    print("   • Natural arithmetic operations")
    print("   • Automatic unit handling")
    print("   • Type safety and error checking")
    print("   • Division by zero preserves units")
    print("   • Fast operations without expensive unit suggestions")
    print("   • Clean, readable code")

if __name__ == "__main__":
    main() 