# UnitedSystem

A comprehensive Python library for handling physical units, dimensions, and united data structures with support for dataframes, arrays, and scalars.

## Features

- **Physical Units & Dimensions**: Full support for SI units, derived units, and custom unit definitions
- **United Scalars**: Type-safe scalar values with units (real and complex)
- **United Arrays**: NumPy-compatible arrays with unit support
- **United DataFrames**: Pandas-compatible dataframes with unit-aware columns
- **HDF5 Serialization**: Efficient storage and retrieval of united data structures
- **JSON Serialization**: Human-readable data exchange format
- **Type Safety**: Full type hints and protocol support
- **Thread Safety**: Thread-safe operations with reader-writer locks

## Installation

### From PyPI (recommended)

```bash
pip install united-system
```

### From GitHub

```bash
pip install git+https://github.com/benediktbrandes/united-system.git
```

### From source

```bash
git clone https://github.com/benediktbrandes/united-system.git
cd united-system
pip install -e .
```

### Development installation

```bash
git clone https://github.com/benediktbrandes/united-system.git
cd united-system
pip install -e ".[dev]"
```

## Quick Start

### Basic Unit Operations

```python
from united_system import Unit, Dimension, RealUnitedScalar

# Create units
meter = Unit("m")
second = Unit("s")
kilogram = Unit("kg")

# Create united scalars
distance = RealUnitedScalar(100.0, meter)
time = RealUnitedScalar(10.0, second)
mass = RealUnitedScalar(5.0, kilogram)

# Arithmetic operations
velocity = distance / time  # 10.0 m/s
momentum = mass * velocity  # 50.0 kg⋅m/s

# Unit conversion
velocity_kmh = velocity.to(Unit("km/h"))  # 36.0 km/h
```

### Working with Arrays

```python
from united_system import RealUnitedArray, Unit
import numpy as np

# Create united arrays
positions = RealUnitedArray([1.0, 2.0, 3.0], Unit("m"))
velocities = RealUnitedArray([10.0, 20.0, 30.0], Unit("m/s"))

# Array operations
accelerations = velocities / RealUnitedScalar(2.0, Unit("s"))  # [5.0, 10.0, 15.0] m/s²

# Broadcasting
time_array = RealUnitedArray([1.0, 2.0, 3.0], Unit("s"))
displacements = velocities * time_array  # [10.0, 40.0, 90.0] m
```

### DataFrames with Units

```python
from united_system import UnitedDataframe, Unit, ColumnType
from united_system import RealUnitedArray, IntArray, StringArray

# Create united arrays for dataframe columns
positions = RealUnitedArray([1.0, 2.0, 3.0], Unit("m"))
velocities = RealUnitedArray([10.0, 20.0, 30.0], Unit("m/s"))
masses = RealUnitedArray([1.0, 2.0, 3.0], Unit("kg"))
ids = IntArray([1, 2, 3])
names = StringArray(["A", "B", "C"])

# Create dataframe
df = UnitedDataframe({
    "id": ids,
    "name": names,
    "position": positions,
    "velocity": velocities,
    "mass": masses
})

# Access united columns
print(df["position"].unit)  # Unit("m")
print(df["velocity"].unit)  # Unit("m/s")

# Perform calculations
df["momentum"] = df["mass"] * df["velocity"]  # kg⋅m/s
df["kinetic_energy"] = 0.5 * df["mass"] * df["velocity"]**2  # kg⋅m²/s²
```

### Serialization

```python
import tempfile
import os

# HDF5 serialization
with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
    filename = tmp.name

try:
    # Save to HDF5
    df.to_hdf5(filename, "data")
    
    # Load from HDF5
    loaded_df = UnitedDataframe.from_hdf5(filename, "data")
    
    # Verify data integrity
    assert df.equals(loaded_df)
finally:
    os.unlink(filename)

# JSON serialization
json_data = df.to_json()
loaded_df = UnitedDataframe.from_json(json_data)
```

## Advanced Features

### Custom Units

```python
from united_system import Unit, NamedQuantity

# Define custom units
custom_force = Unit("custom_N", NamedQuantity.FORCE)
custom_energy = Unit("custom_J", NamedQuantity.ENERGY)

# Use in calculations
force = RealUnitedScalar(100.0, custom_force)
distance = RealUnitedScalar(5.0, Unit("m"))
work = force * distance  # 500.0 custom_N⋅m
```

### Dimension Analysis

```python
from united_system import Dimension

# Check dimensions
velocity_dim = Dimension.LENGTH / Dimension.TIME
force_dim = Dimension.MASS * Dimension.LENGTH / Dimension.TIME**2

# Verify unit dimensions
assert Unit("m/s").dimension == velocity_dim
assert Unit("N").dimension == force_dim
```

### Complex Units

```python
from united_system import ComplexUnitedScalar, Unit

# Complex values with units
complex_voltage = ComplexUnitedScalar(3 + 4j, Unit("V"))
complex_current = ComplexUnitedScalar(2 + 1j, Unit("A"))

# Complex arithmetic
complex_power = complex_voltage * complex_current  # (2+11j) V⋅A
```

## Documentation

For detailed documentation, visit [https://united-system.readthedocs.io/](https://united-system.readthedocs.io/)

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=united_system

# Run specific test categories
pytest tests/test_unit.py
pytest tests/test_dataframe.py
```

### Code Quality

```bash
# Format code
black united_system tests

# Sort imports
isort united_system tests

# Type checking
mypy united_system
pyright united_system

# Linting
flake8 united_system tests
```

### Building Documentation

```bash
# Install docs dependencies
pip install -e ".[docs]"

# Build documentation
cd docs
make html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for detailed information.

### Quick Start for Contributors

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install development dependencies (`pip install -e ".[dev]"`)
4. Set up pre-commit hooks (`pre-commit install`)
5. Make your changes
6. Run tests (`pytest`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

### Development Status

[![CI/CD](https://github.com/benediktbrandes/united-system/workflows/CI/CD/badge.svg)](https://github.com/benediktbrandes/united-system/actions)
[![Security](https://github.com/benediktbrandes/united-system/workflows/Security/badge.svg)](https://github.com/benediktbrandes/united-system/actions)
[![Codecov](https://codecov.io/gh/benediktbrandes/united-system/branch/main/graph/badge.svg)](https://codecov.io/gh/benediktbrandes/united-system)
[![PyPI](https://img.shields.io/pypi/v/united-system.svg)](https://pypi.org/project/united-system/)
[![Python](https://img.shields.io/pypi/pyversions/united-system.svg)](https://pypi.org/project/united-system/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENCE)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use UnitedSystem in your research, please cite:

```bibtex
@software{unitedsystem2025,
  title={UnitedSystem: A Python library for physical units and united data structures},
  author={Brandes, Benedikt Axel},
  year={2025},
  url={https://github.com/benediktbrandes/united-system}
}
```

## Acknowledgments

- Inspired by the need for type-safe unit handling in scientific computing
- Built on top of NumPy and Pandas for efficient numerical operations
- Uses HDF5 for efficient data storage and exchange
# Trigger CI run
