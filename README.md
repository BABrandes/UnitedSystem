# UnitedSystem

A comprehensive Python library for handling physical units, dimensions, and united data structures with support for dataframes, arrays, and scalars.

## Features

- **Physical Units & Dimensions**: Full support for SI units, derived units, and dimensional analysis
- **United Scalars**: Type-safe scalar values with units (real numbers)
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

# Create united scalars (multiple ways)
distance = RealUnitedScalar.create_from_value_and_unit(100.0, meter)
time = RealUnitedScalar.create_from_value_and_unit(10.0, second)
mass = RealUnitedScalar.create_from_value_and_unit(5.0, kilogram)

# Alternative creation using multiplication
length = 5.0 * Unit("m")
velocity = 10.0 * Unit("m/s")

# Arithmetic operations with automatic unit handling
velocity = distance / time  # 10.0 m/s
momentum = mass * velocity  # 50.0 kg⋅m/s

# Complex unit arithmetic examples
energy = 6.0 * Unit("kJ")  # 6000.0 J
time_interval = 3.0 * Unit("s")  # 3.0 s
power = energy / time_interval  # 2000.0 W (2 kW)

# More examples
force = 10.0 * Unit("N")  # 10.0 N
distance_moved = 5.0 * Unit("m")  # 5.0 m
work_done = force * distance_moved  # 50.0 J

# Unit conversion
velocity_kmh = velocity.in_unit(Unit("km/h"))  # 36.0 km/h
```

### Working with Arrays

```python
from united_system import RealUnitedArray, Unit
import numpy as np

# Create united arrays (multiple ways)
positions = RealUnitedArray([1.0, 2.0, 3.0], Unit("m"))  # Using Unit object
velocities = RealUnitedArray([10.0, 20.0, 30.0], "m/s")  # Using string
masses = [1.0, 2.0, 3.0] * Unit("kg")  # Using multiplication

# Array operations with automatic unit handling
accelerations = velocities / RealUnitedScalar.create_from_value_and_unit(2.0, Unit("s"))  # [5.0, 10.0, 15.0] m/s²

# Broadcasting with proper unit arithmetic
time_array = RealUnitedArray([1.0, 2.0, 3.0], Unit("s"))
displacements = velocities * time_array  # [10.0, 40.0, 90.0] m

# Complex array calculations
energies = RealUnitedArray([6.0, 12.0, 18.0], "kJ")  # [6000.0, 12000.0, 18000.0] J
times = RealUnitedArray([2.0, 3.0, 6.0], "s")
powers = energies / times  # [3000.0, 4000.0, 3000.0] W
```

### DataFrames with Units

```python
from united_system import UnitedDataframe, Unit, DataframeColumnType
from united_system import RealUnitedArray, IntArray, StringArray
from united_system import DataframeColumnKey

# Create united arrays for dataframe columns
positions = RealUnitedArray([1.0, 2.0, 3.0], Unit("m"))
velocities = RealUnitedArray([10.0, 20.0, 30.0], Unit("m/s"))
masses = RealUnitedArray([1.0, 2.0, 3.0], Unit("kg"))
ids = IntArray([1, 2, 3])
names = StringArray(["A", "B", "C"])

# Create dataframe
df = UnitedDataframe.create_from_data({
    DataframeColumnKey("id"): (DataframeColumnType.INTEGER_64, None, ids),
    DataframeColumnKey("name"): (DataframeColumnType.STRING, None, names),
    DataframeColumnKey("position"): (DataframeColumnType.REAL_NUMBER_64, Unit("m"), positions),
    DataframeColumnKey("velocity"): (DataframeColumnType.REAL_NUMBER_64, Unit("m/s"), velocities),
    DataframeColumnKey("mass"): (DataframeColumnType.REAL_NUMBER_64, Unit("kg"), masses)
})

# Access united columns
print(df[DataframeColumnKey("position")].unit)  # Unit("m")
print(df[DataframeColumnKey("velocity")].unit)  # Unit("m/s")

# Perform calculations with automatic unit handling
df[DataframeColumnKey("momentum")] = df[DataframeColumnKey("mass")] * df[DataframeColumnKey("velocity")]  # kg⋅m/s
df[DataframeColumnKey("kinetic_energy")] = 0.5 * df[DataframeColumnKey("mass")] * df[DataframeColumnKey("velocity")]**2  # kg⋅m²/s²

# Complex calculations with proper units
df[DataframeColumnKey("power")] = df[DataframeColumnKey("kinetic_energy")] / (2.0 * Unit("s"))  # W
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
    df.to_hdf5(filename, key="data")
    
    # Load from HDF5
    loaded_df = UnitedDataframe.from_hdf5(filename, key="data", column_key_type=DataframeColumnKey)
    
    # Verify data integrity
    assert df.equals(loaded_df)
finally:
    os.unlink(filename)

# JSON serialization
json_data = df.to_json()
loaded_df = UnitedDataframe.from_json(json_data, column_key_type=DataframeColumnKey)
```

## Advanced Features

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

### Named Quantities

```python
from united_system import NamedQuantity, Unit

# Use predefined named quantities
force = RealUnitedScalar.create_from_value_and_unit(100.0, Unit("N"))
distance = RealUnitedScalar.create_from_value_and_unit(5.0, Unit("m"))
work = force * distance  # 500.0 N⋅m

# Alternative creation using multiplication
force_alt = 100.0 * Unit("N")
distance_alt = 5.0 * Unit("m")

# Access named quantities
assert NamedQuantity.FORCE.unit == Unit("N")
assert NamedQuantity.ENERGY.unit == Unit("J")
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
