# Contributing to UnitedSystem

Thank you for your interest in contributing to UnitedSystem! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/united-system.git
   cd united-system
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- A code editor (VS Code, PyCharm, etc.)

### Local Development

1. **Clone your fork**
   ```bash
   git remote add upstream https://github.com/benediktbrandes/united-system.git
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the code style guidelines
   - Write tests for new functionality
   - Update documentation as needed

4. **Run tests locally**
   ```bash
   pytest
   pytest --cov=united_system
   ```

## Code Style

We use several tools to maintain code quality:

### Formatting

- **Black**: Code formatting
  ```bash
  black united_system tests
  ```

- **isort**: Import sorting
  ```bash
  isort united_system tests
  ```

### Linting

- **flake8**: Style guide enforcement
  ```bash
  flake8 united_system tests
  ```

### Type Checking

- **mypy**: Static type checking
  ```bash
  mypy united_system
  ```

- **pyright**: Additional type checking
  ```bash
  pyright united_system
  ```

### Pre-commit Hooks

We recommend setting up pre-commit hooks:

```bash
pip install pre-commit
pre-commit install
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=united_system

# Run specific test files
pytest tests/test_unit.py

# Run specific test functions
pytest tests/test_unit.py::test_unit_creation
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Use fixtures for common setup
- Follow the AAA pattern (Arrange, Act, Assert)

Example test:
```python
def test_unit_creation():
    """Test that units can be created correctly."""
    # Arrange
    unit_symbol = "m"
    
    # Act
    unit = Unit(unit_symbol)
    
    # Assert
    assert unit.symbol == unit_symbol
    assert unit.dimension == Dimension.LENGTH
```

## Documentation

### Code Documentation

- Use docstrings for all public functions and classes
- Follow Google docstring format
- Include type hints for all parameters and return values

Example:
```python
def create_unit(symbol: str, dimension: Dimension) -> Unit:
    """Create a new unit with the given symbol and dimension.
    
    Args:
        symbol: The unit symbol (e.g., 'm', 'kg', 's')
        dimension: The physical dimension of the unit
        
    Returns:
        A new Unit instance
        
    Raises:
        ValueError: If the symbol is invalid
    """
    # Implementation here
```

### Documentation Updates

When adding new features:
1. Update docstrings
2. Update README.md if needed
3. Add examples to the documentation
4. Update any relevant documentation files

## Pull Request Process

1. **Ensure your code is ready**
   - All tests pass
   - Code is formatted and linted
   - Documentation is updated
   - Type checking passes

2. **Create a pull request**
   - Use the provided PR template
   - Provide a clear description
   - Link to any related issues

3. **Review process**
   - At least one maintainer must approve
   - All CI checks must pass
   - Code review feedback must be addressed

4. **Merge**
   - Squash commits if requested
   - Delete the feature branch after merge

## Release Process

### For Maintainers

1. **Prepare the release**
   - Update version in `pyproject.toml`
   - Update CHANGELOG.md
   - Create a release branch

2. **Create a GitHub release**
   - Tag the release
   - Write release notes
   - Upload assets if needed

3. **Publish to PyPI**
   - The GitHub Action will automatically publish when a release is created
   - Ensure the `PYPI_API_TOKEN` secret is set

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality in a backwards-compatible manner
- **PATCH**: Backwards-compatible bug fixes

## Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub Discussions for questions and general discussion
- **Documentation**: Check the [documentation](https://united-system.readthedocs.io/)

## Recognition

Contributors will be recognized in:
- The project README
- Release notes
- Documentation

Thank you for contributing to UnitedSystem! 