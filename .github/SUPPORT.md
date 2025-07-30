# Support

We're here to help! Here are the best ways to get support for UnitedSystem.

## Getting Help

### 📚 Documentation
- **Main Documentation**: [https://united-system.readthedocs.io/](https://united-system.readthedocs.io/)
- **GitHub Setup Guide**: [docs/github-setup.md](docs/github-setup.md)
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md)

### 🐛 Bug Reports
If you've found a bug, please:
1. Check if it's already reported in [Issues](https://github.com/benediktbrandes/united-system/issues)
2. Use the [Bug Report template](https://github.com/benediktbrandes/united-system/issues/new?template=bug_report.md)
3. Include a minimal code example that reproduces the issue

### 💡 Feature Requests
Have an idea for a new feature?
1. Check if it's already requested in [Issues](https://github.com/benediktbrandes/united-system/issues)
2. Use the [Feature Request template](https://github.com/benediktbrandes/united-system/issues/new?template=feature_request.md)
3. Describe the use case and benefits

### ❓ Questions & Discussion
- **GitHub Discussions**: [Start a discussion](https://github.com/benediktbrandes/united-system/discussions)
- **GitHub Issues**: For specific questions about bugs or features

### 🔧 Installation Issues
Having trouble installing UnitedSystem?

**Common Issues:**
- **Python Version**: Ensure you're using Python 3.8 or higher
- **Dependencies**: Try `pip install --upgrade pip` first
- **Virtual Environment**: Use a virtual environment to avoid conflicts

**Installation Commands:**
```bash
# From PyPI (recommended)
pip install united-system

# From GitHub
pip install git+https://github.com/benediktbrandes/united-system.git

# Development installation
git clone https://github.com/benediktbrandes/united-system.git
cd united-system
pip install -e ".[dev]"
```

### 🧪 Testing Issues
Problems with tests or CI/CD?

**Local Testing:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=united_system

# Run specific tests
pytest tests/test_unit.py
```

**CI/CD Issues:**
- Check the [Actions tab](https://github.com/benediktbrandes/united-system/actions)
- Look for specific error messages
- Ensure all dependencies are properly specified

### 📦 Packaging Issues
Problems with building or distributing?

**Building:**
```bash
# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*
```

### 🔒 Security Issues
Found a security vulnerability?
- **DO NOT** create a public issue
- **DO** email benedikt.brandes@me.com
- See [SECURITY.md](.github/SECURITY.md) for details

## Community Resources

### 📖 Examples
- Check the [examples/](examples/) directory for usage examples
- Look at the [README.md](README.md) for quick start examples

### 🧪 Testing
- Run the test suite to verify your installation
- Check [tests/](tests/) for examples of how to use the library

### 📝 Contributing
Want to contribute?
1. Read [CONTRIBUTING.md](CONTRIBUTING.md)
2. Set up your development environment
3. Pick an issue to work on
4. Submit a pull request

## Response Times

- **Critical Issues**: Within 24 hours
- **Bug Reports**: Within 48 hours
- **Feature Requests**: Within 1 week
- **General Questions**: Within 3-5 days

## Before Asking for Help

1. **Search existing issues** to see if your question has been answered
2. **Check the documentation** for your specific use case
3. **Try the examples** to see if they work for you
4. **Provide a minimal example** that reproduces your issue
5. **Include your environment details** (OS, Python version, etc.)

## Environment Information

When reporting issues, please include:
- **Operating System**: Windows/macOS/Linux version
- **Python Version**: `python --version`
- **UnitedSystem Version**: `pip show united-system`
- **Dependencies**: NumPy, Pandas versions
- **Error Messages**: Full error traceback

## Code of Conduct

Please be respectful and follow our [Code of Conduct](.github/CODE_OF_CONDUCT.md) when seeking help.

## Contact

- **Email**: benedikt.brandes@me.com
- **GitHub**: [@benediktbrandes](https://github.com/benediktbrandes)
- **Issues**: [GitHub Issues](https://github.com/benediktbrandes/united-system/issues)
- **Discussions**: [GitHub Discussions](https://github.com/benediktbrandes/united-system/discussions)

Thank you for using UnitedSystem! 🚀 