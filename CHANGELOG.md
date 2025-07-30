# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI/CD workflow
- Security scanning with CodeQL and Snyk
- Issue and PR templates
- Contributing guidelines
- Comprehensive documentation

### Changed
- Updated project structure for GitHub compatibility
- Enhanced development workflow

### Fixed
- Various minor bugs and improvements

## [1.0.0] - 2025-01-XX

### Added
- Initial release of UnitedSystem
- Physical units and dimensions support
- United scalars (real and complex)
- United arrays with NumPy compatibility
- United DataFrames with Pandas compatibility
- HDF5 and JSON serialization
- Type safety with full type hints
- Thread-safe operations
- CLI interface
- Comprehensive test suite
- Documentation with examples

### Features
- **Physical Units & Dimensions**: Full support for SI units, derived units, and custom unit definitions
- **United Scalars**: Type-safe scalar values with units (real and complex)
- **United Arrays**: NumPy-compatible arrays with unit support
- **United DataFrames**: Pandas-compatible dataframes with unit-aware columns
- **HDF5 Serialization**: Efficient storage and retrieval of united data structures
- **JSON Serialization**: Human-readable data exchange format
- **Type Safety**: Full type hints and protocol support
- **Thread Safety**: Thread-safe operations with reader-writer locks

### Technical Details
- Python 3.8+ support
- NumPy >= 1.20.0 dependency
- Pandas >= 1.3.0 dependency
- H5py >= 3.0.0 dependency
- ReaderWriterLock >= 1.0.0 dependency
- Apache 2.0 license 