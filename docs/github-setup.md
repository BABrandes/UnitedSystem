# GitHub Setup Guide

This guide explains how to set up and use UnitedSystem with GitHub.

## Repository Structure

The UnitedSystem repository is organized as follows:

```
united-system/
├── .github/                    # GitHub-specific configurations
│   ├── workflows/             # GitHub Actions workflows
│   ├── ISSUE_TEMPLATE/        # Issue templates
│   └── dependabot.yml         # Dependabot configuration
├── united_system/             # Main package source
├── tests/                     # Test suite
├── docs/                      # Documentation
├── examples/                  # Example code
├── .pre-commit-config.yaml    # Pre-commit hooks
├── pyproject.toml            # Project configuration
├── README.md                 # Main documentation
├── CONTRIBUTING.md           # Contributing guidelines
└── CHANGELOG.md             # Version history
```

## GitHub Features

### Continuous Integration/Continuous Deployment (CI/CD)

The repository includes comprehensive GitHub Actions workflows:

- **CI/CD Pipeline**: Automated testing, linting, and building
- **Security Scanning**: CodeQL and Snyk vulnerability scanning
- **Dependency Updates**: Automated dependency updates via Dependabot
- **Code Quality**: Pre-commit hooks for consistent code style

### Workflow Triggers

- **Push to main/develop**: Runs tests and linting
- **Pull Requests**: Runs full CI pipeline
- **Releases**: Automatically publishes to PyPI

### Security Features

- **Dependency Review**: Scans for vulnerabilities in dependencies
- **CodeQL Analysis**: Static analysis for security issues
- **Snyk Integration**: Additional security scanning
- **Weekly Security Updates**: Automated security checks

## Setting Up Your Development Environment

### Prerequisites

1. **GitHub Account**: Create an account at [github.com](https://github.com)
2. **Git**: Install Git on your system
3. **Python**: Install Python 3.8 or higher
4. **Code Editor**: VS Code, PyCharm, or your preferred editor

### Fork and Clone

1. **Fork the repository**:
   - Go to [https://github.com/benediktbrandes/united-system](https://github.com/benediktbrandes/united-system)
   - Click the "Fork" button
   - Choose your GitHub account

2. **Clone your fork**:
   ```bash
   git clone https://github.com/your-username/united-system.git
   cd united-system
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/benediktbrandes/united-system.git
   ```

### Development Setup

1. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Verify setup**:
   ```bash
   pytest
   ```

## Working with GitHub

### Creating Issues

Use the provided issue templates:
- **Bug Report**: For reporting bugs and issues
- **Feature Request**: For suggesting new features

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**:
   - Follow the code style guidelines
   - Write tests for new functionality
   - Update documentation

3. **Test your changes**:
   ```bash
   pytest
   black united_system tests
   isort united_system tests
   flake8 united_system tests
   mypy united_system
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**:
   - Go to your fork on GitHub
   - Click "Compare & pull request"
   - Fill out the PR template
   - Submit the PR

### Code Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Review Process**: Maintainers review your code
3. **Address Feedback**: Make requested changes
4. **Merge**: Once approved, your PR is merged

## GitHub Actions

### Available Workflows

- **CI/CD**: Runs on every push and PR
- **Security**: Weekly security scanning
- **Dependabot**: Automated dependency updates

### Workflow Status

Check the status of workflows:
- Go to the "Actions" tab on GitHub
- View workflow runs and their results
- Debug failed workflows if needed

## Release Process

### For Maintainers

1. **Update version** in `pyproject.toml`
2. **Update CHANGELOG.md** with release notes
3. **Create a release** on GitHub:
   - Go to "Releases" tab
   - Click "Create a new release"
   - Tag the release (e.g., v1.0.1)
   - Write release notes
   - Publish the release

4. **Automated Publishing**: GitHub Actions will automatically publish to PyPI

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md`
- [ ] Run full test suite
- [ ] Update documentation if needed
- [ ] Create GitHub release
- [ ] Verify PyPI publication

## Troubleshooting

### Common Issues

1. **CI/CD Failures**:
   - Check the Actions tab for error details
   - Run tests locally to reproduce issues
   - Fix linting issues with `black` and `isort`

2. **Dependency Issues**:
   - Update dependencies with `pip install -U -e ".[dev]"`
   - Check for conflicting package versions

3. **Pre-commit Hook Failures**:
   - Run `pre-commit run --all-files` to fix issues
   - Manually fix any remaining issues

### Getting Help

- **GitHub Issues**: Use the issue tracker for bugs and feature requests
- **GitHub Discussions**: Use Discussions for questions and general discussion
- **Documentation**: Check the main documentation for usage examples

## Best Practices

### Code Quality

- Follow the established code style (Black, isort, flake8)
- Write comprehensive tests
- Include type hints
- Document new features

### Git Workflow

- Use descriptive commit messages
- Keep commits focused and atomic
- Update documentation with code changes
- Respond to review feedback promptly

### Security

- Never commit sensitive information
- Keep dependencies updated
- Report security issues privately
- Follow security best practices

## Resources

- [GitHub Documentation](https://docs.github.com/)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Python Packaging Guide](https://packaging.python.org/)
- [Contributing Guidelines](CONTRIBUTING.md) 