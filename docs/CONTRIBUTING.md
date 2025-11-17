# Contributing to CCL + WaveCaster

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow project standards

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/Implementing_julia_bridge_al-ULS.git
cd Implementing_julia_bridge_al-ULS
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
make install-dev

# Install pre-commit hooks
pre-commit install
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Workflow

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-unit
make test-integration
make test-security

# Run with coverage
pytest --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
make format

# Run linters
make lint

# Type checking
mypy .
```

### Running Examples

```bash
# Run example scripts
make examples

# Or run individually
python examples/basic_modulation.py
python examples/ccl_analysis.py
```

## Contribution Guidelines

### Code Style

**Python:**
- Follow PEP 8
- Use Black for formatting (line length: 127)
- Add type hints to all functions
- Write docstrings for public APIs

**Julia:**
- Follow Julia style guide
- Use meaningful variable names
- Document all public functions

**Documentation:**
- Write clear, concise documentation
- Include code examples
- Update README when adding features

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(wavecaster): add OFDM modulation scheme
fix(ccl): handle empty function list
docs(readme): update installation instructions
```

### Pull Request Process

1. **Update Documentation**
   - Update README if needed
   - Add/update docstrings
   - Update CHANGELOG

2. **Add Tests**
   - Write unit tests for new code
   - Ensure >80% code coverage
   - Test edge cases

3. **Run Quality Checks**
   ```bash
   make lint
   make test
   ```

4. **Submit PR**
   - Provide clear description
   - Reference related issues
   - Request review

5. **Address Feedback**
   - Respond to comments
   - Make requested changes
   - Keep discussion focused

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added
- [ ] Integration tests added
- [ ] All tests passing

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed code
- [ ] Commented complex code
- [ ] Updated documentation
- [ ] No new warnings
```

## Areas for Contribution

### High Priority

1. **Testing**
   - Increase test coverage
   - Add integration tests
   - Performance benchmarks

2. **Documentation**
   - API reference
   - User guides
   - Video tutorials

3. **Features**
   - Signal demodulation
   - Web UI
   - Additional modulation schemes

### Medium Priority

1. **Performance**
   - Optimize signal processing
   - Parallel processing
   - Caching strategies

2. **Security**
   - Security audits
   - Vulnerability scanning
   - Best practices

3. **DevOps**
   - CI/CD improvements
   - Docker optimization
   - Monitoring tools

### Low Priority

1. **Enhancements**
   - CLI improvements
   - Logging enhancements
   - Configuration options

2. **Examples**
   - More use cases
   - Advanced examples
   - Best practices

## Reporting Issues

### Bug Reports

Include:
- Clear description
- Steps to reproduce
- Expected vs actual behavior
- Environment details
- Error messages/logs

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternative approaches
- Additional context

### Issue Template

```markdown
## Description
Clear description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS:
- Python version:
- Julia version:
- Package version:

## Additional Context
Screenshots, logs, etc.
```

## Community

### Getting Help

- GitHub Issues: Bug reports and feature requests
- Discussions: Questions and ideas
- Email: project-email@example.com

### Stay Updated

- Watch repository for updates
- Subscribe to release notifications
- Follow project blog

## Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in documentation

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to ask! We're here to help.

Thank you for contributing! 🎉
