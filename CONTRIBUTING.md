# Contributing to MCP Gateway

Thank you for your interest in contributing to MCP Gateway! We welcome contributions from the community and are grateful for any help you can provide.

## 🎯 Ways to Contribute

- **Bug Reports**: Report bugs using GitHub issues
- **Feature Requests**: Suggest new features or improvements
- **Documentation**: Improve docs, examples, and tutorials
- **Code**: Fix bugs, implement features, or improve performance
- **Testing**: Add tests or improve test coverage

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/mcp-gateway.git
cd mcp-gateway
```

### 2. Set Up Development Environment

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
pip install -e ".[dev]"  # Install development dependencies
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## 📝 Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all checks before submitting:

```bash
# Format code
black .
isort .

# Run linting
flake8 .

# Type checking
mypy .
```

### Testing

- Write tests for new features and bug fixes
- Ensure all tests pass before submitting
- Add tests that cover edge cases

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=gateway --cov-report=html
```

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom routing algorithms
fix: resolve memory leak in session management
docs: update API documentation for /tools endpoint
test: add unit tests for Redis queue processing
```

### Documentation

- Update documentation for any new features
- Include docstrings for new functions and classes
- Update README.md if needed
- Add examples for new functionality

## 🐛 Reporting Bugs

When reporting bugs, please include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, relevant package versions
6. **Logs**: Any relevant error messages or logs

### Bug Report Template

```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Step one
2. Step two
3. See error

**Expected Behavior**
What you expected to happen.

**Environment**
- OS: [e.g., macOS 12.0]
- Python: [e.g., 3.9.0]
- MCP Gateway: [e.g., 0.1.0]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

We welcome feature requests! Please:

1. Check if the feature already exists or is being developed
2. Describe the problem you're trying to solve
3. Explain your proposed solution
4. Consider the impact on existing functionality

## 🔄 Pull Request Process

### Before Submitting

1. **Test**: Ensure all tests pass
2. **Format**: Run code formatters (black, isort)
3. **Lint**: Fix any linting issues
4. **Document**: Update documentation if needed
5. **Changelog**: Consider if your change needs a changelog entry

### Submitting

1. **Title**: Use a clear, descriptive title
2. **Description**: Explain what your PR does and why
3. **Link Issues**: Reference any related issues
4. **Screenshots**: Include screenshots for UI changes

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Manual testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
```

## 🏗️ Architecture Guidelines

### Adding New MCP Servers

1. Create server file in `mcp_servers/` directory
2. Follow the existing pattern (see `exa_server.py`)
3. Include proper error handling and logging
4. Add configuration documentation

### Extending the Gateway

1. Maintain backward compatibility when possible
2. Use async/await for I/O operations
3. Add proper type hints
4. Include comprehensive error handling
5. Log important events and errors

### Session Management

- Follow existing session patterns
- Ensure proper cleanup of resources
- Handle concurrent access safely

## 📚 Resources

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)
- [AsyncIO Documentation](https://docs.python.org/3/library/asyncio.html)

## 💬 Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and general discussion
- **Email**: support@placeholder-labs.com

## 🙏 Recognition

All contributors will be recognized in our README and release notes. Thank you for helping make MCP Gateway better!

## 📄 License

By contributing to MCP Gateway, you agree that your contributions will be licensed under the MIT License.