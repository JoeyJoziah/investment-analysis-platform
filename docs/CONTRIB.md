# Contributing Guide

Thank you for contributing to the Investment Analysis Platform! This guide will help you get started.

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Development Setup](#development-setup)
3. [Coding Standards](#coding-standards)
4. [Testing Requirements](#testing-requirements)
5. [Git Workflow](#git-workflow)
6. [Pull Request Process](#pull-request-process)
7. [Code Review](#code-review)

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker and docker-compose
- Node.js 18+ (for frontend)
- PostgreSQL client tools
- Git

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/your-org/investment-analysis-platform.git
cd investment-analysis-platform

# Copy environment file
cp .env.example .env
# Edit .env with your settings

# Run setup
./setup.sh

# Start development environment
./start.sh dev
```

---

## Development Setup

### Backend Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend directly
cd backend
uvicorn backend.api.main:app --reload

# Run tests
pytest backend/tests/
```

### Frontend Development

```bash
cd frontend/web

# Install dependencies
npm install

# Start development server
npm start

# Run tests
npm test
```

### Docker Development

```bash
# Start all services
./start.sh dev

# View logs
./logs.sh [service-name]

# Stop services
./stop.sh
```

---

## Coding Standards

### Immutability (CRITICAL)

Always create new objects instead of mutating existing ones:

```python
# WRONG: Mutation
def update_portfolio(portfolio, new_stock):
    portfolio['stocks'].append(new_stock)  # MUTATION!
    return portfolio

# CORRECT: Immutability
def update_portfolio(portfolio, new_stock):
    return {
        **portfolio,
        'stocks': [*portfolio['stocks'], new_stock]
    }
```

### File Organization

- **Many small files > Few large files**
- Target: 200-400 lines per file, 800 max
- High cohesion, low coupling
- Organize by feature/domain, not by type

### Error Handling

Always handle errors comprehensively:

```python
try:
    result = await risky_operation()
    return result
except SpecificError as e:
    logger.error(f"Operation failed: {e}")
    raise HTTPException(status_code=400, detail="User-friendly message")
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal server error")
```

### Input Validation

Always validate user input (use Pydantic for Python):

```python
from pydantic import BaseModel, EmailStr, Field

class UserCreate(BaseModel):
    email: EmailStr
    age: int = Field(ge=0, le=150)
    username: str = Field(min_length=3, max_length=50)
```

### Code Quality Checklist

Before submitting code:

- [ ] Code is readable with clear naming
- [ ] Functions are small (<50 lines)
- [ ] Files are focused (<800 lines)
- [ ] No deep nesting (>4 levels)
- [ ] Proper error handling
- [ ] No debug statements (print, console.log)
- [ ] No hardcoded values (use config/env)
- [ ] Immutable patterns used
- [ ] Type hints added (Python) / TypeScript used (frontend)

---

## Testing Requirements

### Minimum Coverage: 80%

All code must have at least 80% test coverage.

### Test Types (ALL Required)

1. **Unit Tests** - Individual functions, utilities, components
2. **Integration Tests** - API endpoints, database operations
3. **E2E Tests** - Critical user flows (Playwright)

### Test-Driven Development (TDD)

We follow TDD methodology:

1. **RED** - Write a failing test first
2. **GREEN** - Write minimal code to pass the test
3. **REFACTOR** - Improve the code while keeping tests green

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/test_recommendations.py

# Run E2E tests
cd frontend/web && npm run test:e2e
```

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
├── fixtures/      # Test fixtures
└── conftest.py    # Pytest configuration
```

---

## Git Workflow

### Branch Naming

```
feature/short-description
fix/issue-number-description
refactor/component-name
docs/documentation-topic
```

### Commit Message Format

```
<type>: <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `refactor` - Code refactoring
- `docs` - Documentation
- `test` - Adding tests
- `chore` - Maintenance
- `perf` - Performance improvement
- `ci` - CI/CD changes

**Examples:**

```
feat: Add stock recommendation endpoint

Implements GET /api/recommendations with filtering by sector and risk level.
Includes caching for improved performance.

Closes #123
```

```
fix: Resolve memory leak in data pipeline

The background loader was not properly closing database connections.
```

### Feature Implementation Workflow

1. **Plan First**
   - Create implementation plan
   - Identify dependencies and risks
   - Break down into phases

2. **TDD Approach**
   - Write tests first (RED)
   - Implement to pass tests (GREEN)
   - Refactor (IMPROVE)
   - Verify 80%+ coverage

3. **Code Review**
   - Self-review before PR
   - Address all feedback
   - Fix CRITICAL and HIGH issues immediately

4. **Commit & Push**
   - Atomic commits (one logical change per commit)
   - Follow conventional commits format

---

## Pull Request Process

### Creating a PR

1. Ensure all tests pass locally
2. Update documentation if needed
3. Create PR with comprehensive description

### PR Template

```markdown
## Summary
[1-3 bullet points describing the change]

## Changes
- [ ] List of specific changes

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Checklist
- [ ] Tests pass
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] No sensitive data in commits
```

### PR Requirements

- All CI checks must pass
- At least one approval required
- No merge conflicts
- Branch is up to date with main

---

## Code Review

### What We Look For

1. **Correctness** - Does it work as intended?
2. **Security** - No vulnerabilities introduced?
3. **Performance** - Any obvious bottlenecks?
4. **Maintainability** - Easy to understand and modify?
5. **Testing** - Adequate test coverage?

### Review Priorities

| Priority | Description | Action |
|----------|-------------|--------|
| CRITICAL | Security issues, data loss risks | Must fix before merge |
| HIGH | Bugs, performance issues | Must fix before merge |
| MEDIUM | Code quality, best practices | Should fix |
| LOW | Style, minor improvements | Nice to have |

### Responding to Reviews

- Address all comments
- Explain decisions if you disagree
- Push fixes as new commits (for easier review)
- Request re-review when ready

---

## Getting Help

- **Documentation**: Check `docs/` folder
- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions for questions

---

## Additional Resources

- [Scripts Reference](./SCRIPTS_REFERENCE.md)
- [Environment Variables](./ENVIRONMENT.md)
- [Runbook](./RUNBOOK.md)
- [API Documentation](http://localhost:8000/docs)
