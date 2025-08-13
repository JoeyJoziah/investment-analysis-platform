# GitHub Actions CI/CD Pipeline

This directory contains a comprehensive GitHub Actions CI/CD pipeline for the Investment Analysis App, designed for production-ready, secure, and cost-optimized deployment.

## 🚀 Pipeline Overview

### Workflows Created

1. **[CI Pipeline](workflows/ci.yml)** - Main continuous integration
2. **[Staging Deployment](workflows/staging-deploy.yml)** - Automated staging deployment
3. **[Production Deployment](workflows/production-deploy.yml)** - Production release deployment
4. **[Security Scanning](workflows/security-scan.yml)** - Comprehensive security analysis
5. **[Dependency Updates](workflows/dependency-updates.yml)** - Automated dependency management
6. **[Database Migration Check](workflows/migration-check.yml)** - Database migration validation
7. **[Cleanup](workflows/cleanup.yml)** - Automated resource cleanup
8. **[Reusable Test Workflow](workflows/reusable-test.yml)** - Reusable testing components
9. **[Reusable Build Workflow](workflows/reusable-build.yml)** - Reusable build components

### Configuration Files

- **[Dependabot Config](dependabot.yml)** - Automated dependency updates
- **[CodeQL Config](codeql/codeql-config.yml)** - Code security analysis
- **[GitLeaks Config](../.gitleaks.toml)** - Secret scanning configuration
- **[Issue Template](ISSUE_TEMPLATE/bug_report.yml)** - Bug report template
- **[Pull Request Template](pull_request_template.md)** - PR template

## 📋 Features

### ✅ Continuous Integration (CI)
- **Multi-version testing**: Python 3.11 & 3.12, Node.js 18
- **Code quality**: Black, isort, flake8, mypy, pylint, ESLint, Prettier
- **Test coverage**: 85% minimum with Codecov integration
- **Security scanning**: Bandit, safety checks, SARIF upload
- **Docker builds**: Multi-arch support with layer caching
- **Integration tests**: Full stack testing with Docker Compose
- **Parallel execution**: Optimized for speed and cost

### 🚀 Deployment Pipelines
- **Staging deployment**: Automatic on main branch pushes
- **Production deployment**: Release tag or manual trigger
- **Security gates**: Vulnerability scanning before deployment
- **Blue-green strategy**: Zero-downtime deployments
- **Rollback capability**: Automatic rollback on failure
- **Health checks**: Comprehensive post-deployment validation
- **Performance testing**: Load testing on staging

### 🔒 Security & Compliance
- **Code analysis**: CodeQL, Semgrep, ESLint security rules
- **Dependency scanning**: Safety, pip-audit, npm audit, Snyk
- **Secret detection**: TruffleHog, GitLeaks, custom patterns
- **Container security**: Trivy, Hadolint, Dockle scanning
- **SARIF integration**: GitHub Security tab integration
- **Financial data protection**: Investment-specific security rules

### 🔄 Automated Maintenance
- **Dependency updates**: Dependabot with custom schedules
- **Security patches**: Daily security update monitoring
- **Resource cleanup**: Automated artifact and cache cleanup
- **Database migrations**: Forward/backward testing with performance checks
- **Performance monitoring**: Automated load testing

## 🎯 Triggers & Schedules

| Workflow | Triggers | Schedule |
|----------|----------|-----------|
| CI Pipeline | Push to main/develop, PR | On demand |
| Staging Deploy | Push to main | Automatic |
| Production Deploy | Release tags | On release |
| Security Scan | Push, PR, Schedule | Daily 2 AM UTC |
| Dependency Updates | Schedule, Manual | Weekly (Mon-Fri) |
| Migration Check | Migration file changes | On demand |
| Cleanup | Schedule, Manual | Weekly (Sunday 2 AM) |

## 🛠 Setup Instructions

### 1. Required Secrets

Add these secrets in your GitHub repository settings:

#### Container Registry
```
GITHUB_TOKEN  # Automatically provided
```

#### Kubernetes Deployment
```
STAGING_KUBECONFIG     # Base64 encoded kubeconfig for staging
PRODUCTION_KUBECONFIG  # Base64 encoded kubeconfig for production
```

#### Database & Services
```
STAGING_DATABASE_URL    # PostgreSQL connection string
PRODUCTION_DATABASE_URL # PostgreSQL connection string
STAGING_REDIS_URL      # Redis connection string
PRODUCTION_REDIS_URL   # Redis connection string
STAGING_JWT_SECRET     # JWT signing key for staging
PRODUCTION_JWT_SECRET  # JWT signing key for production
```

#### External APIs
```
ALPHA_VANTAGE_API_KEY  # Alpha Vantage API key
FINNHUB_API_KEY        # Finnhub API key
POLYGON_API_KEY        # Polygon.io API key
NEWS_API_KEY           # NewsAPI key
```

#### Security & Monitoring
```
CODECOV_TOKEN          # Codecov integration token
SNYK_TOKEN            # Snyk security scanning
SLACK_WEBHOOK_URL     # Slack notifications
SMTP_USERNAME         # Email notifications
SMTP_PASSWORD         # Email notifications
NOTIFICATION_EMAIL    # Notification recipient
```

#### Staging/Production URLs
```
STAGING_API_URL        # https://api-staging.investment-analysis.com
STAGING_FRONTEND_URL   # https://staging.investment-analysis.com
PRODUCTION_API_URL     # https://api.investment-analysis.com
PRODUCTION_FRONTEND_URL # https://investment-analysis.com
```

### 2. Environment Variables

Add these in your repository settings (optional):
```
EMAIL_ENABLED=true     # Enable email notifications
SLACK_WEBHOOK_URL      # Already in secrets, reference for notifications
```

### 3. Branch Protection Rules

Configure these branch protection rules for `main`:
- Require pull request reviews (2 reviewers)
- Require status checks to pass:
  - `backend-test`
  - `backend-quality` 
  - `frontend-test`
  - `frontend-quality`
  - `docker-build`
  - `integration-test`
- Require branches to be up to date
- Include administrators
- Allow force pushes: No
- Allow deletions: No

## 🎨 Workflow Customization

### Matrix Builds
Customize Python/Node versions in workflows:
```yaml
strategy:
  matrix:
    python-version: ['3.11', '3.12']  # Add/remove versions
    node-version: ['18', '20']        # Add/remove versions
```

### Test Selection
Skip slow tests on PR builds:
```yaml
- name: Run tests
  run: |
    pytest backend/tests/ -m "not slow" # Skip slow tests
```

### Deployment Environments
Add new environments by:
1. Creating new secrets with environment prefix
2. Adding environment to staging/production workflows
3. Updating Kubernetes manifests

### Security Scanning
Customize security tools in `security-scan.yml`:
- Adjust severity thresholds
- Add/remove scanning tools
- Modify file exclusions

## 📊 Monitoring & Observability

### GitHub Actions Insights
- View workflow runs in Actions tab
- Monitor success/failure rates
- Track deployment frequency
- Review security scan results

### Notifications
- Slack integration for failures and deployments
- Email notifications for critical issues
- GitHub Security tab for vulnerability reports
- Step summaries for detailed results

### Artifacts & Reports
- Test coverage reports (HTML + XML)
- Security scan results (JSON + SARIF)
- Performance test reports
- Build artifacts and SBOMs
- Migration test results

## 🔧 Troubleshooting

### Common Issues

**1. Database Connection Failures**
```bash
# Check if DATABASE_URL is correct
echo $DATABASE_URL
# Verify PostgreSQL service is running
pg_isready -h localhost -p 5432 -U postgres
```

**2. Container Build Failures**
```bash
# Clear Docker buildx cache
docker buildx prune -f
# Check Dockerfile syntax
docker build --no-cache .
```

**3. Test Failures**
```bash
# Run tests locally with same environment
export DATABASE_URL="postgresql://postgres:testpass@localhost:5432/test_db"
pytest backend/tests/ -v
```

**4. Secret Scanning False Positives**
- Update `.gitleaks.toml` allowlist
- Add `# gitleaks:allow` comment to specific lines
- Use environment variables for dynamic values

### Performance Optimization

**1. Reduce Workflow Runtime**
- Use caching for dependencies
- Parallelize independent jobs
- Skip unnecessary steps on PR builds

**2. Cost Optimization**
- Use `ubuntu-latest` runners (cheapest)
- Cancel redundant runs with concurrency groups
- Cleanup old artifacts and packages

**3. Resource Limits**
```yaml
# Add resource limits for long-running jobs
timeout-minutes: 30  # Prevent runaway jobs
```

## 🚀 Production Readiness Checklist

### Before First Deployment
- [ ] All required secrets configured
- [ ] Branch protection rules enabled
- [ ] Kubernetes cluster ready
- [ ] Database backups configured
- [ ] Monitoring dashboards set up
- [ ] Incident response procedures documented

### Pre-Production Testing
- [ ] Run CI pipeline on feature branch
- [ ] Test staging deployment end-to-end
- [ ] Verify security scans pass
- [ ] Test rollback procedures
- [ ] Load test staging environment
- [ ] Verify monitoring and alerting

### Post-Deployment
- [ ] Monitor application health
- [ ] Check performance metrics
- [ ] Verify financial calculations
- [ ] Test key user journeys
- [ ] Review security scan results
- [ ] Update documentation

## 📚 Additional Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Kubernetes Deployment Guide](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [Investment Analysis App Documentation](../README.md)

## 🤝 Contributing

When adding new workflows or modifying existing ones:

1. Test thoroughly in a fork first
2. Follow existing naming conventions
3. Add appropriate documentation
4. Update this README if needed
5. Consider security implications
6. Test rollback procedures

## 📞 Support

For pipeline issues:
1. Check GitHub Actions logs
2. Review security scan results
3. Verify secrets and environment variables
4. Test locally with same configuration
5. Create issue with detailed error logs

---

*This CI/CD pipeline is designed for the Investment Analysis App and optimized for financial data processing, security, and compliance requirements.*