# GitHub Secrets Configuration Checklist

Complete this checklist to ensure all required secrets are configured for your CI/CD pipeline.

## Quick Setup Instructions

1. Navigate to your repository on GitHub
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"** for each secret below
4. Copy the exact **Secret Name** and provide your **Secret Value**

## Database & Application Secrets

### Core Application
- [ ] **`SECRET_KEY`**
  - **Description**: Django/FastAPI application secret key
  - **Example**: `django-insecure-your-secret-key-minimum-50-characters-long-random`
  - **How to generate**: `python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"`

- [ ] **`JWT_SECRET_KEY`**
  - **Description**: JWT token signing key
  - **Example**: `your-jwt-secret-32-characters-min`
  - **How to generate**: `openssl rand -hex 32`

### Database Configuration
- [ ] **`DB_PASSWORD`**
  - **Description**: PostgreSQL database password
  - **Example**: `secure_postgres_password_123!`
  - **Requirements**: At least 12 characters, alphanumeric + symbols

- [ ] **`REDIS_PASSWORD`**
  - **Description**: Redis cache password
  - **Example**: `secure_redis_password_456!`
  - **Requirements**: At least 12 characters, alphanumeric + symbols

## External API Keys

### Financial Data APIs
- [ ] **`ALPHA_VANTAGE_API_KEY`**
  - **Description**: Alpha Vantage API key (25 calls/day free)
  - **Get it from**: https://www.alphavantage.co/support/#api-key
  - **Example**: `ABCDEF1234567890`
  - **Usage**: Daily stock prices, technical indicators

- [ ] **`FINNHUB_API_KEY`**
  - **Description**: Finnhub API key (60 calls/minute free)
  - **Get it from**: https://finnhub.io/register
  - **Example**: `c123456789abcdef`
  - **Usage**: Real-time stock data, company profiles

- [ ] **`POLYGON_API_KEY`**
  - **Description**: Polygon.io API key (5 calls/minute free)
  - **Get it from**: https://polygon.io/pricing
  - **Example**: `ABCD1234567890EFGH`
  - **Usage**: Market data, stock aggregates

- [ ] **`NEWS_API_KEY`**
  - **Description**: NewsAPI key for sentiment analysis
  - **Get it from**: https://newsapi.org/pricing
  - **Example**: `1234567890abcdef1234567890abcdef`
  - **Usage**: Financial news for sentiment analysis

## Container Registry & Deployment

### GitHub Container Registry
- [ ] **`REGISTRY_USERNAME`**
  - **Description**: GitHub username for container registry
  - **Example**: `your-github-username`
  - **Value**: Your exact GitHub username

- [ ] **`REGISTRY_TOKEN`**
  - **Description**: GitHub Personal Access Token with packages scope
  - **How to create**:
    1. GitHub Profile → Settings → Developer settings
    2. Personal access tokens → Tokens (classic)
    3. Generate new token with scopes: `repo`, `write:packages`, `read:packages`
  - **Example**: `ghp_1234567890abcdef1234567890abcdef12345678`

## Cloud Infrastructure

### DigitalOcean (Primary Option)
- [ ] **`DIGITALOCEAN_ACCESS_TOKEN`**
  - **Description**: DigitalOcean API token for Kubernetes
  - **How to create**:
    1. DigitalOcean Control Panel → API → Tokens
    2. Generate New Token → Read and Write scopes
  - **Example**: `dop_v1_[YOUR-TOKEN-HERE]` (64 character token)

- [ ] **`KUBECONFIG_CONTENT`**
  - **Description**: Kubernetes cluster configuration
  - **How to get**:
    1. DigitalOcean → Kubernetes → Your Cluster
    2. Download Config File
    3. Copy entire file contents
  - **Value**: Complete YAML content of kubeconfig file

### AWS (Alternative Option)
- [ ] **`AWS_ACCESS_KEY_ID`**
  - **Description**: AWS Access Key for EKS cluster
  - **Example**: `AKIA1234567890ABCDEF`
  
- [ ] **`AWS_SECRET_ACCESS_KEY`**
  - **Description**: AWS Secret Access Key
  - **Example**: `abcd1234567890efgh1234567890ijkl1234567890`

- [ ] **`AWS_REGION`**
  - **Description**: AWS region for deployment
  - **Example**: `us-east-1`

## Production Environment

### Production Database
- [ ] **`PRODUCTION_DATABASE_URL`**
  - **Description**: Production PostgreSQL connection string
  - **Example**: `postgresql://username:password@host:5432/investment_db`
  - **Format**: `postgresql://[username]:[password]@[host]:[port]/[database]`

- [ ] **`PRODUCTION_REDIS_URL`**
  - **Description**: Production Redis connection string
  - **Example**: `redis://username:password@host:6379`
  - **Format**: `redis://[username]:[password]@[host]:[port]`

## Monitoring & Notifications

### Slack Integration
- [ ] **`SLACK_WEBHOOK_URL`**
  - **Description**: Slack webhook for deployment notifications
  - **How to create**:
    1. Slack → Apps → Incoming Webhooks
    2. Add to Workspace → Choose channel
  - **Example**: `https://hooks.slack.com/services/T12345678/B12345678/abcd1234567890efgh1234567890`

- [ ] **`SLACK_BOT_TOKEN`** (Optional)
  - **Description**: Slack bot token for advanced notifications
  - **How to create**:
    1. Slack API → Create App → Bot User
    2. OAuth & Permissions → Install App
  - **Example**: `xoxb-[YOUR-BOT-TOKEN-HERE]`

### Monitoring Services (Optional)
- [ ] **`SENTRY_DSN`** (Optional)
  - **Description**: Sentry error tracking DSN
  - **Example**: `https://abcd1234567890@o123456.ingest.sentry.io/123456`

- [ ] **`DATADOG_API_KEY`** (Optional)
  - **Description**: Datadog monitoring API key
  - **Example**: `abcd1234567890efgh1234567890ijkl12345678`

## Security & Development

### Additional Security
- [ ] **`ENCRYPTION_KEY`** (Optional)
  - **Description**: Additional encryption key for sensitive data
  - **How to generate**: `openssl rand -base64 32`
  - **Example**: `abcd1234567890efgh1234567890ijkl1234567890==`

### Development & Testing
- [ ] **`TEST_DATABASE_URL`** (Optional)
  - **Description**: Test database connection string
  - **Example**: `postgresql://test_user:test_pass@localhost:5432/test_db`

- [ ] **`CODECOV_TOKEN`** (Optional)
  - **Description**: Codecov token for coverage reports
  - **Get it from**: https://codecov.io/
  - **Example**: `abcd1234-5678-90ef-ghij-1234567890ab`

## Verification Commands

After setting all secrets, run these commands to verify:

### Check API Keys Locally
```bash
# Test Alpha Vantage
curl "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&apikey=YOUR_API_KEY"

# Test Finnhub
curl -X GET "https://finnhub.io/api/v1/quote?symbol=AAPL&token=YOUR_TOKEN"

# Test Polygon.io
curl "https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/2023-01-09/2023-01-09?adjusted=true&sort=asc&limit=120&apikey=YOUR_API_KEY"

# Test NewsAPI
curl "https://newsapi.org/v2/everything?q=tesla&sortBy=publishedAt&apiKey=YOUR_API_KEY"
```

### Verify Docker Registry Access
```bash
# Login to GitHub Container Registry
echo $REGISTRY_TOKEN | docker login ghcr.io -u $REGISTRY_USERNAME --password-stdin

# Test push (after building an image)
docker push ghcr.io/your-username/investment-analysis-app:test
```

### Check Kubernetes Access
```bash
# Save kubeconfig and test
echo "$KUBECONFIG_CONTENT" > kubeconfig-test.yaml
export KUBECONFIG=kubeconfig-test.yaml
kubectl get nodes
```

## Secrets Summary

**Total Required Secrets**: 25+ secrets
**Time to Configure**: 30-60 minutes
**Security Level**: Production-ready

### Critical Secrets (Must Have)
1. `SECRET_KEY` - Application security
2. `JWT_SECRET_KEY` - Authentication
3. `DB_PASSWORD` - Database access
4. `ALPHA_VANTAGE_API_KEY` - Core data source
5. `FINNHUB_API_KEY` - Real-time data
6. `REGISTRY_TOKEN` - Container deployment
7. `KUBECONFIG_CONTENT` - Production deployment

### High Priority Secrets
8. `POLYGON_API_KEY` - Additional data source
9. `NEWS_API_KEY` - Sentiment analysis
10. `SLACK_WEBHOOK_URL` - Notifications
11. `PRODUCTION_DATABASE_URL` - Production database

### Optional Secrets
12. AWS credentials (if using AWS)
13. Monitoring service tokens
14. Additional security keys

## Security Best Practices

1. **Never commit secrets to code**
2. **Use strong, unique passwords**
3. **Rotate secrets regularly (every 90 days)**
4. **Use least privilege access**
5. **Monitor secret usage**
6. **Enable 2FA on all accounts**

## Troubleshooting

### Common Issues
- **Secret not found**: Check spelling (case-sensitive)
- **Permission denied**: Verify token scopes
- **API limits exceeded**: Check free tier limits
- **Connection refused**: Verify URLs and ports

### Getting Help
- Check workflow logs in GitHub Actions
- Review API documentation
- Test secrets individually
- Contact support if issues persist

---

**Next Step**: After completing this checklist, proceed with the main CI/CD setup guide!