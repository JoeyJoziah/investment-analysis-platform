# 🚀 Investment Analysis Platform - Initialization Guide

## Quick Start (Recommended)

The easiest way to initialize and start the platform:

```bash
./start.sh
```

This script will:
- Check prerequisites
- Run initialization if needed
- Start all services
- Show access URLs

## Manual Initialization Steps

If you prefer to initialize manually or need to troubleshoot:

### 1. Prerequisites Check

Ensure you have:
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- Git

### 2. Environment Setup

```bash
# Create .env file
cp .env.example .env

# Edit .env and add your API keys
nano .env
```

Required API keys (all have free tiers):
- **Alpha Vantage**: https://www.alphavantage.co/support/#api-key
- **Finnhub**: https://finnhub.io/register
- **Polygon.io**: https://polygon.io/dashboard/signup
- **NewsAPI**: https://newsapi.org/register

### 3. Run Initialization

```bash
python3 init_app.py
```

This will:
1. ✅ Check prerequisites
2. ✅ Set up environment
3. ✅ Build Docker images
4. ✅ Start infrastructure (PostgreSQL, Redis, Elasticsearch)
5. ✅ Initialize database schema
6. ✅ Start backend services
7. ✅ Start frontend
8. ✅ Start monitoring (Prometheus, Grafana)
9. ✅ Load sample data
10. ✅ Verify installation

### 4. Verify Installation

After initialization, check that everything is running:

```bash
# Check service status
docker-compose ps

# Test API health
curl http://localhost:8000/api/health

# View logs
docker-compose logs -f
```

## 🌐 Access Points

Once initialized, access the platform at:

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Main web interface |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |
| **API Health** | http://localhost:8000/api/health | Health check endpoint |
| **Prometheus** | http://localhost:9090 | Metrics monitoring |
| **Grafana** | http://localhost:3001 | Dashboard visualization |

## 🛠️ Common Operations

### Start Services
```bash
./start.sh
# or
docker-compose up -d
```

### Stop Services
```bash
./stop.sh
# or
docker-compose down
```

### Restart Services
```bash
./restart.sh
# or
docker-compose restart
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
```

### Load Initial Stock Data
```bash
docker-compose exec backend python -m backend.utils.load_initial_stocks
```

### Run Tests
```bash
python run_all_tests.py
```

### Debug Issues
```bash
python debug_validate.py
```

### Fix Common Errors
```bash
python fix_common_errors.py
```

## 🔧 Troubleshooting

### Issue: API Keys Not Set
**Solution**: Update `.env` file with actual API keys

### Issue: Port Already in Use
**Solution**: 
```bash
# Check what's using the port
lsof -i :8000

# Stop conflicting service or change port in docker-compose.yml
```

### Issue: Database Connection Failed
**Solution**:
```bash
# Ensure PostgreSQL is running
docker-compose up -d postgres

# Wait 10 seconds for startup
sleep 10

# Retry initialization
python3 init_app.py
```

### Issue: Frontend Not Loading
**Solution**:
```bash
# Rebuild frontend
docker-compose build frontend

# Restart frontend
docker-compose restart frontend
```

### Issue: Out of Disk Space
**Solution**:
```bash
# Clean up Docker
docker system prune -a

# Remove unused volumes
docker volume prune
```

## 📊 Initial Data

The platform loads sample data for 50 major US stocks including:
- **Technology**: AAPL, MSFT, GOOGL, NVDA, META
- **Financials**: JPM, BAC, WFC, V, MA
- **Healthcare**: UNH, JNJ, PFE, LLY, ABBV
- **Consumer**: WMT, AMZN, TSLA, PG, KO
- **Energy**: XOM, CVX, COP
- **And more...**

## 🔐 Security Notes

1. **Change default passwords** in production
2. **Use strong JWT secret** (generate with `openssl rand -hex 32`)
3. **Enable HTTPS** for production deployment
4. **Restrict database access** to localhost only
5. **Keep API keys secure** and never commit them

## 📈 Next Steps

After successful initialization:

1. **Explore the API**
   - Visit http://localhost:8000/docs
   - Try sample requests
   - Review available endpoints

2. **Use the Frontend**
   - Browse to http://localhost:3000
   - Search for stocks
   - View analysis and recommendations

3. **Monitor Performance**
   - Check Grafana dashboards
   - Monitor API usage
   - Track costs

4. **Load More Data**
   - Use data ingestion endpoints
   - Schedule regular updates
   - Monitor data quality

## 🆘 Getting Help

If you encounter issues:

1. Check `debug_report.json` for detailed diagnostics
2. Review Docker logs: `docker-compose logs`
3. Run validation: `python debug_validate.py`
4. Check the [troubleshooting guide](#-troubleshooting)

## ✅ Success Indicators

You know initialization was successful when:
- All Docker containers are running (`docker-compose ps`)
- API health check returns 200 OK
- Frontend loads at http://localhost:3000
- No errors in `docker-compose logs`
- Sample stocks appear in the database

---

**Ready to analyze stocks? Start exploring at http://localhost:3000! 🚀**