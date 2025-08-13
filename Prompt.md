# Optimized Prompt for Claude Opus 4: Building a World-Leading, Low-Cost Investment Analysis and Recommendation Application

## Introduction

You are Claude Opus 4, an advanced AI model specialized in generating comprehensive, deployable software solutions. Your objective is to design, develop, and deliver a state-of-the-art financial analysis and investment recommendation application that automates workflows for analyzing all publicly traded stocks on the NYSE, NASDAQ, and AMEX exchanges (approximately 6,000+ tickers as of July 2025). The application must operate fully autonomously, generating daily recommendations without required user input, while supporting optional customizations such as risk preferences or sector filters.

To achieve world-leading status, incorporate cutting-edge features including ensemble ML models for forecasting, advanced NLP with LLMs like FinBERT, explainable AI, alternative data integration, and real-time risk monitoring. Ensure scalability for future expansions (e.g., international markets) and compliance with 2025 SEC and GDPR regulations.

Prioritize low-cost operation: Utilize exclusively free/open-source tools and data sources with generous free tiers (e.g., APIs with high call limits), implement batch processing and caching to minimize API usage, optimize computations for CPU efficiency, and design for deployment on low-cost cloud providers like AWS Free Tier or DigitalOcean Kubernetes (targeting under $50/month for moderate usage). Include built-in cost monitoring (e.g., API call trackers and alerts) to prevent overages.

Generate complete, executable code, configurations, and documentation for immediate deployment. Structure your output with modular sections, including code blocks in appropriate languages (Python for backend/ML, JavaScript for frontend).

## Core Functional Requirements

### Stock Universe Management
Maintain a dynamic, up-to-date database of all U.S.-listed stocks. Fetch tickers and metadata from free APIs such as Alpha Vantage, Finnhub, Financial Modeling Prep (FMP), Polygon.io (free tier for real-time), and NASDAQ Data Link. Automatically handle corporate actions via SEC EDGAR APIs using the sec-api library. Store in PostgreSQL with ticker-based indexing and sharding for scalability; use Elasticsearch for fast queries. Schedule daily updates with Apache Airflow and Celery, batching requests to reduce API calls (e.g., cache tickers for 24 hours).

### Automated Data Ingestion and Validation
Build ETL pipelines using Apache Airflow for zero-manual-intervention ingestion from free sources like Alpha Vantage, Finnhub, FMP, Polygon.io, and Marketstack. Validate data with Great Expectations, implementing auto-retries and error alerts via ELK Stack. Optimize costs by batching daily pulls (e.g., one bulk call per exchange) and compressing data with snappy.

### Financial and Trading Data Integration
Integrate SEC filings from data.sec.gov, generating dynamic, Excel-exportable models (balance sheets, income statements) using openpyxl. Pull daily metrics (prices, volume, volatility, moving averages) from Polygon.io or Alpha Vantage; compute derivatives with NumPy/SciPy. Cache historical data to avoid redundant API hits.

### Anecdotal and Sentiment Data Feeds
Aggregate from NewsAPI.org, Marketaux, RSS feeds (Feedparser), and X via Tweepy (respecting free tier limits). Use Hugging Face Transformers with FinBERT for sentiment analysis and anomaly detection. Modularize feeds as REST endpoints, with cost-saving measures like querying only high-impact keywords.

### Automation and Maintenance
Employ Apache Kafka for real-time streaming and Kubernetes for auto-scaling. Monitor integrity with Prometheus/Grafana, including API usage dashboards to alert on potential cost overages (e.g., nearing free tier limits).

### User Experience & Output
Create an intuitive interface with React.js and Material-UI for dashboards, React Native for mobile, and Plotly Dash for visualizations. Support automated daily reports (PDF/Excel exports) and optional user inputs. Package in Docker for minimal setup.

## Advanced Components

### Alternative Data & Enrichment
Integrate low-cost alternatives: OpenWeatherMap for sector impacts, NASA Earthdata for satellite imagery, Google Trends for web analytics. Parse unstructured data (transcripts, filings) with FinBERT or lightweight BERT variants, flagging insights efficiently on CPU.

### Personalized & Explainable AI Recommendations
Store optional preferences in PostgreSQL. Tailor via scikit-learn clustering and ensemble models. Use SHAP/LIME for explanations, generating visualizations. Default to balanced recommendations for autonomous mode.

### Predictive Analytics & Scenario Modeling
Implement forecasting with PyTorch LSTMs, Prophet (for time series), and ensemble methods (e.g., combining with scikit-learn). Add Monte Carlo simulations via NumPy/SciPy. Optimize by training models offline and inferring in batches.

### Collaboration & Social Investing Features
Integrate Discourse for forums and anonymized benchmarking with Pandas. Ensure low overhead by hosting on free tiers.

### Compliance & Risk Monitoring
Use SimpleRisk for regulatory tracking. Build real-time dashboards with Grafana for stress tests and exposures, compliant with 2025 SEC rules.

### Integration & Interoperability
API-first with FastAPI, supporting OAuth for brokers. Include voice commands via Web Speech API. Design for low-latency, cost-effective integrations.

### Continuous Learning & Model Improvement
Retraining with MLflow; A/B tests via Optuna. Collect feedback autonomously (e.g., from market outcomes) to refine models without user input.

### Security, Privacy & Ethics
End-to-end encryption with HTTPS/Fernet. Anonymize per GDPR/SEC; audit biases with AIF360. Include ethical sourcing disclosures.

## Cost Optimization Strategies
- **API Management**: Track calls with custom logging; fallback to cached data if limits approach (e.g., Alpha Vantage: 25 calls/day free).
- **Compute Efficiency**: Use CPU-optimized models (e.g., quantized PyTorch); schedule jobs during off-peak hours.
- **Deployment**: Kubernetes on DigitalOcean ($10/month starter) or AWS Free Tier; auto-scale down to zero pods when idle.
- **Monitoring**: Integrate cost alerts (e.g., via Prometheus) to notify if usage exceeds free thresholds.

## Delivery Instructions
Provide step-by-step code generation for each module, including:
- Technologies: As specified, with alternatives like Prophet for ML.
- Architecture Diagram (ASCII):

```
+-------------------+     +-------------------+     +-------------------+
|   User Interface  |     |    API Layer      |     |   Analytics Layer |
| (React/Material-UI| <-->| (FastAPI/REST)    | <-->| (ML Models:       |
|  React Native)    |     |                   |     |  PyTorch, SHAP)   |
+-------------------+     +-------------------+     +-------------------+
          |                            |                           |
          v                            v                           v
+-------------------+     +-------------------+     +-------------------+
|   Data Pipelines  |     |   Database Layer  |     |   Monitoring      |
| (Airflow/Kafka)   | <-->| (PostgreSQL/      | <-->| (Prometheus/      |
|                   |     |  Elasticsearch)   |     |  Grafana)         |
+-------------------+     +-------------------+     +-------------------+
          ^                            ^
          |                            |
+-------------------+     +-------------------+
|   Data Sources    |     |   Compliance      |
| (APIs: Alpha      | --> | (SimpleRisk/      |
|  Vantage, EDGAR)  |     |  Audit Logs)      |
+-------------------+     +-------------------+
```

- Project Plan: Planning (2 weeks), Ingestion (4 weeks), Analytics (6 weeks), UI/Integrations (4 weeks), Compliance/Security (3 weeks), Testing/Deployment (4 weeks). Total: 23 weeks.
- Compliance: Use public data; include audit scripts.
- Documentation: Sphinx for devs, MkDocs for users.

Output all code, scripts, Docker/Kubernetes files, and instructions. If clarification needed, specify; otherwise, deliver fully.