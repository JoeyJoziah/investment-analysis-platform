---
license: mit
task_categories:
  - text-generation
  - feature-extraction
language:
  - en
tags:
  - investment
  - finance
  - stock-analysis
  - ml
  - ai-agents
  - fastapi
  - react
  - python
  - typescript
  - code
size_categories:
  - 10K<n<100K
---

# Investment Analysis Platform Dataset

A comprehensive dataset documenting the architecture, code structure, API specifications,
ML models, and agent system definitions of an enterprise-grade AI-powered investment
analysis platform.

## Dataset Description

This dataset contains structured metadata and documentation extracted from a full-stack
investment analysis platform featuring:

- **Backend**: FastAPI with 18 API routers, async PostgreSQL, Redis caching
- **Frontend**: React with Redux, TypeScript, Material-UI
- **ML/AI**: LSTM, XGBoost, Prophet models, FinBERT sentiment analysis
- **Infrastructure**: Docker (12 services), Celery, Airflow, Prometheus/Grafana
- **Agent System**: 60+ specialized AI agents with Claude Flow integration
- **Compliance**: SEC 2025 and GDPR compliance built-in

## Dataset Structure

### Splits

- **codebase**: File metadata and extracted code structure (classes, functions, imports)
- **api_endpoints**: REST API endpoint specifications
- **ml_models**: Machine learning model configurations and metadata
- **agents**: AI agent definitions and capabilities
- **metrics**: Project-level metrics and statistics

### Features

#### Codebase Split
- `file_path`: Relative path to the file
- `file_type`: Type of file (python, typescript, tsx, agent_md)
- `file_name`: Name of the file
- `directory`: Parent directory
- `size_bytes`: File size in bytes
- `modified_timestamp`: Last modification timestamp
- `file_hash`: MD5 hash of file content
- `metadata`: JSON string with extracted metadata (classes, functions, imports)

#### API Endpoints Split
- `router_file`: Source router file
- `method`: HTTP method (GET, POST, PUT, DELETE)
- `path`: API path
- `function_name`: Handler function name
- `response_model`: Pydantic response model
- `prefix`: Router prefix

#### ML Models Split
- `file_path`: Path to model file
- `file_name`: Model file name
- `model_type`: Type (pytorch, pickle, joblib, config)
- `size_bytes`: File size
- `size_mb`: File size in MB
- `modified_timestamp`: Last modification

#### Agents Split
- `agent_file`: Agent definition file path
- `agent_name`: Agent identifier
- `category`: Agent category (core, github, consensus, etc.)
- `title`: Agent title
- `description`: Agent description
- `tools`: Available tools
- `subagent_type`: Subagent type identifier

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("financeindustryknowledgeskills/investment-analysis-platform")

# Access specific splits
codebase = dataset["codebase"]
api_endpoints = dataset["api_endpoints"]
ml_models = dataset["ml_models"]
agents = dataset["agents"]

# Query examples
python_files = codebase.filter(lambda x: x["file_type"] == "python")
api_routes = api_endpoints.filter(lambda x: x["method"] == "GET")
pytorch_models = ml_models.filter(lambda x: x["model_type"] == "pytorch")
```

## Project Metrics

- **Total Code Files**: 26,500+
- **Python Files**: 400+
- **TypeScript/TSX Files**: 50+
- **Test Files**: 20
- **API Routers**: 18
- **Database Tables**: 22
- **Docker Services**: 12
- **ML Models**: 7
- **AI Agents**: 60+
- **Project Completion**: 89%
- **Monthly Cost Target**: <$50

## License

MIT License

## Citation

```bibtex
@dataset{investment_analysis_platform_2026,
  title={Investment Analysis Platform Dataset},
  author={financeindustryknowledgeskills},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/financeindustryknowledgeskills/investment-analysis-platform}
}
```
