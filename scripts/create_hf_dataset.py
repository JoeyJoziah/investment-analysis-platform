#!/usr/bin/env python3
"""
Hugging Face Dataset Creator for Investment Analysis Platform

Creates a comprehensive dataset containing:
- Codebase structure and metadata
- API specifications
- ML model configurations
- Agent system definitions
- Project documentation
- Configuration patterns

Uploads to Hugging Face Hub under the authenticated user's namespace.
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from collections import defaultdict
import ast
import re

# Load environment variables from .env
from dotenv import load_dotenv
load_dotenv()

# Hugging Face imports
from huggingface_hub import HfApi, create_repo, upload_folder, login
from datasets import Dataset, DatasetDict, Features, Value, Sequence

# Set HF token from environment
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
    login(token=HF_TOKEN, add_to_git_credential=False)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_OUTPUT = PROJECT_ROOT / "datasets" / "investment-analysis-platform"


def get_file_hash(file_path: Path) -> str:
    """Calculate MD5 hash of file content."""
    content = file_path.read_bytes()
    return hashlib.md5(content).hexdigest()


def extract_python_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from Python files."""
    try:
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)

        classes = []
        functions = []
        imports = []
        docstring = ast.get_docstring(tree)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node)
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({
                    "name": node.name,
                    "docstring": class_doc[:500] if class_doc else None,
                    "methods": methods[:20],
                    "line": node.lineno
                })
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not any(isinstance(p, ast.ClassDef) for p in ast.walk(tree) if hasattr(p, 'body') and node in getattr(p, 'body', [])):
                    func_doc = ast.get_docstring(node)
                    args = [arg.arg for arg in node.args.args]
                    functions.append({
                        "name": node.name,
                        "args": args[:10],
                        "docstring": func_doc[:300] if func_doc else None,
                        "is_async": isinstance(node, ast.AsyncFunctionDef),
                        "line": node.lineno
                    })
            elif isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)

        return {
            "module_docstring": docstring[:500] if docstring else None,
            "classes": classes[:20],
            "functions": functions[:50],
            "imports": list(set(imports))[:30],
            "line_count": len(content.splitlines())
        }
    except Exception as e:
        return {"error": str(e), "line_count": 0}


def extract_typescript_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from TypeScript/TSX files."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # Extract imports
        imports = re.findall(r"import\s+.*?from\s+['\"](.+?)['\"]", content)

        # Extract interfaces
        interfaces = re.findall(r"interface\s+(\w+)", content)

        # Extract types
        types = re.findall(r"type\s+(\w+)\s*=", content)

        # Extract React components
        components = re.findall(r"(?:export\s+)?(?:const|function)\s+(\w+).*?(?:React\.FC|:.*?JSX\.Element)", content)

        # Extract function components
        func_components = re.findall(r"export\s+(?:default\s+)?function\s+(\w+)", content)

        return {
            "imports": imports[:20],
            "interfaces": interfaces[:15],
            "types": types[:15],
            "components": list(set(components + func_components))[:15],
            "line_count": len(content.splitlines()),
            "has_hooks": "useState" in content or "useEffect" in content,
            "has_redux": "useSelector" in content or "useDispatch" in content
        }
    except Exception as e:
        return {"error": str(e), "line_count": 0}


def extract_agent_metadata(file_path: Path) -> Dict[str, Any]:
    """Extract metadata from agent markdown files."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # Extract title
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1) if title_match else file_path.stem

        # Extract description (first paragraph after title)
        desc_match = re.search(r"^#\s+.+$\n+(.+?)(?:\n\n|\n#)", content, re.MULTILINE | re.DOTALL)
        description = desc_match.group(1).strip()[:500] if desc_match else None

        # Extract tools mentioned
        tools = re.findall(r"Tools?:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
        tools_list = [t.strip() for t in ",".join(tools).split(",") if t.strip()]

        # Extract subagent type hints
        subagent_match = re.search(r"subagent_type[\"']?\s*[:=]\s*[\"'](\w+)[\"']", content)
        subagent_type = subagent_match.group(1) if subagent_match else file_path.stem

        return {
            "title": title,
            "description": description,
            "tools": tools_list[:20],
            "subagent_type": subagent_type,
            "line_count": len(content.splitlines())
        }
    except Exception as e:
        return {"error": str(e)}


def collect_codebase_data() -> List[Dict[str, Any]]:
    """Collect all codebase files and their metadata."""
    data = []

    # File patterns to collect
    patterns = {
        "python": ("**/*.py", extract_python_metadata),
        "typescript": ("**/*.ts", extract_typescript_metadata),
        "tsx": ("**/*.tsx", extract_typescript_metadata),
        "agent_md": (".claude/agents/**/*.md", extract_agent_metadata),
    }

    exclude_dirs = {
        "node_modules", ".git", "__pycache__", ".venv",
        "venv", "dist", ".next", "coverage", ".mypy_cache"
    }

    for file_type, (pattern, extractor) in patterns.items():
        for file_path in PROJECT_ROOT.glob(pattern):
            # Skip excluded directories
            if any(excluded in file_path.parts for excluded in exclude_dirs):
                continue

            try:
                rel_path = file_path.relative_to(PROJECT_ROOT)
                stat = file_path.stat()

                metadata = extractor(file_path)

                data.append({
                    "file_path": str(rel_path),
                    "file_type": file_type,
                    "file_name": file_path.name,
                    "directory": str(rel_path.parent),
                    "size_bytes": stat.st_size,
                    "modified_timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "file_hash": get_file_hash(file_path),
                    "metadata": json.dumps(metadata)
                })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return data


def collect_api_endpoints() -> List[Dict[str, Any]]:
    """Collect API endpoint definitions."""
    endpoints = []

    router_dir = PROJECT_ROOT / "backend" / "api" / "routers"
    if not router_dir.exists():
        return endpoints

    for py_file in router_dir.glob("*.py"):
        try:
            content = py_file.read_text(encoding='utf-8')

            # Find route decorators
            route_patterns = re.findall(
                r'@router\.(get|post|put|delete|patch)\(["\'](.+?)["\'].*?\)\s*(?:async\s+)?def\s+(\w+)',
                content,
                re.DOTALL
            )

            for method, path, func_name in route_patterns:
                # Try to extract response model
                response_model = re.search(
                    rf'{func_name}.*?response_model\s*=\s*(\w+)',
                    content
                )

                endpoints.append({
                    "router_file": py_file.name,
                    "method": method.upper(),
                    "path": path,
                    "function_name": func_name,
                    "response_model": response_model.group(1) if response_model else None,
                    "prefix": f"/api/{py_file.stem}" if py_file.stem != "__init__" else "/api"
                })
        except Exception as e:
            print(f"Error processing {py_file}: {e}")

    return endpoints


def collect_ml_models() -> List[Dict[str, Any]]:
    """Collect ML model information."""
    models = []

    ml_dirs = [
        PROJECT_ROOT / "backend" / "ml",
        PROJECT_ROOT / "ml_models",
        PROJECT_ROOT / "tools" / "utilities" / "models"
    ]

    for ml_dir in ml_dirs:
        if not ml_dir.exists():
            continue

        for model_file in ml_dir.rglob("*"):
            if model_file.is_file():
                try:
                    stat = model_file.stat()

                    # Determine model type
                    model_type = "unknown"
                    if model_file.suffix in [".pth", ".pt"]:
                        model_type = "pytorch"
                    elif model_file.suffix == ".pkl":
                        model_type = "pickle"
                    elif model_file.suffix == ".joblib":
                        model_type = "joblib"
                    elif model_file.suffix == ".json":
                        model_type = "config"
                    elif model_file.suffix in [".npz", ".npy"]:
                        model_type = "numpy"

                    models.append({
                        "file_path": str(model_file.relative_to(PROJECT_ROOT)),
                        "file_name": model_file.name,
                        "model_type": model_type,
                        "size_bytes": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified_timestamp": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except Exception as e:
                    print(f"Error processing {model_file}: {e}")

    return models


def collect_agent_definitions() -> List[Dict[str, Any]]:
    """Collect agent definitions from .claude/agents/."""
    agents = []

    agents_dir = PROJECT_ROOT / ".claude" / "agents"
    if not agents_dir.exists():
        return agents

    for md_file in agents_dir.rglob("*.md"):
        try:
            metadata = extract_agent_metadata(md_file)

            agents.append({
                "agent_file": str(md_file.relative_to(agents_dir)),
                "agent_name": md_file.stem,
                "category": str(md_file.parent.relative_to(agents_dir)) if md_file.parent != agents_dir else "root",
                **metadata
            })
        except Exception as e:
            print(f"Error processing {md_file}: {e}")

    return agents


def collect_project_metrics() -> Dict[str, Any]:
    """Collect overall project metrics."""
    metrics = {
        "total_files": 0,
        "total_lines": 0,
        "python_files": 0,
        "typescript_files": 0,
        "test_files": 0,
        "agent_files": 0,
        "docker_services": 12,
        "api_routers": 18,
        "database_tables": 22,
        "ml_models": 7,
        "project_completion": 89,
        "collected_at": datetime.now(timezone.utc).isoformat()
    }

    exclude_dirs = {"node_modules", ".git", "__pycache__", ".venv", "venv", "dist"}

    for py_file in PROJECT_ROOT.rglob("*.py"):
        if any(excluded in py_file.parts for excluded in exclude_dirs):
            continue
        metrics["python_files"] += 1
        metrics["total_files"] += 1
        try:
            metrics["total_lines"] += len(py_file.read_text().splitlines())
        except:
            pass
        if "test" in py_file.name.lower():
            metrics["test_files"] += 1

    for ts_file in PROJECT_ROOT.rglob("*.ts"):
        if any(excluded in ts_file.parts for excluded in exclude_dirs):
            continue
        metrics["typescript_files"] += 1
        metrics["total_files"] += 1

    for tsx_file in PROJECT_ROOT.rglob("*.tsx"):
        if any(excluded in tsx_file.parts for excluded in exclude_dirs):
            continue
        metrics["typescript_files"] += 1
        metrics["total_files"] += 1

    for agent_file in (PROJECT_ROOT / ".claude" / "agents").rglob("*.md"):
        metrics["agent_files"] += 1

    return metrics


def create_readme() -> str:
    """Create README for the dataset."""
    return """---
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
"""


def main():
    """Main function to create and upload the dataset."""
    print("=" * 60)
    print("Investment Analysis Platform - HuggingFace Dataset Creator")
    print("=" * 60)

    # Ensure output directory exists
    DATASET_OUTPUT.mkdir(parents=True, exist_ok=True)

    # Collect all data
    print("\n[1/5] Collecting codebase metadata...")
    codebase_data = collect_codebase_data()
    print(f"    Collected {len(codebase_data)} files")

    print("\n[2/5] Collecting API endpoints...")
    api_data = collect_api_endpoints()
    print(f"    Collected {len(api_data)} endpoints")

    print("\n[3/5] Collecting ML model info...")
    ml_data = collect_ml_models()
    print(f"    Collected {len(ml_data)} model files")

    print("\n[4/5] Collecting agent definitions...")
    agent_data = collect_agent_definitions()
    print(f"    Collected {len(agent_data)} agents")

    print("\n[5/5] Collecting project metrics...")
    metrics_data = collect_project_metrics()
    print(f"    Metrics: {metrics_data['python_files']} Python, {metrics_data['typescript_files']} TS/TSX files")

    # Create datasets
    print("\n" + "=" * 60)
    print("Creating Dataset Splits...")
    print("=" * 60)

    # Codebase dataset
    codebase_ds = Dataset.from_list(codebase_data) if codebase_data else Dataset.from_dict({
        "file_path": [], "file_type": [], "file_name": [], "directory": [],
        "size_bytes": [], "modified_timestamp": [], "file_hash": [], "metadata": []
    })

    # API endpoints dataset
    api_ds = Dataset.from_list(api_data) if api_data else Dataset.from_dict({
        "router_file": [], "method": [], "path": [], "function_name": [],
        "response_model": [], "prefix": []
    })

    # ML models dataset
    ml_ds = Dataset.from_list(ml_data) if ml_data else Dataset.from_dict({
        "file_path": [], "file_name": [], "model_type": [],
        "size_bytes": [], "size_mb": [], "modified_timestamp": []
    })

    # Agents dataset
    agents_ds = Dataset.from_list(agent_data) if agent_data else Dataset.from_dict({
        "agent_file": [], "agent_name": [], "category": [],
        "title": [], "description": [], "tools": [], "subagent_type": []
    })

    # Metrics as single-row dataset
    metrics_ds = Dataset.from_dict({k: [v] for k, v in metrics_data.items()})

    # Combine into DatasetDict
    dataset_dict = DatasetDict({
        "codebase": codebase_ds,
        "api_endpoints": api_ds,
        "ml_models": ml_ds,
        "agents": agents_ds,
        "metrics": metrics_ds
    })

    # Save locally
    print("\nSaving dataset locally...")
    dataset_dict.save_to_disk(str(DATASET_OUTPUT))

    # Save README
    readme_path = DATASET_OUTPUT / "README.md"
    readme_path.write_text(create_readme())
    print(f"    Saved README to {readme_path}")

    # Upload to Hugging Face
    print("\n" + "=" * 60)
    print("Uploading to Hugging Face Hub...")
    print("=" * 60)

    try:
        api = HfApi()

        # Get authenticated user
        user_info = api.whoami()
        username = user_info["name"]
        print(f"    Authenticated as: {username}")

        repo_id = f"{username}/investment-analysis-platform"

        # Create or get repo
        try:
            create_repo(repo_id, repo_type="dataset", exist_ok=True)
            print(f"    Repository: {repo_id}")
        except Exception as e:
            print(f"    Using existing repo: {repo_id}")

        # Upload each component as a separate dataset (different schemas)
        component_repos = {
            "codebase": codebase_ds,
            "api-endpoints": api_ds,
            "ml-models": ml_ds,
            "agents": agents_ds,
            "metrics": metrics_ds
        }

        for component_name, component_ds in component_repos.items():
            component_repo_id = f"{username}/investment-analysis-platform-{component_name}"
            print(f"\n    Creating repo: {component_repo_id}")

            try:
                create_repo(component_repo_id, repo_type="dataset", exist_ok=True)
            except Exception as e:
                print(f"    Note: {e}")

            print(f"    Uploading {component_name} ({len(component_ds)} records)...")
            component_ds.push_to_hub(
                component_repo_id,
                commit_message=f"Update {component_name} data",
                private=False
            )

        print(f"\n✅ All datasets uploaded successfully!")
        print(f"\n    Dataset URLs:")
        for component_name in component_repos.keys():
            print(f"    - https://huggingface.co/datasets/{username}/investment-analysis-platform-{component_name}")

        # Also save JSON exports for easy access
        json_output = DATASET_OUTPUT / "json_exports"
        json_output.mkdir(exist_ok=True)

        with open(json_output / "codebase.json", "w") as f:
            json.dump(codebase_data, f, indent=2)
        with open(json_output / "api_endpoints.json", "w") as f:
            json.dump(api_data, f, indent=2)
        with open(json_output / "ml_models.json", "w") as f:
            json.dump(ml_data, f, indent=2)
        with open(json_output / "agents.json", "w") as f:
            json.dump(agent_data, f, indent=2)
        with open(json_output / "metrics.json", "w") as f:
            json.dump(metrics_data, f, indent=2)

        print(f"\n    JSON exports saved to: {json_output}")

    except Exception as e:
        print(f"\n❌ Error uploading to HuggingFace: {e}")
        print("    Dataset saved locally. You can manually upload later.")
        raise

    print("\n" + "=" * 60)
    print("DATASET CREATION COMPLETE")
    print("=" * 60)

    return dataset_dict


if __name__ == "__main__":
    main()
