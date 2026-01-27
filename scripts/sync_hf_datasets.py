#!/usr/bin/env python3
"""
Hugging Face Dataset Sync for Investment Analysis Platform

Syncs project data bidirectionally with Hugging Face Hub:
- Updates HF datasets when codebase changes
- Downloads latest dataset versions for local use
- Integrates with Claude Flow memory for cross-session persistence

Usage:
    python scripts/sync_hf_datasets.py --push    # Upload local changes to HF
    python scripts/sync_hf_datasets.py --pull    # Download latest from HF
    python scripts/sync_hf_datasets.py --status  # Check sync status
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import HfApi, login, hf_hub_download, list_repo_files
from datasets import load_dataset, Dataset

# Configuration
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = "financeindustryknowledgeskills"
PROJECT_ROOT = Path(__file__).parent.parent
DATASETS_DIR = PROJECT_ROOT / "datasets" / "investment-analysis-platform"
SYNC_STATE_FILE = DATASETS_DIR / ".sync_state.json"

# Dataset repository mapping
DATASET_REPOS = {
    "codebase": f"{HF_USERNAME}/investment-analysis-platform-codebase",
    "api-endpoints": f"{HF_USERNAME}/investment-analysis-platform-api-endpoints",
    "ml-models": f"{HF_USERNAME}/investment-analysis-platform-ml-models",
    "agents": f"{HF_USERNAME}/investment-analysis-platform-agents",
    "metrics": f"{HF_USERNAME}/investment-analysis-platform-metrics",
}


def get_api() -> HfApi:
    """Get authenticated HF API client."""
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
    return HfApi()


def load_sync_state() -> Dict[str, Any]:
    """Load sync state from file."""
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {"last_push": None, "last_pull": None, "datasets": {}}


def save_sync_state(state: Dict[str, Any]):
    """Save sync state to file."""
    SYNC_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


def push_datasets():
    """Push local datasets to Hugging Face Hub."""
    print("=" * 60)
    print("PUSHING DATASETS TO HUGGING FACE HUB")
    print("=" * 60)

    # Run the main dataset creation script
    from create_hf_dataset import main as create_datasets
    create_datasets()

    # Update sync state
    state = load_sync_state()
    state["last_push"] = datetime.now(timezone.utc).isoformat()
    for name, repo_id in DATASET_REPOS.items():
        state["datasets"][name] = {
            "repo_id": repo_id,
            "last_pushed": state["last_push"]
        }
    save_sync_state(state)

    print(f"\n✅ Push complete at {state['last_push']}")


def pull_datasets():
    """Pull latest datasets from Hugging Face Hub."""
    print("=" * 60)
    print("PULLING DATASETS FROM HUGGING FACE HUB")
    print("=" * 60)

    api = get_api()
    state = load_sync_state()
    json_exports = DATASETS_DIR / "json_exports"
    json_exports.mkdir(parents=True, exist_ok=True)

    for name, repo_id in DATASET_REPOS.items():
        print(f"\n[{name}] Loading from {repo_id}...")
        try:
            # Load dataset from HF
            ds = load_dataset(repo_id, split="train")
            print(f"    Loaded {len(ds)} records")

            # Save as JSON locally
            records = [dict(row) for row in ds]
            json_path = json_exports / f"{name.replace('-', '_')}.json"
            json_path.write_text(json.dumps(records, indent=2, default=str))
            print(f"    Saved to {json_path}")

            # Update state
            state["datasets"][name] = {
                "repo_id": repo_id,
                "last_pulled": datetime.now(timezone.utc).isoformat(),
                "record_count": len(ds)
            }

        except Exception as e:
            print(f"    ❌ Error: {e}")

    state["last_pull"] = datetime.now(timezone.utc).isoformat()
    save_sync_state(state)

    print(f"\n✅ Pull complete at {state['last_pull']}")


def check_status():
    """Check sync status of all datasets."""
    print("=" * 60)
    print("HUGGING FACE DATASET SYNC STATUS")
    print("=" * 60)

    api = get_api()
    state = load_sync_state()

    print(f"\nLast Push: {state.get('last_push', 'Never')}")
    print(f"Last Pull: {state.get('last_pull', 'Never')}")

    print("\nDatasets:")
    print("-" * 60)

    for name, repo_id in DATASET_REPOS.items():
        print(f"\n📦 {name}")
        print(f"   Repo: {repo_id}")

        local_state = state.get("datasets", {}).get(name, {})
        print(f"   Last Pushed: {local_state.get('last_pushed', 'Never')}")
        print(f"   Last Pulled: {local_state.get('last_pulled', 'Never')}")

        # Check HF for latest info
        try:
            files = list_repo_files(repo_id, repo_type="dataset")
            parquet_files = [f for f in files if f.endswith(".parquet")]
            print(f"   HF Files: {len(parquet_files)} parquet files")
            print(f"   URL: https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"   ❌ Error checking HF: {e}")

    # Check local files
    print("\n" + "-" * 60)
    print("Local Files:")

    json_exports = DATASETS_DIR / "json_exports"
    if json_exports.exists():
        for json_file in json_exports.glob("*.json"):
            size = json_file.stat().st_size
            print(f"   {json_file.name}: {size:,} bytes")
    else:
        print("   No local JSON exports found")


def integrate_with_claude_flow():
    """Store dataset references in Claude Flow memory."""
    print("\n" + "=" * 60)
    print("INTEGRATING WITH CLAUDE FLOW MEMORY")
    print("=" * 60)

    try:
        import subprocess

        # Store each dataset reference
        for name, repo_id in DATASET_REPOS.items():
            key = f"hf-dataset-{name}"
            value = json.dumps({
                "repo_id": repo_id,
                "url": f"https://huggingface.co/datasets/{repo_id}",
                "synced_at": datetime.now(timezone.utc).isoformat()
            })

            result = subprocess.run(
                ["npx", "@claude-flow/cli@latest", "memory", "store",
                 "--key", key, "--value", value, "--namespace", "hf-datasets"],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0:
                print(f"   ✅ Stored {key} in Claude Flow memory")
            else:
                print(f"   ⚠️ Could not store {key}: {result.stderr}")

        print("\n✅ Claude Flow integration complete")

    except Exception as e:
        print(f"\n⚠️ Claude Flow integration skipped: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync Investment Analysis Platform datasets with Hugging Face Hub"
    )
    parser.add_argument("--push", action="store_true", help="Push local data to HF Hub")
    parser.add_argument("--pull", action="store_true", help="Pull latest from HF Hub")
    parser.add_argument("--status", action="store_true", help="Check sync status")
    parser.add_argument("--integrate", action="store_true", help="Integrate with Claude Flow")

    args = parser.parse_args()

    if not any([args.push, args.pull, args.status, args.integrate]):
        args.status = True  # Default to status check

    if args.push:
        push_datasets()
        integrate_with_claude_flow()

    if args.pull:
        pull_datasets()

    if args.status:
        check_status()

    if args.integrate:
        integrate_with_claude_flow()


if __name__ == "__main__":
    main()
