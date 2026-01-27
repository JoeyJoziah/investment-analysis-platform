#!/usr/bin/env python3
"""
Unified Sync Coordinator for Investment Analysis Platform

Provides bidirectional synchronization between:
- GitHub Projects (via gh CLI)
- GitHub Issues
- Notion Database
- Local TODO.md

Features:
- Bidirectional sync with conflict resolution
- State tracking with timestamps
- Field mapping between systems
- Configurable sync direction and conflict resolution

Usage:
    python scripts/unified_sync.py sync --bidirectional
    python scripts/unified_sync.py sync --github-only
    python scripts/unified_sync.py sync --notion-only
    python scripts/unified_sync.py status
    python scripts/unified_sync.py diff

Configuration:
    Sync state stored in: .sync_state.json
    Config file: .github/board-sync.yml
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

# Try to import requests for Notion API
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not installed. Notion sync disabled.")

# Try to import yaml for config parsing
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
SYNC_STATE_FILE = PROJECT_ROOT / ".sync_state.json"
CONFIG_FILE = PROJECT_ROOT / ".github" / "board-sync.yml"
TODO_FILE = PROJECT_ROOT / "TODO.md"
NOTION_API_KEY_PATH = Path.home() / ".config" / "notion" / "api_key"

# Default Notion configuration
NOTION_VERSION = "2025-09-03"
NOTION_DATABASE_ID = "2f3b9fc9-1d9d-8026-9719-f82b0e311f50"
NOTION_DATA_SOURCE_ID = "2f3b9fc9-1d9d-80cc-b349-000b8158ee99"


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    LOCAL_WINS = "local_wins"  # Local repo data wins (same as github_wins)
    GITHUB_WINS = "github_wins"  # Alias for LOCAL_WINS (backward compatibility)
    NOTION_WINS = "notion_wins"
    NEWEST_WINS = "newest_wins"
    MANUAL = "manual"


class ItemStatus(Enum):
    """Unified status across systems."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    BLOCKED = "blocked"
    DONE = "done"


class ItemPriority(Enum):
    """Unified priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class SyncItem:
    """Represents a syncable item across systems."""
    id: str  # Unique identifier (hash of title)
    title: str
    status: ItemStatus = ItemStatus.TODO
    priority: ItemPriority = ItemPriority.MEDIUM
    category: str = ""
    milestone: str = ""
    assignees: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    notes: str = ""

    # Source tracking
    github_issue_number: Optional[int] = None
    github_project_item_id: Optional[str] = None
    notion_page_id: Optional[str] = None
    todo_line_number: Optional[int] = None

    # Timestamps for conflict resolution
    github_updated_at: Optional[str] = None
    notion_updated_at: Optional[str] = None
    local_updated_at: Optional[str] = None

    # Hash for change detection
    content_hash: str = ""

    def __post_init__(self):
        if not self.id:
            self.id = self._generate_id()
        if not self.content_hash:
            self.content_hash = self._compute_hash()

    def _generate_id(self) -> str:
        """Generate unique ID from title."""
        return hashlib.md5(self.title.lower().strip().encode()).hexdigest()[:12]

    def _compute_hash(self) -> str:
        """Compute content hash for change detection."""
        content = f"{self.title}|{self.status.value}|{self.priority.value}|{self.category}|{self.notes}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def has_changed_from(self, other: "SyncItem") -> bool:
        """Check if this item has changed compared to another."""
        return self._compute_hash() != other.content_hash

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["status"] = self.status.value
        data["priority"] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SyncItem":
        """Create from dictionary."""
        data = data.copy()
        data["status"] = ItemStatus(data.get("status", "todo"))
        data["priority"] = ItemPriority(data.get("priority", "medium"))
        return cls(**data)


@dataclass
class SyncState:
    """Tracks synchronization state between systems."""
    last_sync: Optional[str] = None
    last_github_sync: Optional[str] = None
    last_notion_sync: Optional[str] = None
    items: dict = field(default_factory=dict)  # id -> SyncItem dict
    conflicts: list = field(default_factory=list)
    sync_history: list = field(default_factory=list)

    def save(self, path: Path = SYNC_STATE_FILE):
        """Save state to file."""
        data = {
            "last_sync": self.last_sync,
            "last_github_sync": self.last_github_sync,
            "last_notion_sync": self.last_notion_sync,
            "items": self.items,
            "conflicts": self.conflicts,
            "sync_history": self.sync_history[-50:],  # Keep last 50 entries
        }
        path.write_text(json.dumps(data, indent=2, default=str))

    @classmethod
    def load(cls, path: Path = SYNC_STATE_FILE) -> "SyncState":
        """Load state from file."""
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
            return cls(
                last_sync=data.get("last_sync"),
                last_github_sync=data.get("last_github_sync"),
                last_notion_sync=data.get("last_notion_sync"),
                items=data.get("items", {}),
                conflicts=data.get("conflicts", []),
                sync_history=data.get("sync_history", []),
            )
        except (json.JSONDecodeError, KeyError):
            return cls()

    def add_sync_event(self, event_type: str, details: str):
        """Add a sync event to history."""
        self.sync_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": event_type,
            "details": details,
        })


class GitHubClient:
    """Client for GitHub Projects and Issues via gh CLI."""

    def __init__(self, owner: str = None, repo: str = None):
        self.owner = owner or self._get_repo_owner()
        self.repo = repo or self._get_repo_name()

    def _get_repo_owner(self) -> str:
        """Get repository owner from git remote."""
        try:
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "owner", "-q", ".owner.login"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return os.environ.get("GITHUB_REPOSITORY_OWNER", "")

    def _get_repo_name(self) -> str:
        """Get repository name from git remote."""
        try:
            result = subprocess.run(
                ["gh", "repo", "view", "--json", "name", "-q", ".name"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return os.environ.get("GITHUB_REPOSITORY_NAME", "investment-analysis-platform")

    def check_auth(self) -> bool:
        """Check if gh CLI is authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_issues(self, state: str = "all", limit: int = 200) -> list[dict]:
        """Fetch issues from repository."""
        try:
            result = subprocess.run(
                [
                    "gh", "issue", "list",
                    "--repo", f"{self.owner}/{self.repo}",
                    "--state", state,
                    "--limit", str(limit),
                    "--json", "number,title,state,labels,assignees,milestone,updatedAt,body"
                ],
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout) if result.stdout else []
        except subprocess.CalledProcessError as e:
            print(f"Error fetching issues: {e.stderr}")
            return []

    def get_issue(self, number: int) -> Optional[dict]:
        """Fetch single issue by number."""
        try:
            result = subprocess.run(
                [
                    "gh", "issue", "view", str(number),
                    "--repo", f"{self.owner}/{self.repo}",
                    "--json", "number,title,state,labels,assignees,milestone,updatedAt,body"
                ],
                capture_output=True, text=True, check=True
            )
            return json.loads(result.stdout) if result.stdout else None
        except subprocess.CalledProcessError:
            return None

    def create_issue(self, title: str, body: str = "", labels: list = None) -> Optional[int]:
        """Create a new issue."""
        cmd = [
            "gh", "issue", "create",
            "--repo", f"{self.owner}/{self.repo}",
            "--title", title,
            "--body", body or f"Auto-created by unified sync\n\nCreated: {datetime.now().isoformat()}"
        ]
        if labels:
            cmd.extend(["--label", ",".join(labels)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            # Parse issue number from URL
            url = result.stdout.strip()
            match = re.search(r'/issues/(\d+)', url)
            return int(match.group(1)) if match else None
        except subprocess.CalledProcessError as e:
            print(f"Error creating issue: {e.stderr}")
            return None

    def update_issue(self, number: int, title: str = None, body: str = None,
                     state: str = None, labels: list = None) -> bool:
        """Update an existing issue."""
        cmd = [
            "gh", "issue", "edit", str(number),
            "--repo", f"{self.owner}/{self.repo}"
        ]
        if title:
            cmd.extend(["--title", title])
        if body:
            cmd.extend(["--body", body])
        if labels:
            cmd.extend(["--add-label", ",".join(labels)])

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"Error updating issue {number}: {e.stderr}")
            return False

    def close_issue(self, number: int) -> bool:
        """Close an issue."""
        try:
            subprocess.run(
                ["gh", "issue", "close", str(number), "--repo", f"{self.owner}/{self.repo}"],
                capture_output=True, text=True, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def reopen_issue(self, number: int) -> bool:
        """Reopen an issue."""
        try:
            subprocess.run(
                ["gh", "issue", "reopen", str(number), "--repo", f"{self.owner}/{self.repo}"],
                capture_output=True, text=True, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False

    def get_project_id(self, project_name: str = "Investment Analysis Platform") -> Optional[str]:
        """Get project ID by name."""
        try:
            result = subprocess.run(
                ["gh", "project", "list", "--owner", self.owner, "--format", "json"],
                capture_output=True, text=True, check=True
            )
            projects = json.loads(result.stdout) if result.stdout else {"projects": []}
            for project in projects.get("projects", []):
                if project.get("title") == project_name:
                    return project.get("id")
            return None
        except subprocess.CalledProcessError:
            return None

    def get_project_items(self, project_id: str) -> list[dict]:
        """Get all items from a project."""
        try:
            result = subprocess.run(
                ["gh", "project", "item-list", project_id, "--owner", self.owner, "--format", "json"],
                capture_output=True, text=True, check=True
            )
            data = json.loads(result.stdout) if result.stdout else {"items": []}
            return data.get("items", [])
        except subprocess.CalledProcessError:
            return []

    def add_issue_to_project(self, project_id: str, issue_url: str) -> bool:
        """Add an issue to a project."""
        try:
            subprocess.run(
                ["gh", "project", "item-add", project_id, "--owner", self.owner, "--url", issue_url],
                capture_output=True, text=True, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False


class NotionClient:
    """Client for Notion API."""

    def __init__(self):
        self.api_key = self._load_api_key()
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json"
        }
        self.enabled = bool(self.api_key and HAS_REQUESTS)

    def _load_api_key(self) -> Optional[str]:
        """Load API key from config file."""
        if not NOTION_API_KEY_PATH.exists():
            return None
        return NOTION_API_KEY_PATH.read_text().strip()

    def query_database(self, database_id: str = NOTION_DATABASE_ID) -> list[dict]:
        """Query all pages from the database."""
        if not self.enabled:
            return []

        url = f"{self.base_url}/data_sources/{NOTION_DATA_SOURCE_ID}/query"
        all_results = []
        has_more = True
        start_cursor = None

        while has_more:
            payload = {"page_size": 100}
            if start_cursor:
                payload["start_cursor"] = start_cursor

            try:
                response = requests.post(url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                all_results.extend(data.get("results", []))
                has_more = data.get("has_more", False)
                start_cursor = data.get("next_cursor")
            except requests.exceptions.RequestException as e:
                print(f"Error querying Notion: {e}")
                break

        return all_results

    def create_page(self, properties: dict, database_id: str = NOTION_DATABASE_ID) -> Optional[dict]:
        """Create a new page in the database."""
        if not self.enabled:
            return None

        url = f"{self.base_url}/pages"
        payload = {
            "parent": {"database_id": database_id},
            "properties": properties
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error creating Notion page: {e}")
            return None

    def update_page(self, page_id: str, properties: dict) -> Optional[dict]:
        """Update an existing page."""
        if not self.enabled:
            return None

        url = f"{self.base_url}/pages/{page_id}"
        payload = {"properties": properties}

        try:
            response = requests.patch(url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error updating Notion page: {e}")
            return None

    def get_page(self, page_id: str) -> Optional[dict]:
        """Get a single page by ID."""
        if not self.enabled:
            return None

        url = f"{self.base_url}/pages/{page_id}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException:
            return None


class FieldMapper:
    """Maps fields between GitHub, Notion, and local formats."""

    # Status mappings
    GITHUB_TO_STATUS = {
        "open": ItemStatus.TODO,
        "closed": ItemStatus.DONE,
    }

    NOTION_TO_STATUS = {
        "to do": ItemStatus.TODO,
        "not started": ItemStatus.TODO,
        "in progress": ItemStatus.IN_PROGRESS,
        "in review": ItemStatus.IN_REVIEW,
        "blocked": ItemStatus.BLOCKED,
        "done": ItemStatus.DONE,
        "completed": ItemStatus.DONE,
    }

    STATUS_TO_NOTION = {
        ItemStatus.TODO: "To Do",
        ItemStatus.IN_PROGRESS: "In Progress",
        ItemStatus.IN_REVIEW: "In Review",
        ItemStatus.BLOCKED: "Blocked",
        ItemStatus.DONE: "Done",
    }

    # Priority mappings
    LABEL_TO_PRIORITY = {
        "priority:critical": ItemPriority.CRITICAL,
        "priority:high": ItemPriority.HIGH,
        "priority:medium": ItemPriority.MEDIUM,
        "priority:low": ItemPriority.LOW,
        "critical": ItemPriority.CRITICAL,
        "high": ItemPriority.HIGH,
        "medium": ItemPriority.MEDIUM,
        "low": ItemPriority.LOW,
    }

    NOTION_TO_PRIORITY = {
        "critical": ItemPriority.CRITICAL,
        "high": ItemPriority.HIGH,
        "medium": ItemPriority.MEDIUM,
        "low": ItemPriority.LOW,
    }

    PRIORITY_TO_LABEL = {
        ItemPriority.CRITICAL: "priority:critical",
        ItemPriority.HIGH: "priority:high",
        ItemPriority.MEDIUM: "priority:medium",
        ItemPriority.LOW: "priority:low",
    }

    PRIORITY_TO_NOTION = {
        ItemPriority.CRITICAL: "Critical",
        ItemPriority.HIGH: "High",
        ItemPriority.MEDIUM: "Medium",
        ItemPriority.LOW: "Low",
    }

    # Category mappings
    LABEL_TO_CATEGORY = {
        "backend": "Backend",
        "frontend": "Frontend",
        "infrastructure": "Infrastructure",
        "ml-pipeline": "Data/ML",
        "security": "Security",
        "documentation": "Documentation",
        "testing": "Testing",
        "enhancement": "Feature Development",
        "bug": "Bug Fix",
    }

    @classmethod
    def github_issue_to_sync_item(cls, issue: dict) -> SyncItem:
        """Convert GitHub issue to SyncItem."""
        labels = [l.get("name", "") for l in issue.get("labels", [])]

        # Determine priority from labels
        priority = ItemPriority.MEDIUM
        for label in labels:
            label_lower = label.lower()
            if label_lower in cls.LABEL_TO_PRIORITY:
                priority = cls.LABEL_TO_PRIORITY[label_lower]
                break

        # Determine status
        state = issue.get("state", "open").lower()
        status = cls.GITHUB_TO_STATUS.get(state, ItemStatus.TODO)

        # Determine category
        category = "Feature Development"
        for label in labels:
            label_lower = label.lower()
            if label_lower in cls.LABEL_TO_CATEGORY:
                category = cls.LABEL_TO_CATEGORY[label_lower]
                break

        # Get assignees
        assignees = [a.get("login", "") for a in issue.get("assignees", [])]

        # Get milestone
        milestone = ""
        if issue.get("milestone"):
            milestone = issue["milestone"].get("title", "")

        return SyncItem(
            id="",  # Will be generated
            title=issue.get("title", ""),
            status=status,
            priority=priority,
            category=category,
            milestone=milestone,
            assignees=assignees,
            labels=labels,
            notes=issue.get("body", "") or "",
            github_issue_number=issue.get("number"),
            github_updated_at=issue.get("updatedAt"),
        )

    @classmethod
    def notion_page_to_sync_item(cls, page: dict) -> SyncItem:
        """Convert Notion page to SyncItem."""
        props = page.get("properties", {})

        # Extract title
        title_prop = props.get("Task", {}).get("title", [])
        title = title_prop[0]["text"]["content"] if title_prop else "Untitled"

        # Extract status
        status_prop = props.get("Status", {}).get("status", {})
        status_name = (status_prop.get("name", "To Do") if status_prop else "To Do").lower()
        status = cls.NOTION_TO_STATUS.get(status_name, ItemStatus.TODO)

        # Extract priority
        priority_prop = props.get("Priority", {}).get("select", {})
        priority_name = (priority_prop.get("name", "Medium") if priority_prop else "Medium").lower()
        priority = cls.NOTION_TO_PRIORITY.get(priority_name, ItemPriority.MEDIUM)

        # Extract category
        category_prop = props.get("Category", {}).get("select", {})
        category = category_prop.get("name", "Feature Development") if category_prop else "Feature Development"

        # Extract milestone
        milestone_prop = props.get("Launch Milestone", {}).get("select", {})
        milestone = milestone_prop.get("name", "") if milestone_prop else ""

        # Extract notes
        notes_prop = props.get("Notes", {}).get("rich_text", [])
        notes = notes_prop[0]["text"]["content"] if notes_prop else ""

        return SyncItem(
            id="",  # Will be generated
            title=title,
            status=status,
            priority=priority,
            category=category,
            milestone=milestone,
            notes=notes,
            notion_page_id=page.get("id"),
            notion_updated_at=page.get("last_edited_time"),
        )

    @classmethod
    def sync_item_to_notion_properties(cls, item: SyncItem) -> dict:
        """Convert SyncItem to Notion page properties."""
        props = {
            "Task": {"title": [{"text": {"content": item.title}}]},
            "Status": {"status": {"name": cls.STATUS_TO_NOTION.get(item.status, "To Do")}},
            "Priority": {"select": {"name": cls.PRIORITY_TO_NOTION.get(item.priority, "Medium")}},
        }

        if item.category:
            props["Category"] = {"select": {"name": item.category}}

        if item.milestone:
            props["Launch Milestone"] = {"select": {"name": item.milestone}}

        if item.notes:
            props["Notes"] = {"rich_text": [{"text": {"content": item.notes[:2000]}}]}

        return props

    @classmethod
    def sync_item_to_github_labels(cls, item: SyncItem) -> list[str]:
        """Convert SyncItem to GitHub labels."""
        labels = []

        # Priority label
        priority_label = cls.PRIORITY_TO_LABEL.get(item.priority)
        if priority_label:
            labels.append(priority_label)

        # Category label
        for label, cat in cls.LABEL_TO_CATEGORY.items():
            if cat.lower() == item.category.lower():
                labels.append(label)
                break

        # Preserve existing labels
        for label in item.labels:
            if label not in labels:
                labels.append(label)

        return labels


class UnifiedSyncCoordinator:
    """Coordinates bidirectional sync between GitHub and Notion."""

    def __init__(self, conflict_resolution: ConflictResolution = ConflictResolution.LOCAL_WINS):
        self.github = GitHubClient()
        self.notion = NotionClient()
        self.mapper = FieldMapper()
        self.state = SyncState.load()
        self.conflict_resolution = conflict_resolution

        # Load config if available
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load sync configuration from YAML file."""
        if not CONFIG_FILE.exists():
            return {}

        if not HAS_YAML:
            return {}

        try:
            return yaml.safe_load(CONFIG_FILE.read_text()) or {}
        except Exception:
            return {}

    def fetch_github_items(self) -> dict[str, SyncItem]:
        """Fetch all items from GitHub (issues)."""
        print("Fetching items from GitHub...")
        items = {}

        issues = self.github.get_issues(state="all")
        for issue in issues:
            sync_item = self.mapper.github_issue_to_sync_item(issue)
            items[sync_item.id] = sync_item

        print(f"  Found {len(items)} GitHub issues")
        return items

    def fetch_notion_items(self) -> dict[str, SyncItem]:
        """Fetch all items from Notion."""
        if not self.notion.enabled:
            print("Notion sync disabled (no API key or requests library)")
            return {}

        print("Fetching items from Notion...")
        items = {}

        pages = self.notion.query_database()
        for page in pages:
            sync_item = self.mapper.notion_page_to_sync_item(page)
            items[sync_item.id] = sync_item

        print(f"  Found {len(items)} Notion pages")
        return items

    def detect_changes(self,
                       github_items: dict[str, SyncItem],
                       notion_items: dict[str, SyncItem]) -> dict:
        """Detect changes between systems."""
        changes = {
            "new_in_github": [],
            "new_in_notion": [],
            "updated_in_github": [],
            "updated_in_notion": [],
            "conflicts": [],
            "in_sync": [],
        }

        all_ids = set(github_items.keys()) | set(notion_items.keys())

        for item_id in all_ids:
            github_item = github_items.get(item_id)
            notion_item = notion_items.get(item_id)
            stored_item_data = self.state.items.get(item_id)

            if github_item and not notion_item:
                # Only in GitHub
                if not stored_item_data:
                    changes["new_in_github"].append(github_item)
                else:
                    # Might have been deleted from Notion
                    changes["new_in_github"].append(github_item)

            elif notion_item and not github_item:
                # Only in Notion
                if not stored_item_data:
                    changes["new_in_notion"].append(notion_item)
                else:
                    # Might have been deleted from GitHub
                    changes["new_in_notion"].append(notion_item)

            elif github_item and notion_item:
                # In both - check for changes
                stored = SyncItem.from_dict(stored_item_data) if stored_item_data else None

                github_changed = stored and github_item.has_changed_from(stored)
                notion_changed = stored and notion_item.has_changed_from(stored)

                if github_changed and notion_changed:
                    # Conflict - both changed
                    changes["conflicts"].append({
                        "github": github_item,
                        "notion": notion_item,
                        "stored": stored,
                    })
                elif github_changed:
                    changes["updated_in_github"].append(github_item)
                elif notion_changed:
                    changes["updated_in_notion"].append(notion_item)
                else:
                    changes["in_sync"].append(github_item)

        return changes

    def resolve_conflict(self, conflict: dict) -> SyncItem:
        """Resolve a conflict based on resolution strategy."""
        github_item = conflict["github"]
        notion_item = conflict["notion"]

        if self.conflict_resolution in (ConflictResolution.LOCAL_WINS, ConflictResolution.GITHUB_WINS):
            return github_item
        elif self.conflict_resolution == ConflictResolution.NOTION_WINS:
            return notion_item
        elif self.conflict_resolution == ConflictResolution.NEWEST_WINS:
            github_time = github_item.github_updated_at or ""
            notion_time = notion_item.notion_updated_at or ""
            if github_time > notion_time:
                return github_item
            else:
                return notion_item
        else:  # MANUAL
            print(f"\nConflict detected for: {github_item.title}")
            print(f"  Local:   status={github_item.status.value}, priority={github_item.priority.value}")
            print(f"  Notion:  status={notion_item.status.value}, priority={notion_item.priority.value}")
            print(f"  Using: local_wins (default)")
            return github_item

    def sync_to_github(self, item: SyncItem, create: bool = False) -> bool:
        """Sync an item to GitHub."""
        if create:
            # Create new issue
            labels = self.mapper.sync_item_to_github_labels(item)
            issue_num = self.github.create_issue(
                title=item.title,
                body=item.notes,
                labels=labels
            )
            if issue_num:
                item.github_issue_number = issue_num
                print(f"  Created GitHub issue #{issue_num}: {item.title}")
                return True
            return False
        else:
            # Update existing issue
            if not item.github_issue_number:
                return False

            labels = self.mapper.sync_item_to_github_labels(item)
            success = self.github.update_issue(
                number=item.github_issue_number,
                title=item.title,
                body=item.notes,
                labels=labels
            )

            # Handle status changes
            if item.status == ItemStatus.DONE:
                self.github.close_issue(item.github_issue_number)
            elif item.github_issue_number:
                # Reopen if was closed but now not done
                issue = self.github.get_issue(item.github_issue_number)
                if issue and issue.get("state") == "closed" and item.status != ItemStatus.DONE:
                    self.github.reopen_issue(item.github_issue_number)

            if success:
                print(f"  Updated GitHub issue #{item.github_issue_number}: {item.title}")
            return success

    def sync_to_notion(self, item: SyncItem, create: bool = False) -> bool:
        """Sync an item to Notion."""
        if not self.notion.enabled:
            return False

        properties = self.mapper.sync_item_to_notion_properties(item)

        if create:
            # Create new page
            result = self.notion.create_page(properties)
            if result:
                item.notion_page_id = result.get("id")
                print(f"  Created Notion page: {item.title}")
                return True
            return False
        else:
            # Update existing page
            if not item.notion_page_id:
                return False

            result = self.notion.update_page(item.notion_page_id, properties)
            if result:
                print(f"  Updated Notion page: {item.title}")
                return True
            return False

    def sync_bidirectional(self, dry_run: bool = False) -> dict:
        """Perform bidirectional sync between GitHub and Notion."""
        print("\n=== Unified Bidirectional Sync ===\n")

        # Fetch from both systems
        github_items = self.fetch_github_items()
        notion_items = self.fetch_notion_items()

        # Detect changes
        changes = self.detect_changes(github_items, notion_items)

        stats = {
            "github_created": 0,
            "github_updated": 0,
            "notion_created": 0,
            "notion_updated": 0,
            "conflicts_resolved": 0,
            "skipped": 0,
        }

        # Process new items from GitHub -> Notion
        for item in changes["new_in_github"]:
            if dry_run:
                print(f"  [DRY RUN] Would create in Notion: {item.title}")
                stats["notion_created"] += 1
            else:
                if self.sync_to_notion(item, create=True):
                    stats["notion_created"] += 1
                else:
                    stats["skipped"] += 1

        # Process new items from Notion -> GitHub
        for item in changes["new_in_notion"]:
            if dry_run:
                print(f"  [DRY RUN] Would create in GitHub: {item.title}")
                stats["github_created"] += 1
            else:
                if self.sync_to_github(item, create=True):
                    stats["github_created"] += 1
                else:
                    stats["skipped"] += 1

        # Process updates from GitHub -> Notion
        for item in changes["updated_in_github"]:
            if dry_run:
                print(f"  [DRY RUN] Would update in Notion: {item.title}")
                stats["notion_updated"] += 1
            else:
                # Get the notion page ID from existing item
                notion_item = notion_items.get(item.id)
                if notion_item:
                    item.notion_page_id = notion_item.notion_page_id
                if self.sync_to_notion(item, create=False):
                    stats["notion_updated"] += 1
                else:
                    stats["skipped"] += 1

        # Process updates from Notion -> GitHub
        for item in changes["updated_in_notion"]:
            if dry_run:
                print(f"  [DRY RUN] Would update in GitHub: {item.title}")
                stats["github_updated"] += 1
            else:
                # Get the GitHub issue number from existing item
                github_item = github_items.get(item.id)
                if github_item:
                    item.github_issue_number = github_item.github_issue_number
                if self.sync_to_github(item, create=False):
                    stats["github_updated"] += 1
                else:
                    stats["skipped"] += 1

        # Resolve conflicts
        for conflict in changes["conflicts"]:
            resolved = self.resolve_conflict(conflict)
            if dry_run:
                print(f"  [DRY RUN] Would resolve conflict: {resolved.title}")
                stats["conflicts_resolved"] += 1
            else:
                # Update both systems with resolved item
                github_item = conflict["github"]
                notion_item = conflict["notion"]

                resolved.github_issue_number = github_item.github_issue_number
                resolved.notion_page_id = notion_item.notion_page_id

                self.sync_to_github(resolved)
                self.sync_to_notion(resolved)
                stats["conflicts_resolved"] += 1

        # Update state
        if not dry_run:
            all_items = {**github_items, **notion_items}
            for item_id, item in all_items.items():
                self.state.items[item_id] = item.to_dict()

            now = datetime.now(timezone.utc).isoformat()
            self.state.last_sync = now
            self.state.last_github_sync = now
            self.state.last_notion_sync = now
            self.state.add_sync_event("bidirectional", f"Synced {len(all_items)} items")
            self.state.save()

        print(f"\n=== Sync Complete ===")
        print(f"  GitHub created: {stats['github_created']}")
        print(f"  GitHub updated: {stats['github_updated']}")
        print(f"  Notion created: {stats['notion_created']}")
        print(f"  Notion updated: {stats['notion_updated']}")
        print(f"  Conflicts resolved: {stats['conflicts_resolved']}")
        print(f"  Skipped: {stats['skipped']}")
        print(f"  In sync: {len(changes['in_sync'])}")

        return stats

    def sync_github_only(self, dry_run: bool = False) -> dict:
        """Sync from Notion to GitHub only."""
        print("\n=== Sync: Notion -> GitHub ===\n")

        github_items = self.fetch_github_items()
        notion_items = self.fetch_notion_items()

        stats = {"created": 0, "updated": 0, "skipped": 0}

        for item_id, notion_item in notion_items.items():
            if item_id in github_items:
                # Update
                notion_item.github_issue_number = github_items[item_id].github_issue_number
                if dry_run:
                    print(f"  [DRY RUN] Would update: {notion_item.title}")
                    stats["updated"] += 1
                elif self.sync_to_github(notion_item):
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # Create
                if dry_run:
                    print(f"  [DRY RUN] Would create: {notion_item.title}")
                    stats["created"] += 1
                elif self.sync_to_github(notion_item, create=True):
                    stats["created"] += 1
                else:
                    stats["skipped"] += 1

        if not dry_run:
            self.state.last_github_sync = datetime.now(timezone.utc).isoformat()
            self.state.add_sync_event("github_only", f"Created {stats['created']}, updated {stats['updated']}")
            self.state.save()

        print(f"\nSummary: Created {stats['created']}, Updated {stats['updated']}, Skipped {stats['skipped']}")
        return stats

    def sync_notion_only(self, dry_run: bool = False) -> dict:
        """Sync from GitHub to Notion only."""
        print("\n=== Sync: GitHub -> Notion ===\n")

        github_items = self.fetch_github_items()
        notion_items = self.fetch_notion_items()

        stats = {"created": 0, "updated": 0, "skipped": 0}

        for item_id, github_item in github_items.items():
            if item_id in notion_items:
                # Update
                github_item.notion_page_id = notion_items[item_id].notion_page_id
                if dry_run:
                    print(f"  [DRY RUN] Would update: {github_item.title}")
                    stats["updated"] += 1
                elif self.sync_to_notion(github_item):
                    stats["updated"] += 1
                else:
                    stats["skipped"] += 1
            else:
                # Create
                if dry_run:
                    print(f"  [DRY RUN] Would create: {github_item.title}")
                    stats["created"] += 1
                elif self.sync_to_notion(github_item, create=True):
                    stats["created"] += 1
                else:
                    stats["skipped"] += 1

        if not dry_run:
            self.state.last_notion_sync = datetime.now(timezone.utc).isoformat()
            self.state.add_sync_event("notion_only", f"Created {stats['created']}, updated {stats['updated']}")
            self.state.save()

        print(f"\nSummary: Created {stats['created']}, Updated {stats['updated']}, Skipped {stats['skipped']}")
        return stats

    def show_status(self) -> dict:
        """Show current sync status."""
        print("\n=== Unified Sync Status ===\n")

        # Check GitHub
        print("GitHub:")
        if self.github.check_auth():
            print(f"  Authenticated: Yes")
            print(f"  Repository: {self.github.owner}/{self.github.repo}")
            issues = self.github.get_issues(state="open")
            print(f"  Open issues: {len(issues)}")
        else:
            print("  Authenticated: No (run 'gh auth login')")

        # Check Notion
        print("\nNotion:")
        if self.notion.enabled:
            print("  API Key: Configured")
            pages = self.notion.query_database()
            print(f"  Database items: {len(pages)}")
        else:
            print("  API Key: Not configured")
            print(f"  Configure at: {NOTION_API_KEY_PATH}")

        # Show sync state
        print("\nSync State:")
        print(f"  Last sync: {self.state.last_sync or 'Never'}")
        print(f"  Last GitHub sync: {self.state.last_github_sync or 'Never'}")
        print(f"  Last Notion sync: {self.state.last_notion_sync or 'Never'}")
        print(f"  Tracked items: {len(self.state.items)}")
        print(f"  Pending conflicts: {len(self.state.conflicts)}")

        # Show recent sync history
        if self.state.sync_history:
            print("\nRecent Sync History:")
            for event in self.state.sync_history[-5:]:
                print(f"  [{event['timestamp'][:19]}] {event['type']}: {event['details']}")

        return {
            "github_auth": self.github.check_auth(),
            "notion_enabled": self.notion.enabled,
            "last_sync": self.state.last_sync,
            "tracked_items": len(self.state.items),
        }

    def show_diff(self) -> dict:
        """Show differences between systems."""
        print("\n=== Diff: GitHub vs Notion ===\n")

        github_items = self.fetch_github_items()
        notion_items = self.fetch_notion_items()

        changes = self.detect_changes(github_items, notion_items)

        print(f"\nNew in GitHub ({len(changes['new_in_github'])}):")
        for item in changes["new_in_github"][:10]:
            print(f"  + {item.title}")
        if len(changes["new_in_github"]) > 10:
            print(f"  ... and {len(changes['new_in_github']) - 10} more")

        print(f"\nNew in Notion ({len(changes['new_in_notion'])}):")
        for item in changes["new_in_notion"][:10]:
            print(f"  + {item.title}")
        if len(changes["new_in_notion"]) > 10:
            print(f"  ... and {len(changes['new_in_notion']) - 10} more")

        print(f"\nUpdated in GitHub ({len(changes['updated_in_github'])}):")
        for item in changes["updated_in_github"][:10]:
            print(f"  ~ {item.title}")

        print(f"\nUpdated in Notion ({len(changes['updated_in_notion'])}):")
        for item in changes["updated_in_notion"][:10]:
            print(f"  ~ {item.title}")

        print(f"\nConflicts ({len(changes['conflicts'])}):")
        for conflict in changes["conflicts"]:
            print(f"  ! {conflict['github'].title}")

        print(f"\nIn sync: {len(changes['in_sync'])} items")

        return changes


def main():
    parser = argparse.ArgumentParser(
        description="Unified Sync Coordinator - Bidirectional GitHub/Notion sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sync --bidirectional           Full bidirectional sync
  %(prog)s sync --github-only             Sync Notion -> GitHub only
  %(prog)s sync --notion-only             Sync GitHub -> Notion only
  %(prog)s sync --bidirectional --dry-run Preview changes without applying
  %(prog)s status                         Show sync status
  %(prog)s diff                           Show differences between systems
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Sync command
    sync_parser = subparsers.add_parser("sync", help="Synchronize systems")
    sync_group = sync_parser.add_mutually_exclusive_group(required=True)
    sync_group.add_argument("--bidirectional", action="store_true", help="Full bidirectional sync")
    sync_group.add_argument("--github-only", action="store_true", help="Sync to GitHub only")
    sync_group.add_argument("--notion-only", action="store_true", help="Sync to Notion only")
    sync_parser.add_argument("--dry-run", action="store_true", help="Preview without applying changes")
    sync_parser.add_argument("--conflict-resolution",
                             choices=["local_wins", "github_wins", "notion_wins", "newest_wins"],
                             default="local_wins", help="Conflict resolution strategy (local_wins = local repo data takes precedence)")

    # Status command
    subparsers.add_parser("status", help="Show sync status")

    # Diff command
    subparsers.add_parser("diff", help="Show differences between systems")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Determine conflict resolution strategy
    resolution = ConflictResolution.LOCAL_WINS
    if hasattr(args, "conflict_resolution") and args.conflict_resolution:
        resolution = ConflictResolution(args.conflict_resolution)

    coordinator = UnifiedSyncCoordinator(conflict_resolution=resolution)

    try:
        if args.command == "sync":
            if args.bidirectional:
                coordinator.sync_bidirectional(dry_run=args.dry_run)
            elif args.github_only:
                coordinator.sync_github_only(dry_run=args.dry_run)
            elif args.notion_only:
                coordinator.sync_notion_only(dry_run=args.dry_run)

        elif args.command == "status":
            coordinator.show_status()

        elif args.command == "diff":
            coordinator.show_diff()

    except KeyboardInterrupt:
        print("\nSync interrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
