#!/usr/bin/env python3
"""
Notion Sync Script for Investment Analysis Platform

Syncs TODO.md with the Notion Product Development Tracker database.
Supports bidirectional sync: push local changes to Notion or pull from Notion.

Usage:
    python scripts/notion_sync.py push    # Push TODO.md to Notion
    python scripts/notion_sync.py pull    # Pull from Notion to TODO.md
    python scripts/notion_sync.py status  # Show sync status
    python scripts/notion_sync.py --help  # Show help

Configuration:
    API key stored in: ~/.config/notion/api_key
    Database ID: 2f3b9fc9-1d9d-8026-9719-f82b0e311f50
    Data Source ID: 2f3b9fc9-1d9d-80cc-b349-000b8158ee99
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# Configuration
NOTION_VERSION = "2025-09-03"
DATABASE_ID = "2f3b9fc9-1d9d-8026-9719-f82b0e311f50"
DATA_SOURCE_ID = "2f3b9fc9-1d9d-80cc-b349-000b8158ee99"
API_KEY_PATH = Path.home() / ".config" / "notion" / "api_key"
TODO_PATH = Path(__file__).parent.parent / "TODO.md"
SYNC_STATE_PATH = Path(__file__).parent.parent / ".notion_sync_state.json"


@dataclass
class Task:
    """Represents a task that can be synced between TODO.md and Notion."""
    title: str
    status: str = "To Do"
    priority: str = "Medium"
    category: str = "Feature Development"
    milestone: str = "Launch Ready"
    notes: str = ""
    notion_id: Optional[str] = None
    source: str = "local"  # 'local' or 'notion'

    def to_notion_properties(self) -> dict:
        """Convert task to Notion page properties."""
        props = {
            "Task": {"title": [{"text": {"content": self.title}}]},
            "Status": {"status": {"name": self.status}},
            "Priority": {"select": {"name": self.priority}},
            "Category": {"select": {"name": self.category}},
            "Launch Milestone": {"select": {"name": self.milestone}},
        }
        if self.notes:
            # Truncate notes to 2000 chars (Notion limit for rich_text)
            notes_truncated = self.notes[:2000]
            props["Notes"] = {"rich_text": [{"text": {"content": notes_truncated}}]}
        return props

    @classmethod
    def from_notion_page(cls, page: dict) -> "Task":
        """Create a Task from a Notion page response."""
        props = page.get("properties", {})

        # Extract title
        title_prop = props.get("Task", {}).get("title", [])
        title = title_prop[0]["text"]["content"] if title_prop else "Untitled"

        # Extract status
        status_prop = props.get("Status", {}).get("status", {})
        status = status_prop.get("name", "To Do") if status_prop else "To Do"

        # Extract priority
        priority_prop = props.get("Priority", {}).get("select", {})
        priority = priority_prop.get("name", "Medium") if priority_prop else "Medium"

        # Extract category
        category_prop = props.get("Category", {}).get("select", {})
        category = category_prop.get("name", "Feature Development") if category_prop else "Feature Development"

        # Extract milestone
        milestone_prop = props.get("Launch Milestone", {}).get("select", {})
        milestone = milestone_prop.get("name", "Launch Ready") if milestone_prop else "Launch Ready"

        # Extract notes
        notes_prop = props.get("Notes", {}).get("rich_text", [])
        notes = notes_prop[0]["text"]["content"] if notes_prop else ""

        return cls(
            title=title,
            status=status,
            priority=priority,
            category=category,
            milestone=milestone,
            notes=notes,
            notion_id=page.get("id"),
            source="notion"
        )


class NotionClient:
    """Client for interacting with Notion API."""

    def __init__(self):
        self.api_key = self._load_api_key()
        self.base_url = "https://api.notion.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json"
        }

    def _load_api_key(self) -> str:
        """Load API key from config file."""
        if not API_KEY_PATH.exists():
            raise FileNotFoundError(
                f"Notion API key not found at {API_KEY_PATH}\n"
                "Please run: mkdir -p ~/.config/notion && echo 'your-api-key' > ~/.config/notion/api_key"
            )
        return API_KEY_PATH.read_text().strip()

    def query_database(self, filter_obj: Optional[dict] = None) -> list[dict]:
        """Query all pages from the database."""
        url = f"{self.base_url}/data_sources/{DATA_SOURCE_ID}/query"
        all_results = []
        has_more = True
        start_cursor = None

        while has_more:
            payload = {"page_size": 100}
            if filter_obj:
                payload["filter"] = filter_obj
            if start_cursor:
                payload["start_cursor"] = start_cursor

            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            data = response.json()

            all_results.extend(data.get("results", []))
            has_more = data.get("has_more", False)
            start_cursor = data.get("next_cursor")

        return all_results

    def create_page(self, task: Task) -> dict:
        """Create a new page in the database."""
        url = f"{self.base_url}/pages"
        payload = {
            "parent": {"database_id": DATABASE_ID},
            "properties": task.to_notion_properties()
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def update_page(self, page_id: str, task: Task) -> dict:
        """Update an existing page."""
        url = f"{self.base_url}/pages/{page_id}"
        payload = {"properties": task.to_notion_properties()}
        response = requests.patch(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_page(self, page_id: str) -> dict:
        """Get a single page by ID."""
        url = f"{self.base_url}/pages/{page_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


class TodoParser:
    """Parser for TODO.md file format."""

    # Status mapping from TODO.md to Notion
    STATUS_MAP = {
        "pending": "To Do",
        "to do": "To Do",
        "not started": "To Do",
        "ready for testing": "To Do",
        "in progress": "In Progress",
        "wip": "In Progress",
        "in review": "In Review",
        "review": "In Review",
        "blocked": "Blocked",
        "done": "Done",
        "complete": "Done",
        "completed": "Done",
    }

    # Priority mapping
    PRIORITY_MAP = {
        "high priority": "Critical",
        "critical": "Critical",
        "high": "High",
        "medium priority": "Medium",
        "medium": "Medium",
        "low priority": "Low",
        "low": "Low",
        "optional": "Low",
    }

    def parse_todo_md(self, content: str) -> list[Task]:
        """Parse TODO.md content into Task objects."""
        tasks = []
        current_priority = "Medium"
        current_status = "To Do"

        lines = content.split("\n")
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check for priority sections
            if line.startswith("## HIGH PRIORITY"):
                current_priority = "Critical"
            elif line.startswith("## MEDIUM PRIORITY"):
                current_priority = "High"
            elif line.startswith("## LOW PRIORITY"):
                current_priority = "Medium"
            elif line.startswith("## Already Complete"):
                current_status = "Done"
                current_priority = "Medium"

            # Parse task headers (### N. Task Name)
            task_match = re.match(r"^###\s+(?:~~)?(\d+)\.\s*(.+?)(?:~~)?\s*(?:✅\s*COMPLETE)?$", line)
            if task_match:
                task_num = task_match.group(1)
                task_title = task_match.group(2).strip()

                # Check if marked as complete
                is_complete = "~~" in line or "✅" in line or "COMPLETE" in line.upper()

                # Look ahead for status and notes
                notes = []
                task_status = "Done" if is_complete else current_status
                j = i + 1

                while j < len(lines):
                    next_line = lines[j]
                    if next_line.startswith("###") or next_line.startswith("## "):
                        break

                    # Check for status line
                    status_match = re.match(r"\*\*Status\*\*:\s*(.+)", next_line.strip())
                    if status_match:
                        status_text = status_match.group(1).lower().strip()
                        # Remove emoji and extra text
                        status_text = re.sub(r"✅|complete|completed", "", status_text, flags=re.IGNORECASE).strip()
                        if status_text in self.STATUS_MAP:
                            task_status = self.STATUS_MAP[status_text]
                        elif "complete" in next_line.lower() or "✅" in next_line:
                            task_status = "Done"

                    # Collect notes (non-empty, non-code, non-header lines)
                    stripped = next_line.strip()
                    if stripped and not stripped.startswith("```") and not stripped.startswith("**") and not stripped.startswith("#"):
                        if len(stripped) > 3:  # Skip very short lines
                            notes.append(stripped)

                    j += 1

                # Determine category based on task title
                category = self._infer_category(task_title)

                task = Task(
                    title=task_title,
                    status=task_status,
                    priority=current_priority,
                    category=category,
                    milestone="Launch Ready",
                    notes=" ".join(notes[:3])[:500]  # First 3 note lines, max 500 chars
                )
                tasks.append(task)

            # Parse checklist items in Already Complete section
            if current_status == "Done":
                checklist_match = re.match(r"^-\s*\[x\]\s*\*\*(.+?)\*\*", line)
                if checklist_match:
                    task_title = checklist_match.group(1).strip()
                    task = Task(
                        title=task_title,
                        status="Done",
                        priority="High",
                        category=self._infer_category(task_title),
                        milestone="Launch Ready"
                    )
                    tasks.append(task)

            i += 1

        return tasks

    def _infer_category(self, title: str) -> str:
        """Infer category from task title."""
        title_lower = title.lower()

        if any(kw in title_lower for kw in ["ssl", "deploy", "production", "launch", "smtp", "email"]):
            return "Launch Preparation"
        elif any(kw in title_lower for kw in ["test", "testing"]):
            return "Feature Development"
        elif any(kw in title_lower for kw in ["ml", "model", "train", "prediction", "analysis"]):
            return "Investment Analysis"
        elif any(kw in title_lower for kw in ["bug", "fix", "error"]):
            return "Bug Fix"
        elif any(kw in title_lower for kw in ["backup", "config", "infrastructure", "monitor", "doc"]):
            return "Maintenance"
        else:
            return "Feature Development"


class SyncManager:
    """Manages synchronization between TODO.md and Notion."""

    def __init__(self):
        self.client = NotionClient()
        self.parser = TodoParser()
        self.sync_state = self._load_sync_state()

    def _load_sync_state(self) -> dict:
        """Load sync state from file."""
        if SYNC_STATE_PATH.exists():
            return json.loads(SYNC_STATE_PATH.read_text())
        return {"task_mapping": {}, "last_sync": None}

    def _save_sync_state(self):
        """Save sync state to file."""
        self.sync_state["last_sync"] = datetime.now().isoformat()
        SYNC_STATE_PATH.write_text(json.dumps(self.sync_state, indent=2))

    def push_to_notion(self, dry_run: bool = False) -> dict:
        """Push TODO.md tasks to Notion."""
        # Parse TODO.md
        if not TODO_PATH.exists():
            raise FileNotFoundError(f"TODO.md not found at {TODO_PATH}")

        content = TODO_PATH.read_text()
        local_tasks = self.parser.parse_todo_md(content)

        # Get existing Notion tasks
        notion_pages = self.client.query_database()
        notion_tasks = {
            Task.from_notion_page(p).title: p["id"]
            for p in notion_pages
        }

        stats = {"created": 0, "updated": 0, "skipped": 0}

        for task in local_tasks:
            if task.title in notion_tasks:
                # Task exists - check if update needed
                page_id = notion_tasks[task.title]
                if not dry_run:
                    try:
                        self.client.update_page(page_id, task)
                        stats["updated"] += 1
                        print(f"  Updated: {task.title}")
                    except Exception as e:
                        print(f"  Error updating {task.title}: {e}")
                        stats["skipped"] += 1
                else:
                    print(f"  [DRY RUN] Would update: {task.title}")
                    stats["updated"] += 1
            else:
                # New task - create it
                if not dry_run:
                    try:
                        result = self.client.create_page(task)
                        self.sync_state["task_mapping"][task.title] = result["id"]
                        stats["created"] += 1
                        print(f"  Created: {task.title}")
                    except Exception as e:
                        print(f"  Error creating {task.title}: {e}")
                        stats["skipped"] += 1
                else:
                    print(f"  [DRY RUN] Would create: {task.title}")
                    stats["created"] += 1

        if not dry_run:
            self._save_sync_state()

        return stats

    def pull_from_notion(self, dry_run: bool = False) -> dict:
        """Pull tasks from Notion and generate TODO.md content."""
        notion_pages = self.client.query_database()
        tasks = [Task.from_notion_page(p) for p in notion_pages]

        # Group tasks by status and priority
        high_priority = [t for t in tasks if t.priority in ["Critical", "High"] and t.status != "Done"]
        medium_priority = [t for t in tasks if t.priority == "Medium" and t.status != "Done"]
        low_priority = [t for t in tasks if t.priority == "Low" and t.status != "Done"]
        completed = [t for t in tasks if t.status == "Done"]

        stats = {
            "total": len(tasks),
            "high_priority": len(high_priority),
            "medium_priority": len(medium_priority),
            "low_priority": len(low_priority),
            "completed": len(completed)
        }

        if not dry_run:
            content = self._generate_todo_content(high_priority, medium_priority, low_priority, completed)
            TODO_PATH.write_text(content)
            self._save_sync_state()
            print(f"  Updated TODO.md with {stats['total']} tasks")
        else:
            print(f"  [DRY RUN] Would update TODO.md with {stats['total']} tasks")

        return stats

    def _generate_todo_content(self, high: list, medium: list, low: list, done: list) -> str:
        """Generate TODO.md content from tasks."""
        lines = [
            "# Investment Analysis Platform - Next Steps TODO",
            "",
            f"**Last Updated**: {datetime.now().strftime('%Y-%m-%d')}",
            f"**Current Status**: Synced from Notion",
            "",
            "---",
            "",
        ]

        def add_section(title: str, tasks: list, start_num: int) -> int:
            if not tasks:
                return start_num
            lines.append(f"## {title}")
            lines.append("")
            for i, task in enumerate(tasks, start=start_num):
                status_marker = "✅ COMPLETE" if task.status == "Done" else ""
                prefix = "~~" if task.status == "Done" else ""
                suffix = "~~" if task.status == "Done" else ""
                lines.append(f"### {prefix}{i}. {task.title}{suffix} {status_marker}")
                lines.append("")
                lines.append(f"**Status**: {task.status}")
                if task.notes:
                    lines.append(f"**Notes**: {task.notes}")
                lines.append("")
            return start_num + len(tasks)

        num = 1
        num = add_section("HIGH PRIORITY (Required for Production)", high, num)
        num = add_section("MEDIUM PRIORITY (Recommended Before Production)", medium, num)
        num = add_section("LOW PRIORITY (Optional Enhancements)", low, num)

        if done:
            lines.append("## Already Complete")
            lines.append("")
            for task in done:
                lines.append(f"- [x] **{task.title}**")
            lines.append("")

        return "\n".join(lines)

    def show_status(self) -> dict:
        """Show current sync status."""
        # Get Notion tasks
        notion_pages = self.client.query_database()
        notion_tasks = [Task.from_notion_page(p) for p in notion_pages]

        # Parse local tasks
        local_tasks = []
        if TODO_PATH.exists():
            content = TODO_PATH.read_text()
            local_tasks = self.parser.parse_todo_md(content)

        # Compare
        notion_titles = {t.title for t in notion_tasks}
        local_titles = {t.title for t in local_tasks}

        only_notion = notion_titles - local_titles
        only_local = local_titles - notion_titles
        both = notion_titles & local_titles

        status = {
            "notion_total": len(notion_tasks),
            "local_total": len(local_tasks),
            "synced": len(both),
            "only_in_notion": len(only_notion),
            "only_in_local": len(only_local),
            "last_sync": self.sync_state.get("last_sync", "Never")
        }

        print("\n=== Sync Status ===")
        print(f"Notion tasks: {status['notion_total']}")
        print(f"Local tasks: {status['local_total']}")
        print(f"Synced: {status['synced']}")
        print(f"Only in Notion: {status['only_in_notion']}")
        print(f"Only in Local: {status['only_in_local']}")
        print(f"Last sync: {status['last_sync']}")

        if only_notion:
            print("\nTasks only in Notion:")
            for t in list(only_notion)[:5]:
                print(f"  - {t}")
            if len(only_notion) > 5:
                print(f"  ... and {len(only_notion) - 5} more")

        if only_local:
            print("\nTasks only in local TODO.md:")
            for t in list(only_local)[:5]:
                print(f"  - {t}")
            if len(only_local) > 5:
                print(f"  ... and {len(only_local) - 5} more")

        return status


def main():
    parser = argparse.ArgumentParser(
        description="Sync TODO.md with Notion Product Development Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s push           Push TODO.md changes to Notion
  %(prog)s push --dry-run Preview what would be pushed
  %(prog)s pull           Pull from Notion to TODO.md
  %(prog)s status         Show sync status
        """
    )

    parser.add_argument(
        "action",
        choices=["push", "pull", "status"],
        help="Sync action to perform"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without applying them"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )

    args = parser.parse_args()

    try:
        manager = SyncManager()

        if args.action == "push":
            print(f"Pushing TODO.md to Notion{' (dry run)' if args.dry_run else ''}...")
            stats = manager.push_to_notion(dry_run=args.dry_run)
            print(f"\nSummary: Created {stats['created']}, Updated {stats['updated']}, Skipped {stats['skipped']}")

        elif args.action == "pull":
            print(f"Pulling from Notion to TODO.md{' (dry run)' if args.dry_run else ''}...")
            stats = manager.pull_from_notion(dry_run=args.dry_run)
            print(f"\nSummary: {stats['total']} total tasks ({stats['completed']} completed)")

        elif args.action == "status":
            manager.show_status()

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Notion API error: {e}", file=sys.stderr)
        if args.verbose:
            print(f"Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
