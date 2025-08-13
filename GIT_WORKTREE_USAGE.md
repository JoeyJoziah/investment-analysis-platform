# Git Worktree Setup for Parallel Claude Sessions

## Overview
This project is configured with git worktrees to enable multiple Claude Code sessions to work in parallel on different features without conflicts.

## Current Worktree Structure

```
Main Repository: /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/7b51113a393465a37d4f1fda36b4d190088ac69ea8d5cf2f90400b3c14148ad3

Worktrees:
├── ../investment-worktrees/data-pipeline     [feature/data-pipeline]
├── ../investment-worktrees/ml-models         [feature/ml-models]
├── ../investment-worktrees/frontend          [feature/frontend]
├── ../investment-worktrees/api-development   [feature/api-development]
├── ../investment-worktrees/testing           [feature/testing]
└── ../investment-worktrees/deployment        [feature/deployment]
```

## How to Use Worktrees for Parallel Development

### 1. Start Different Claude Sessions
Each Claude session should work in a different worktree to avoid conflicts:

**Session 1 - Data Pipeline Development:**
```bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/investment-worktrees/data-pipeline
# Work on Airflow DAGs, data ingestion, ETL pipelines
```

**Session 2 - ML Model Development:**
```bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/investment-worktrees/ml-models
# Work on ML models, training, feature engineering
```

**Session 3 - Frontend Development:**
```bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/investment-worktrees/frontend
# Work on React components, UI/UX
```

**Session 4 - API Development:**
```bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/investment-worktrees/api-development
# Work on FastAPI endpoints, routers, business logic
```

**Session 5 - Testing:**
```bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/investment-worktrees/testing
# Write tests, run test suites, performance testing
```

**Session 6 - Deployment & DevOps:**
```bash
cd /mnt/wsl/docker-desktop-bind-mounts/Ubuntu-24.04/investment-worktrees/deployment
# Work on Docker, Kubernetes, CI/CD pipelines
```

### 2. Common Git Worktree Commands

**List all worktrees:**
```bash
git worktree list
```

**Add a new worktree:**
```bash
git branch feature/new-feature
git worktree add ../investment-worktrees/new-feature feature/new-feature
```

**Remove a worktree:**
```bash
git worktree remove ../investment-worktrees/worktree-name
```

**Prune stale worktree information:**
```bash
git worktree prune
```

### 3. Workflow Best Practices

1. **Separate Concerns**: Each worktree should focus on a specific feature or module
2. **Regular Commits**: Commit frequently in each worktree to track progress
3. **Sync with Main**: Regularly merge or rebase from main to keep branches updated
4. **Clean Merges**: When feature is complete, merge back to main branch

### 4. Merging Changes Back to Main

From the main repository:
```bash
# Switch to main branch
git checkout main

# Merge a feature branch
git merge feature/data-pipeline

# Or create a pull request for review
git push origin feature/data-pipeline
# Then create PR on GitHub
```

### 5. Keeping Worktrees Updated

In each worktree:
```bash
# Fetch latest changes
git fetch origin

# Rebase on main
git rebase origin/main

# Or merge main
git merge origin/main
```

## Benefits of This Setup

1. **Parallel Development**: Multiple Claude sessions can work simultaneously
2. **No Conflicts**: Each session works in isolation
3. **Clean History**: Each feature has its own branch history
4. **Easy Integration**: Changes can be merged when ready
5. **Resource Efficiency**: Shares object database between worktrees

## Example Use Cases

### Scenario 1: Comprehensive Feature Development
- Session 1: Develops backend API endpoints in `api-development`
- Session 2: Creates frontend components in `frontend`
- Session 3: Writes tests in `testing`
- All work in parallel, merge when complete

### Scenario 2: Data Pipeline and ML
- Session 1: Sets up Airflow DAGs in `data-pipeline`
- Session 2: Trains ML models in `ml-models`
- Session 3: Deploys to production in `deployment`

### Scenario 3: Bug Fixes and Features
- Main branch: Quick bug fixes
- Feature branches: New feature development
- Testing branch: Comprehensive test coverage

## Troubleshooting

**Lock file issues:**
```bash
rm -f .git/index.lock
```

**Worktree not accessible:**
```bash
git worktree prune
```

**Branch conflicts:**
```bash
git fetch --all
git rebase origin/main
```

## Notes

- Each worktree has its own working directory but shares the Git object database
- Changes in one worktree don't affect others until merged
- Perfect for parallel Claude Code sessions working on different aspects
- Reduces merge conflicts and improves development velocity