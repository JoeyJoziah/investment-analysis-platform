# GitHub UI Navigation Reference

Quick reference for navigating GitHub's interface during CI/CD setup.

## Main Navigation Areas

### Repository Tabs (Top of Repository Page)
```
[< > Code] [Issues] [Pull requests] [Actions] [Projects] [Security] [Insights] [Settings]
                                     ^^^^^^                                      ^^^^^^^^
                                   Monitor CI/CD                            Configure everything
```

## Settings Navigation Tree

### Primary Settings Sections
```
Repository Settings (gear icon)
├── 📋 General
│   ├── Repository name
│   ├── Description  
│   ├── Website
│   ├── Topics
│   ├── Features (Issues, Projects, Wiki, Discussions)
│   ├── Pull Requests
│   ├── Archives
│   └── Danger Zone
│
├── 🏠 Access
│   ├── Collaborators and teams
│   ├── Moderation options
│   └── Interaction limits
│
├── 🔐 Security
│   ├── Security and analysis
│   ├── Deploy keys
│   └── Secrets and variables ⭐ IMPORTANT
│       ├── Actions ⭐⭐⭐ (Repository secrets)
│       ├── Codespaces
│       └── Environments
│
├── 🔄 Integrations
│   ├── GitHub Apps
│   ├── Email notifications
│   └── Webhooks
│
└── ⚙️ Code and automation
    ├── Branches ⭐ IMPORTANT (Branch protection)
    ├── Tags
    ├── Actions ⭐ IMPORTANT (Workflow permissions)
    │   ├── General
    │   ├── Runners
    │   └── Runner groups
    ├── Webhooks
    ├── Environments
    ├── Codespaces
    ├── Pages
    └── Security
```

## Step-by-Step UI Navigation

### 1. Setting Up Secrets
**Path**: `Repository → Settings → Secrets and variables → Actions`

**Visual Landmarks**:
- Look for the **Settings** tab (gear icon) at the top
- In left sidebar, find **"Secrets and variables"** with lock icon 🔐
- Click to expand, then click **"Actions"**
- You'll see tabs: `Repository secrets` | `Environment secrets` | `Variables`
- Click **"New repository secret"** button (green button, top right)

**Form Fields**:
```
┌─────────────────────────────────────────┐
│ Name * │ [SECRET_NAME_HERE              ] │
├────────┼─────────────────────────────────────┤
│ Secret │ [••••••••••••••••••••••••••••••] │
│        │ [Paste your secret value here    ] │
│        │ [This field is hidden for        ] │
│        │ [security                        ] │
└────────┴─────────────────────────────────────┘
          [Add secret] [Cancel]
```

### 2. Branch Protection Rules
**Path**: `Repository → Settings → Branches`

**Visual Process**:
1. Settings tab → **Branches** in left sidebar (🌿 icon)
2. Under **"Branch protection rules"**: Click **"Add rule"**
3. **Branch name pattern**: Type `main`
4. **Checkboxes to enable** (scroll down to see all):

```
☐ Restrict pushes that create files larger than 100MB
☐ Require a pull request before merging
  ☐ Require approvals: [1] ▼
  ☐ Dismiss stale reviews when new commits are pushed
  ☐ Require review from code owners
  ☐ Restrict pushes that create files larger than [100] MB
☐ Require status checks to pass before merging
  ☐ Require branches to be up to date before merging
  Search for status checks: [backend-quality      ]
                           [frontend-quality     ]
                           [security-scan        ]
☐ Require conversation resolution before merging
☐ Require signed commits  
☐ Require linear history
☐ Require deployments to succeed before merging
☐ Lock branch
☐ Do not allow bypassing the above settings
☐ Restrict pushes that create files larger than 100MB
```

### 3. GitHub Actions Settings
**Path**: `Repository → Settings → Actions → General`

**Configuration Options**:
```
Actions permissions
○ Disable Actions
○ Allow [your-org] actions and reusable workflows  
● Allow all actions and reusable workflows ⭐ SELECT THIS

Fork pull request workflows from outside collaborators
● Require approval for first-time contributors ⭐ RECOMMENDED
○ Require approval for all outside collaborators

Workflow permissions  
● Read and write permissions ⭐ SELECT THIS
○ Read repository contents permission
☐ Allow GitHub Actions to create and approve pull requests ⭐ CHECK THIS
```

### 4. Monitoring Workflows
**Path**: `Repository → Actions`

**Actions Dashboard Layout**:
```
┌─ Workflows (Left Sidebar) ────┬─ Main Content ─────────────────────────┐
│ 🔄 All workflows              │ Workflow runs                          │
│ ✅ CI Pipeline                │ ┌─────────────────────────────────────┐ │
│ 🚀 Staging Deploy             │ │ ✅ CI Pipeline                      │ │  
│ 🌟 Production Deploy          │ │    feat: add new feature            │ │
│ 🔒 Security Scan              │ │    main • 2m ago • #42              │ │
│ 🧹 Cleanup                    │ └─────────────────────────────────────┘ │
│                               │ ┌─────────────────────────────────────┐ │
│ Filter by:                    │ │ ❌ Security Scan                    │ │
│ [Branch] [Actor] [Status]     │ │    fix: security vulnerability     │ │
└───────────────────────────────┤ │    develop • 5m ago • #41           │ │
                                │ └─────────────────────────────────────┘ │
                                └─────────────────────────────────────────┘
```

**Workflow Run Details**:
Click on any workflow run to see:
```
┌─ Jobs (Left) ─────────┬─ Job Details (Right) ──────────────────┐
│ ✅ setup-job          │ Step details and logs                  │
│ ✅ backend-quality    │ ┌─────────────────────────────────────┐ │
│ ❌ backend-tests      │ │ > Set up job                        │ │
│ ⏸️ frontend-quality   │ │   ✅ Complete (2s)                  │ │
│                       │ │ > Checkout code                     │ │
│                       │ │   ✅ Complete (1s)                  │ │
│                       │ │ > Set up Python 3.11                │ │
│                       │ │   ✅ Complete (15s)                 │ │
│                       │ │ > Install dependencies              │ │
│                       │ │   ❌ Failed (Error logs below...)   │ │
└───────────────────────┴─────────────────────────────────────────┘
```

### 5. Personal Access Token Creation
**Path**: `GitHub Profile → Settings → Developer settings → Personal access tokens`

**Navigation Steps**:
```
1. Click your profile picture (top right corner)
2. Click "Settings" from dropdown menu
3. Scroll to bottom of left sidebar
4. Click "Developer settings" 
5. Click "Personal access tokens"
6. Click "Tokens (classic)"
7. Click "Generate new token" → "Generate new token (classic)"
```

**Token Form**:
```
┌─────────────────────────────────────────────────────────────┐
│ Note: [Investment Analysis App CI/CD                       ] │
│ Expiration: [90 days] ▼                                     │
│                                                             │
│ Select scopes:                                              │
│ ☐ repo         Full control of private repositories        │
│   ☐ repo:status    Access commit status                    │
│   ☐ repo_deployment Access deployment status               │
│   ☐ public_repo     Access public repositories              │
│   ☐ repo:invite     Access repository invitations          │
│   ☐ security_events Access security events                 │
│                                                             │
│ ☐ workflow     Update GitHub Action workflows              │
│                                                             │  
│ ☐ write:packages Upload packages to GitHub Package Registry │
│ ☐ read:packages  Download packages from GitHub Package Registry │
│ ☐ delete:packages Delete packages from GitHub Package Registry │
└─────────────────────────────────────────────────────────────┘
           [Generate token] [Cancel]
```

## Visual Indicators & Status Icons

### Workflow Status Icons
- ✅ **Green checkmark**: Success
- ❌ **Red X**: Failed  
- 🟡 **Yellow dot**: In progress
- ⏸️ **Gray circle**: Skipped/Cancelled
- 🔄 **Blue arrow**: Queued

### Branch Protection Status
- 🟢 **Green**: All checks passed
- 🔴 **Red**: Some checks failed
- 🟡 **Yellow**: Checks in progress
- ⚪ **Gray**: No status checks

### Secret Security Levels
- 🔒 **Repository secrets**: Available to all repository workflows
- 🌍 **Environment secrets**: Only available to specific environments
- 👥 **Organization secrets**: Shared across organization repositories

## Common UI Elements

### Buttons
- **Green buttons**: Primary actions (Create, Add, Save)
- **Gray buttons**: Secondary actions (Cancel, Edit)
- **Red buttons**: Destructive actions (Delete, Remove)

### Form Patterns
- **Required fields**: Marked with red asterisk (*)
- **Optional fields**: No asterisk
- **Sensitive fields**: Show dots (••••) instead of text

### Search & Filter
- Most lists have search boxes at the top
- Use filters to narrow down results
- Sort options usually available

## Keyboard Shortcuts

### Global GitHub Shortcuts
- `s` or `/`: Focus search bar
- `g` + `c`: Go to Code tab  
- `g` + `i`: Go to Issues tab
- `g` + `p`: Go to Pull requests tab
- `g` + `a`: Go to Actions tab
- `?`: Show all keyboard shortcuts

### Repository Navigation
- `t`: Activate file finder
- `l`: Jump to line number
- `b`: Open blame view
- `y`: Get permanent link to file

## Troubleshooting UI Issues

### Can't Find Settings Tab?
- Make sure you have admin/write permissions to repository
- Settings tab appears at top of repository page
- If missing, you might be on organization page instead of repository

### Secrets Section Missing?
- Ensure GitHub Actions is enabled in repository settings
- Check you're in "Actions" subsection under "Secrets and variables"
- Verify repository permissions

### Actions Tab Not Visible?
- Actions might be disabled for repository
- Go to Settings → Actions → General to enable
- Organization policy might restrict Actions

### Branch Protection Not Working?
- Rules only apply to future pushes/PRs
- Admin privileges might bypass rules
- Check rule pattern matches your branch names exactly

---

**Pro Tips**:
1. **Browser bookmarks**: Bookmark frequently used pages like Actions and Settings
2. **Multiple tabs**: Open Settings in one tab, Actions monitoring in another  
3. **Notifications**: Enable email notifications for workflow failures
4. **Mobile access**: GitHub mobile app works for monitoring builds
5. **Browser extensions**: GitHub-specific extensions can enhance UI experience

This reference card should be kept handy during initial CI/CD setup!