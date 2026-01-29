# GitHub Workflows Quick Reference

## 🚀 Common Actions

### Trigger Full CI Pipeline
```bash
gh workflow run workflow-coordinator.yml --field workflow_type=full-ci
```

### Deploy to Production
```bash
gh workflow run workflow-coordinator.yml \
  --field workflow_type=release-candidate \
  --field environment=production
```

### Create New Release
```bash
# Patch release (0.0.X)
gh workflow run automated-release.yml --field release_type=patch

# Minor release (0.X.0)
gh workflow run automated-release.yml --field release_type=minor

# Major release (X.0.0)
gh workflow run automated-release.yml --field release_type=major

# Prerelease (beta, rc, alpha)
gh workflow run automated-release.yml \
  --field release_type=prerelease \
  --field prerelease_tag=beta
```

### Emergency Hotfix
```bash
gh workflow run workflow-coordinator.yml \
  --field workflow_type=hotfix \
  --field environment=production
```

## 📋 PR Best Practices

### PR Title Format
```
type(scope): description

Examples:
feat(backend): Add user authentication endpoint
fix(frontend): Resolve dashboard loading issue
docs(readme): Update installation instructions
```

### Auto-merge a PR
```bash
gh pr edit 123 --add-label "auto-merge"
```

### Link Issues in PR Description
```markdown
Closes #123
Fixes #456
Resolves #789
```

## 🏷️ Labels Reference

### Issue Types
- `type:bug` - Bug reports
- `type:feature` - Feature requests
- `type:question` - Questions
- `type:documentation` - Documentation improvements

### Issue Priority
- `priority:critical` - Production issues, data loss, security
- `priority:high` - Major bugs, important features
- `priority:medium` - Standard priority (default)
- `priority:low` - Nice-to-have improvements

### Components
- `component:backend` - Backend API changes
- `component:frontend` - Frontend UI changes
- `component:database` - Database changes
- `component:ci-cd` - CI/CD workflow changes
- `component:infrastructure` - Infrastructure changes

### Special Labels
- `security` - Security-related issues
- `performance` - Performance improvements
- `auto-merge` - Enable automatic PR merging
- `stale` - Inactive for 14+ days (PRs) or 30+ days (issues)

## 🔔 Notification Channels

### Slack Notifications
Sent for:
- ❌ Workflow failures
- ✅ Deployment completions
- 🎉 Release creations
- ⚠️  Critical health alerts
- 📊 Daily summaries

### Email Notifications
Sent for:
- Critical workflow failures
- Production deployments
- Security alerts

### GitHub Notifications
- PR/Issue comments
- Workflow summaries
- Check run results

## 📊 Health Score

### What It Means
- 💚 **80-100**: Healthy - Keep up the good work!
- 💛 **60-79**: Warning - Attention needed
- ❤️  **<60**: Critical - Immediate action required

### How It's Calculated
```
Start: 100 points
- Critical issues: -10 points each
- High priority issues: -5 points each
- Recent workflow failures: -3 points each
- Stale PRs: -2 points each
```

### Check Current Health
```bash
# View latest health check
gh run list --workflow=monitoring-notifications.yml --limit 1

# View health report
gh run view <run-id>
```

## 🔍 Troubleshooting

### View Workflow Logs
```bash
# List recent runs
gh run list --limit 10

# View specific run
gh run view <run-id>

# View failed run logs
gh run view <run-id> --log-failed
```

### Re-run Failed Workflow
```bash
gh run rerun <run-id> --failed
```

### Check Workflow Status
```bash
gh run watch <run-id>
```

### View PR Checks
```bash
gh pr checks 123
```

## 🔐 Security

### Report Security Issue
1. Create issue with `security` label
2. **Do not** include exploit code publicly
3. Consider using GitHub Security Advisories
4. Wait for security team response

### Check Security Scan Results
```bash
gh run list --workflow=security-scan.yml --limit 5
```

## 📈 Metrics

### View Recent Activity
```bash
# Recent PRs
gh pr list --state all --limit 20

# Recent issues
gh issue list --state all --limit 20

# Recent releases
gh release list --limit 10
```

### Check Coverage
```bash
# Latest CI run with coverage
gh run list --workflow=ci.yml --limit 1
gh run view <run-id> --log
```

## 🛠️ Workflow Types

### `workflow-coordinator.yml`
- **full-ci**: Complete CI with tests, security, build
- **fast-ci**: Quick validation with tests only
- **release-candidate**: Full pipeline + deployment
- **hotfix**: Emergency deployment
- **security-audit**: Security scanning only
- **performance-check**: Performance tests only

### `pr-automation.yml`
Auto-runs on:
- PR opened/synchronized
- PR reviews submitted

### `issue-management.yml`
Auto-runs on:
- Issue opened/edited/closed
- Scheduled stale check (daily)

### `automated-release.yml`
Trigger manually:
- Select release type (patch/minor/major/prerelease)
- Auto-generates changelog
- Creates GitHub release
- Triggers deployment

### `monitoring-notifications.yml`
Auto-runs:
- On workflow completions (any major workflow)
- Hourly health checks
- Daily summary at 9 AM UTC

## 💡 Tips

### Speed Up PR Reviews
1. Keep PRs small (<500 lines)
2. Add clear descriptions
3. Link related issues
4. Respond quickly to review comments
5. Use `auto-merge` for trivial changes

### Keep Issues Organized
1. Use descriptive titles
2. Add reproduction steps for bugs
3. Link related issues/PRs
4. Update labels as status changes
5. Close resolved issues promptly

### Maintain Repository Health
1. Review health reports weekly
2. Address critical issues within 24 hours
3. Keep stale items under control
4. Monitor workflow failure trends
5. Rotate secrets quarterly

## 📚 Documentation

- **Full Guide**: `docs/github-workflows-guide.md`
- **Implementation Summary**: `docs/WORKFLOW_COORDINATION_SUMMARY.md`
- **This Quick Reference**: `.github/WORKFLOWS_QUICK_REFERENCE.md`

## 🆘 Get Help

1. Check documentation
2. View workflow logs
3. Create issue with `component:ci-cd` label
4. Contact DevOps team

---

**Last Updated**: January 27, 2026
**Maintained By**: Platform Engineering Team
