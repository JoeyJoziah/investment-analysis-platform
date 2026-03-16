# Agent Categories - Quick Reference Card

**Print this page for easy reference when finding agents**

## 🎯 Which Category Do I Need?

| I need to... | Use Category | Example Agents |
|--------------|--------------|----------------|
| Write code, review, test | **1-core** | coder, reviewer, tester |
| Coordinate multiple agents | **2-swarm-coordination** | hierarchical-coordinator, mesh-coordinator |
| Check security or optimize | **3-security-performance** | security-reviewer, performance-analyzer |
| Work with GitHub/PRs | **4-github-repository** | pr-manager, issue-tracker |
| Follow SPARC methodology | **5-sparc-methodology** | specification, architecture |
| Build specific features | **6-specialized-development** | backend-dev, ml-developer |
| Run tests or validate | **7-testing-validation** | tdd-london-swarm, e2e-runner |

## 📁 Category Details

### 1-core (5 agents)
**When**: Every project needs these
**Location**: `.claude/agents/1-core/`
```
coder.md           - Write and implement code
reviewer.md        - Code review and quality
tester.md          - Testing and validation
planner.md         - Project planning
researcher.md      - Research and analysis
```

### 2-swarm-coordination (25 agents)
**When**: Multi-agent orchestration needed
**Location**: `.claude/agents/2-swarm-coordination/`
```
Key agents:
- hierarchical-coordinator.md  - Queen-led coordination
- mesh-coordinator.md          - Peer-to-peer coordination
- byzantine-coordinator.md     - Fault-tolerant consensus
- load-balancer.md            - Resource distribution
- memory-coordinator.md        - Shared memory management
```

### 3-security-performance (15 agents)
**When**: Security review or optimization needed
**Location**: `.claude/agents/3-security-performance/`
```
Key agents:
- security-reviewer.md         - Security analysis
- performance-analyzer.md      - Performance profiling
- sona-learning-optimizer.md   - Neural optimization
- risk-assessor.md            - Risk evaluation
- code-analyzer.md            - Static analysis
```

### 4-github-repository (20 agents)
**When**: GitHub workflows and repository management
**Location**: `.claude/agents/4-github-repository/`
```
Key agents:
- pr-manager.md               - Pull request management
- issue-tracker.md            - Issue tracking
- release-manager.md          - Release coordination
- code-review-swarm.md        - Automated reviews
- workflow-automation.md      - CI/CD automation
```

### 5-sparc-methodology (10 agents)
**When**: Following structured development process
**Location**: `.claude/agents/5-sparc-methodology/`
```
Key agents:
- specification.md            - Requirements analysis
- architecture.md             - System design
- pseudocode.md              - Algorithm design
- refinement.md              - Code refinement
- tdd-guide.md               - Test-driven development
```

### 6-specialized-development (35 agents)
**When**: Domain-specific development tasks
**Location**: `.claude/agents/6-specialized-development/`
```
Key agents:
- backend-api-swarm.md        - Backend development
- ml-developer.md             - Machine learning
- financial-modeler.md        - Financial analysis
- ui-visualization-swarm.md   - UI/UX design
- mobile-dev.md              - Mobile development
```

### 7-testing-validation (10 agents)
**When**: Comprehensive testing needed
**Location**: `.claude/agents/7-testing-validation/`
```
Key agents:
- tdd-london-swarm.md         - Test-driven development
- e2e-runner.md               - End-to-end testing
- production-validator.md     - Production validation
- project-quality-swarm.md    - Quality assurance
```

## 🔍 Quick Search Guide

### By Task Type

| Task | Category | Agent |
|------|----------|-------|
| Implement feature | 1-core | coder |
| Review code | 1-core | reviewer |
| Write tests | 1-core | tester |
| Plan project | 1-core | planner |
| Research tech | 1-core | researcher |
| Coordinate team | 2-swarm-coordination | hierarchical-coordinator |
| Security audit | 3-security-performance | security-reviewer |
| Optimize performance | 3-security-performance | performance-analyzer |
| Create PR | 4-github-repository | pr-manager |
| Track issues | 4-github-repository | issue-tracker |
| Write specs | 5-sparc-methodology | specification |
| Design architecture | 5-sparc-methodology | architecture |
| Build API | 6-specialized-development | backend-api-swarm |
| Train ML model | 6-specialized-development | ml-developer |
| Run E2E tests | 7-testing-validation | e2e-runner |

### By Technology

| Technology | Category | Agent |
|------------|----------|-------|
| Backend/API | 6-specialized-development | backend-api-swarm |
| Mobile | 6-specialized-development | mobile-dev |
| Machine Learning | 6-specialized-development | ml-developer |
| Financial | 6-specialized-development | financial-modeler |
| UI/UX | 6-specialized-development | ui-visualization-swarm |
| GitHub Actions | 4-github-repository | ops-cicd-github |
| Testing | 7-testing-validation | tdd-london-swarm |

## 💡 Pro Tips

### Finding the Right Agent

1. **Start with the category**: Know what you're trying to do
2. **Check the quick search**: Use the tables above
3. **Look at key agents**: Each category lists the most common agents
4. **Browse the directory**: All agents in one logical place

### Common Workflows

**Feature Development**:
```
1-core/planner → 1-core/coder → 1-core/tester → 1-core/reviewer
```

**Pull Request**:
```
4-github-repository/pr-manager → 4-github-repository/code-review-swarm
```

**SPARC Methodology**:
```
5-sparc-methodology/specification → architecture → refinement → implementer-sparc-coder
```

**Quality Assurance**:
```
7-testing-validation/tdd-london-swarm → e2e-runner → production-validator
```

## 📊 Category Size Reference

| Category | Agents | Complexity |
|----------|--------|------------|
| 1-core | 5 | ⚪⚪⚫⚫⚫ Basic |
| 2-swarm-coordination | 25 | ⚪⚪⚪⚪⚫ Advanced |
| 3-security-performance | 15 | ⚪⚪⚪⚫⚫ Intermediate |
| 4-github-repository | 20 | ⚪⚪⚪⚫⚫ Intermediate |
| 5-sparc-methodology | 10 | ⚪⚪⚫⚫⚫ Basic |
| 6-specialized-development | 35 | ⚪⚪⚪⚫⚫ Varies |
| 7-testing-validation | 10 | ⚪⚪⚫⚫⚫ Basic |

Legend: ⚫ = Complexity level (more dots = more complex)

## 🎓 Learning Path

**Beginners**: Start with **1-core** (5 agents)
↓
**Intermediate**: Add **5-sparc-methodology** (10 agents)
↓
**Advanced**: Explore **2-swarm-coordination** (25 agents)
↓
**Specialized**: Use **3, 4, 6, 7** as needed

## 📝 Notes

- All paths relative to `.claude/agents/`
- Agent files are Markdown (.md) format
- YAML frontmatter contains metadata
- Numbers prefix categories for sorting
- Categories are mutually exclusive (no duplicates)

---

**Save this page** for quick reference when working with agents!

Last updated: 2026-01-27
