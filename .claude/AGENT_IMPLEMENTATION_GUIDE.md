# Agent Implementation Guide for Investment Analysis Platform

## Overview
This guide explains how to effectively use the consolidated agent set for your investment analysis platform. The agents have been optimized to eliminate duplicates while preserving unique capabilities critical for financial analysis, ML/AI operations, and cost-optimized deployment.

## Quick Start

### 1. Primary Financial Analysis Workflow
```bash
# For comprehensive stock analysis
Use agents: trading_framework + quant_analyst + ml_engineer

# For portfolio risk assessment  
Use agents: risk_manager + portfolio_manager

# For regulatory compliance
Use agents: fintech_engineer + security_auditor
```

### 2. Data Pipeline Development
```bash
# For ETL pipeline creation
Use agents: data_engineer + data_science_architect

# For ML model deployment
Use agents: ml_engineer + performance_engineer

# For database optimization
Use agents: database_optimizer
```

### 3. Cost Optimization (Under $50/month)
```bash
# For infrastructure cost reduction
Use agents: cloud_architect + performance_engineer

# For query optimization
Use agents: database_optimizer + performance_engineer
```

## Agent Capabilities by Category

### Financial Core Agents

#### TradingAgents Framework
- **Location**: `.claude/agents/TradingAgents/`
- **Purpose**: Multi-agent financial analysis with agent debates
- **Use When**: 
  - Analyzing stocks for buy/sell recommendations
  - Generating comprehensive market analysis
  - Requiring multiple perspectives (bull vs bear)
- **Example Task**: "Analyze AAPL stock and provide investment recommendation"

#### Quant Analyst (wshobson)
- **Location**: `.claude/agents/wshobson-agents/quant-analyst.md`
- **Model**: Claude-3-opus (for complex calculations)
- **Use When**:
  - Building financial models
  - Backtesting strategies
  - Calculating R-multiples and expectancy
- **Example Task**: "Create a momentum trading strategy with risk metrics"

#### Risk Manager (wshobson)
- **Location**: `.claude/agents/wshobson-agents/risk-manager.md`
- **Model**: Claude-3-sonnet
- **Use When**:
  - Portfolio risk assessment
  - Position sizing calculations
  - Value at Risk (VaR) analysis
- **Example Task**: "Calculate optimal position sizes for portfolio"

#### Fintech Engineer (voltagent)
- **Location**: `.claude/agents/voltagent-subagents/fintech-engineer.md`
- **Use When**:
  - Implementing SEC compliance features
  - GDPR data protection
  - Payment processing integration
- **Example Task**: "Implement SEC-compliant audit logging"

### Data & ML Pipeline Agents

#### Data Science Architect
- **Location**: `.claude/agents/data-science-architect.md`
- **Use When**:
  - Designing ETL pipelines
  - Optimizing data workflows
  - Statistical analysis architecture
- **Example Task**: "Design efficient data pipeline for 6000+ stocks"

#### ML Engineer (wshobson)
- **Location**: `.claude/agents/wshobson-agents/ml-engineer.md`
- **Model**: Claude-3-opus
- **Use When**:
  - Deploying PyTorch/scikit-learn models
  - Feature engineering
  - Model optimization
- **Example Task**: "Deploy FinBERT for sentiment analysis"

#### Data Engineer (wshobson)
- **Location**: `.claude/agents/wshobson-agents/data-engineer.md`
- **Use When**:
  - Creating Airflow DAGs
  - Setting up Kafka streams
  - Data processing pipelines
- **Example Task**: "Create Airflow DAG for daily stock data processing"

### Infrastructure & Performance Agents

#### Cloud Architect (wshobson)
- **Location**: `.claude/agents/wshobson-agents/cloud-architect.md`
- **Critical Role**: Keeping costs under $50/month
- **Use When**:
  - AWS infrastructure setup
  - Cost optimization
  - Auto-scaling configuration
- **Example Task**: "Optimize AWS deployment to stay under $50/month"

#### Performance Engineer (wshobson)
- **Location**: `.claude/agents/wshobson-agents/performance-engineer.md`
- **Use When**:
  - Application optimization
  - Redis caching strategies
  - Resource management
- **Example Task**: "Implement caching for API responses"

### Development Support Agents

#### Python Pro (wshobson)
- **Location**: `.claude/agents/wshobson-agents/python-pro.md`
- **Use When**:
  - FastAPI development
  - Python optimization
  - Backend implementation
- **Example Task**: "Implement FastAPI endpoints for stock analysis"

#### Godmode Refactorer
- **Location**: `.claude/agents/godmode-refactorer.md`
- **Unique Capability**: Can refactor across ALL technologies
- **Use When**:
  - Major architectural changes
  - Cross-technology refactoring
  - Complex modernization
- **Example Task**: "Refactor entire codebase for microservices architecture"

## Command Usage

### Analyze Codebase Command
```bash
# Location: .claude/commands/analyze_codebase.md
# Purpose: Comprehensive project analysis

# Usage:
"Analyze the current project status and deployment readiness"
# This will create reports in .context/ directory:
# - overall_project_status.md
# - feature_checklist.md
# - identified_issues.md
# - recommendations.md
# - deployment_readiness.md
```

### UI Design Commands
```bash
# Location: .claude/commands/ui_design.md

# Available commands:
- interpret_ui: Convert UI designs to code
- analyze_ui: Evaluate design for accessibility
- compare_designs: Compare and migrate designs
- generate_ui_components: Create components from specs
```

## Workflow Examples

### Example 1: Implementing Stock Analysis Feature
```
1. Start with agent_organizer to coordinate
2. Use trading_framework for analysis logic
3. Use ml_engineer for predictive models
4. Use backend_architect for API design
5. Use react_pro for dashboard
6. Use test_automator for validation
7. Use code_reviewer for quality check
```

### Example 2: Cost Optimization Task
```
1. Use cloud_architect to analyze AWS costs
2. Use performance_engineer for application optimization
3. Use database_optimizer for query efficiency
4. Use data_science_architect for pipeline optimization
```

### Example 3: Regulatory Compliance Implementation
```
1. Use fintech_engineer for compliance requirements
2. Use security_auditor for vulnerability assessment
3. Use api_documenter for audit trail documentation
4. Use test_automator for compliance testing
```

## Best Practices

### 1. Agent Selection
- Start with `agent_organizer` for complex multi-step tasks
- Use `trading_framework` as primary for all financial analysis
- Always include `security_auditor` for production changes
- Use `cloud_architect` for any infrastructure changes to maintain cost limits

### 2. Cost Control
- Always consult `cloud_architect` before adding new services
- Use `performance_engineer` to optimize before scaling
- Implement caching strategies with `performance_engineer`

### 3. Quality Assurance
- Always use `code_reviewer` before production deployment
- Use `test_automator` for comprehensive testing
- Use `security_auditor` for compliance verification

### 4. Documentation
- Use `api_documenter` for all API changes
- Maintain CLAUDE.md with project-specific guidance
- Update consolidated-agent-config.json as needed

## Troubleshooting

### Issue: Agents not responding as expected
**Solution**: Check agent-config.json trigger keywords and update if needed

### Issue: High AWS costs
**Solution**: Immediately use cloud_architect + performance_engineer

### Issue: Slow queries
**Solution**: Use database_optimizer with performance_engineer

### Issue: Compliance concerns
**Solution**: Use fintech_engineer + security_auditor

## Maintenance

### Regular Reviews
- Weekly: Review with performance_engineer for optimization opportunities
- Monthly: Cost review with cloud_architect
- Quarterly: Security audit with security_auditor
- Ongoing: Code quality with code_reviewer

### Agent Updates
- Check for updates in source repositories
- Test new agents in development before production
- Maintain backwards compatibility

## Conclusion

This consolidated agent set provides comprehensive coverage for your investment analysis platform while eliminating redundancy. The focus on financial expertise, cost optimization, and regulatory compliance ensures the platform meets its goals of analyzing 6000+ stocks daily while staying under $50/month operational costs.