# Agent Invocation Examples

## Investment Platform Multi-Agent Coordination Examples

This document provides 15+ comprehensive examples showing proper multi-agent coordination, parallel vs sequential execution patterns, and synthesis of multi-agent outputs for the investment analysis platform.

## Simple Tasks (Complexity 1-3)

### Example 1: Single Agent Direct Invocation
**User Request**: "Fix the typo in the stock analysis documentation"

**Agent Selection**: 
- Primary: `documentation-generator@claude-code`
- Complexity: 2/10 (Simple)
- Execution: Direct single-agent task

**Response Pattern**:
```yaml
invocation:
  agent: documentation-generator@claude-code
  task_type: content_correction
  parallel_execution: false
  synthesis_required: false
  estimated_duration: 5_minutes
```

### Example 2: Simple Code Fix
**User Request**: "Update the API endpoint URL in the configuration"

**Agent Selection**:
- Primary: `api-designer@claude-code`
- Complexity: 2/10 (Simple)
- Support: `code-reviewer@claude-code` (auto-triggered for code changes)

**Response Pattern**:
```yaml
invocation:
  primary_agent: api-designer@claude-code
  supporting_agents:
    - code-reviewer@claude-code
  execution_flow: sequential
  steps:
    1. api-designer@claude-code: Update configuration
    2. code-reviewer@claude-code: Review change
  synthesis_required: false
```

## Moderate Tasks (Complexity 4-6)

### Example 3: Database Query Optimization
**User Request**: "Optimize the slow PostgreSQL query for stock price retrieval"

**Agent Selection**:
- **Team**: Database Team
- **Lead**: `postgres-pro@lst97`
- **Supporting**: `database-optimizer@wshobson`, `performance-optimizer@claude-code`
- **Complexity**: 5/10 (Moderate)

**Execution Plan**:
```yaml
team_coordination:
  lead_agent: postgres-pro@lst97
  protocol: schema_first_optimization_driven
  parallel_phases:
    phase_1: 
      - postgres-pro@lst97: Analyze current query structure
      - database-optimizer@wshobson: Identify bottlenecks
    phase_2:
      - postgres-pro@lst97: Design optimized query
      - performance-optimizer@claude-code: Test performance impact
  synthesis_point: Lead agent consolidates optimization strategy
  estimated_duration: 45_minutes
```

### Example 4: Financial Model Implementation
**User Request**: "Implement Sharpe ratio calculation for portfolio optimization"

**Agent Selection**:
- **Team**: Financial Analysis Team
- **Lead**: `quant-analyst@wshobson`
- **Core**: `risk-manager@wshobson`, `data-scientist@wshobson`
- **Complexity**: 6/10 (Moderate)

**Execution Plan**:
```yaml
team_coordination:
  protocol: domain_expert_led_collaboration
  execution_flow:
    1. Expert Analysis (quant-analyst@wshobson):
       - Define Sharpe ratio requirements for platform
       - Specify input data requirements
    2. Specialist Consultation (parallel):
       - risk-manager@wshobson: Validate risk-free rate methodology
       - data-scientist@wshobson: Design statistical validation tests
    3. Collaborative Implementation:
       - quant-analyst@wshobson: Core algorithm implementation
       - data-scientist@wshobson: Testing and validation
    4. Expert Validation:
       - Final review by quant-analyst@wshobson
  synthesis: Lead expert consolidates final implementation
```

## Complex Tasks (Complexity 7-8)

### Example 5: Real-time WebSocket API Implementation
**User Request**: "Implement real-time stock price streaming with WebSocket API and caching"

**Agent Selection**:
- **Teams**: API Integration Team + Backend Development Team + Performance Team
- **Complexity**: 8/10 (Complex - Multi-team coordination)
- **Meta-Coordinator**: `agent-organizer@lst97`

**Execution Plan**:
```yaml
multi_team_coordination:
  meta_coordinator: agent-organizer@lst97
  teams:
    api_integration_team:
      lead: api-designer@claude-code
      mission: WebSocket endpoint design and documentation
    backend_development_team:
      lead: python-pro@wshobson
      mission: Real-time data processing and WebSocket handling
    performance_team:
      lead: performance-optimizer@claude-code
      mission: Caching strategy and performance optimization

  execution_phases:
    phase_1_parallel_design:
      - api-designer@claude-code: WebSocket API specification
      - python-pro@wshobson: Backend architecture design
      - performance-optimizer@claude-code: Caching strategy design
    
    synthesis_checkpoint_1:
      coordinator: agent-organizer@lst97
      action: Integrate designs, resolve conflicts, validate compatibility

    phase_2_parallel_implementation:
      - python-backend-engineer@awesome: WebSocket server implementation
      - fastapi-expert@furai: FastAPI WebSocket integration
      - redis-expert@furai: Real-time caching implementation
      - api-documenter@lst97: API documentation updates

    synthesis_checkpoint_2:
      coordinator: agent-organizer@lst97
      action: Integration testing coordination

    phase_3_validation:
      - performance-optimizer@claude-code: Performance testing
      - security-auditor@wshobson: Security validation
      - test-suite-generator@claude-code: Test suite creation

  final_synthesis:
    coordinator: agent-organizer@lst97
    deliverables:
      - Integrated WebSocket API
      - Performance benchmarks
      - Security validation report
      - Complete documentation
      - Test coverage report
```

### Example 6: ML Model Training Pipeline
**User Request**: "Create ML model training pipeline for stock prediction with automated retraining"

**Agent Selection**:
- **Teams**: ML/AI Team + Data Pipeline Team + DevOps Team
- **Complexity**: 8/10 (Complex - Multi-domain expertise)
- **Coordination**: Collaborative research and development + Infrastructure as code

**Execution Plan**:
```yaml
multi_team_coordination:
  primary_coordinator: ml-engineer@wshobson
  supporting_coordinator: agent-organizer@lst97

  parallel_research_phase:
    ml_ai_team:
      - ml-engineer@wshobson: Research ensemble methods for stock prediction
      - pytorch-expert@furai: Investigate neural network architectures
      - ai-engineer@lst97: Design automated retraining triggers
    data_pipeline_team:
      - data-engineer@lst97: Design feature pipeline architecture
      - data-scientist@wshobson: Define feature engineering requirements
    devops_team:
      - deployment-engineer@lst97: Container orchestration for ML workloads
      - kubernetes-expert@furai: K8s job scheduling for training

  collaborative_design_phase:
    coordinator: ml-engineer@wshobson
    participants: [data-engineer@lst97, deployment-engineer@lst97, ai-engineer@lst97]
    deliverable: Unified ML pipeline architecture

  parallel_implementation_phase:
    model_development:
      - ml-engineer@wshobson: Core model training logic
      - pytorch-expert@furai: Neural network implementation
      - python-pro@wshobson: Model evaluation metrics
    
    data_infrastructure:
      - data-engineer@lst97: Airflow DAG for feature preparation
      - kafka-expert@furai: Real-time feature streaming
      - postgres-pro@lst97: Model metadata storage

    deployment_infrastructure:
      - deployment-engineer@lst97: MLOps pipeline setup
      - kubernetes-expert@furai: Training job orchestration
      - github-actions-expert@furai: CI/CD for model deployment

  validation_synthesis:
    coordinator: agent-organizer@lst97
    validation_teams:
      - ML/AI Team: Model accuracy and performance validation
      - Data Pipeline Team: Data flow and feature quality validation  
      - DevOps Team: Infrastructure reliability validation
      - Testing Team: End-to-end pipeline testing

  final_integration:
    deliverables:
      - Production-ready ML training pipeline
      - Automated retraining system
      - Model performance monitoring
      - Infrastructure documentation
      - Comprehensive test coverage
```

## Enterprise Tasks (Complexity 9-10)

### Example 7: Complete Authentication System Implementation
**User Request**: "Implement complete authentication system with OAuth2, JWT, role-based access control, and SEC compliance"

**Agent Selection**:
- **Teams**: Security Team + Backend Development Team + API Integration Team + Documentation Team
- **Complexity**: 9/10 (Enterprise - Full system implementation)
- **Meta-Coordination**: `agent-organizer@lst97`

**Execution Plan**:
```yaml
enterprise_coordination:
  meta_coordinator: agent-organizer@lst97
  complexity_level: enterprise
  estimated_duration: 2_weeks

  team_assignments:
    security_team:
      lead: security-auditor@wshobson
      mission: Security architecture, compliance, vulnerability assessment
      agents: [security-auditor@lst97, penetration-tester@voltagent, jwt-expert@furai]
    
    backend_development_team:
      lead: python-pro@wshobson
      mission: Authentication service implementation
      agents: [backend-architect@lst97, python-backend-engineer@awesome]
    
    api_integration_team:
      lead: api-designer@claude-code
      mission: Authentication API endpoints and integration
      agents: [api-designer@voltagent, fastapi-expert@furai]
    
    documentation_team:
      lead: api-documenter@lst97
      mission: Security documentation and compliance documentation
      agents: [documentation-generator@claude-code, technical-writer@voltagent]

  execution_protocol: security_first_compliance_driven

  phase_1_architecture_design:
    lead_coordination: security-auditor@wshobson
    parallel_activities:
      - security-auditor@wshobson: Security architecture design
      - backend-architect@lst97: Service architecture design
      - api-designer@claude-code: API security design
      - jwt-expert@furai: Token management architecture
    synthesis: Unified security architecture document

  phase_2_compliance_validation:
    lead_coordination: security-auditor@wshobson
    activities:
      - security-auditor@lst97: SEC compliance requirements mapping
      - penetration-tester@voltagent: Security threat modeling
      - api-designer@voltagent: API security standards validation
    synthesis: Compliance requirements specification

  phase_3_parallel_implementation:
    backend_implementation:
      - python-pro@wshobson: Core authentication service
      - python-backend-engineer@awesome: OAuth2 provider integration
      - fastapi-expert@furai: FastAPI security middleware
    
    api_implementation:
      - api-designer@claude-code: Authentication endpoints
      - jwt-expert@furai: JWT token management
      - api-designer@voltagent: Role-based access control API
    
    security_implementation:
      - security-auditor@wshobson: Security monitoring integration
      - oauth-oidc-expert@furai: OpenID Connect implementation
      - keycloak-expert@furai: Identity provider setup

  phase_4_integration_testing:
    coordinator: agent-organizer@lst97
    parallel_validation:
      - security_team: Security penetration testing
      - backend_team: Service integration testing
      - api_team: API endpoint testing
      - testing_team: End-to-end authentication flow testing

  phase_5_documentation_compliance:
    lead: api-documenter@lst97
    parallel_documentation:
      - technical-writer@voltagent: Technical documentation
      - security-auditor@wshobson: Security compliance documentation
      - api-documenter@lst97: API documentation updates
      - documentation-generator@claude-code: User guides

  final_synthesis:
    coordinator: agent-organizer@lst97
    validation_criteria:
      - Security: Penetration testing passed
      - Compliance: SEC requirements validated
      - Functionality: All authentication flows working
      - Documentation: Complete technical and compliance docs
      - Performance: Authentication latency < 100ms
    
    deliverables:
      - Production authentication system
      - Security compliance report
      - Complete API documentation
      - Deployment guides
      - Monitoring dashboards
```

## Parallel Execution Patterns

### Pattern 1: Independent Component Development
**Use Case**: Building multiple independent features simultaneously

```yaml
parallel_pattern: independent_components
example: "Implement portfolio dashboard and risk analytics simultaneously"

team_1: frontend_team
  task: Portfolio dashboard React components
  agents: [frontend-developer@lst97, react-pro@lst97, ui-designer@lst97]
  
team_2: financial_analysis_team  
  task: Risk analytics engine
  agents: [quant-analyst@wshobson, risk-manager@wshobson]

coordination:
  synchronization_points: [requirements_review, integration_testing]
  independence_level: high
  synthesis_complexity: low
```

### Pattern 2: Layered Development
**Use Case**: Building system layers in parallel with dependencies

```yaml
parallel_pattern: layered_development
example: "Database layer, API layer, and frontend layer development"

layer_1: database_team
  dependencies: []
  agents: [postgres-pro@lst97, database-optimizer@wshobson]
  
layer_2: backend_development_team
  dependencies: [database_layer_interfaces]
  agents: [python-pro@wshobson, fastapi-expert@furai]
  
layer_3: frontend_team
  dependencies: [api_specifications]
  agents: [frontend-developer@lst97, react-pro@lst97]

coordination:
  handoff_protocol: interface_specification_driven
  dependency_management: automated_interface_validation
```

## Synthesis Patterns

### Pattern 1: Technical Integration Synthesis
**Process**: Combining outputs from multiple technical teams

```yaml
synthesis_pattern: technical_integration
coordinator: system-architect@claude-code
inputs:
  - database_schema: from database_team
  - api_specification: from api_integration_team  
  - frontend_components: from frontend_team
  - security_requirements: from security_team

synthesis_process:
  1. Compatibility verification
  2. Interface alignment
  3. Performance impact analysis
  4. Security validation
  5. Integration testing coordination

output: unified_system_architecture
quality_gates: [performance_benchmarks, security_validation, integration_tests]
```

### Pattern 2: Domain Expert Synthesis  
**Process**: Combining specialized domain knowledge

```yaml
synthesis_pattern: domain_expert_synthesis
coordinator: quant-analyst@wshobson
inputs:
  - market_analysis: from financial_analysis_team
  - ml_predictions: from ml_ai_team
  - risk_assessments: from risk_management_specialists
  - performance_data: from performance_team

synthesis_process:
  1. Domain expertise validation
  2. Methodology consistency check
  3. Result correlation analysis  
  4. Confidence interval calculation
  5. Recommendation generation

output: investment_recommendation_report
validation: peer_review_by_risk_manager
```

## Quality Assurance Integration Examples

### Example 8: Automated Quality Gates
**Trigger**: Any code change in the investment platform

**Agent Cascade**:
```yaml
quality_assurance_cascade:
  primary_change_agent: [varies by domain]
  automatic_triggers:
    code_review:
      agent: code-reviewer@claude-code
      conditions: [any_code_change]
      parallel_with_primary: true
    
    security_review:
      agent: security-auditor@wshobson
      conditions: [api_changes, authentication_changes, data_access_changes]
      dependencies: [code_review_complete]
    
    performance_validation:
      agent: performance-optimizer@claude-code  
      conditions: [database_changes, api_changes, ml_model_changes]
      parallel_with: [security_review]
    
    documentation_update:
      agent: api-documenter@lst97
      conditions: [api_changes, new_features]
      dependencies: [primary_implementation_complete]

  synthesis_coordination:
    coordinator: code-reviewer@claude-code
    final_validation: all_quality_gates_passed
```

## Specialized Financial Platform Examples

### Example 9: Complete Trading Algorithm Implementation
**User Request**: "Implement momentum-based trading algorithm with backtesting and risk management"

**Agent Selection**:
- **Teams**: Financial Analysis Team + ML/AI Team + Backend Development Team + Testing Team
- **Complexity**: 8/10 (Complex financial domain)

**Execution Plan**:
```yaml
financial_algorithm_development:
  domain_lead: quant-analyst@wshobson
  supporting_coordinator: agent-organizer@lst97

  phase_1_financial_design:
    lead: quant-analyst@wshobson
    parallel_research:
      - quant-analyst@wshobson: Momentum indicator research and strategy design
      - risk-manager@wshobson: Risk management parameters and stop-loss rules
      - data-scientist@wshobson: Historical data requirements and backtesting methodology
    synthesis: Trading algorithm specification

  phase_2_implementation:
    parallel_development:
      algorithm_implementation:
        - python-pro@wshobson: Core trading logic implementation
        - quant-analyst@voltagent: Portfolio optimization integration
        - risk-manager@wshobson: Risk management module
      
      ml_enhancement:
        - ml-engineer@wshobson: ML model for signal enhancement
        - data-scientist@wshobson: Feature engineering for momentum indicators
        - pytorch-expert@furai: Neural network for pattern recognition
      
      infrastructure:
        - backend-architect@lst97: Trading system architecture
        - fastapi-expert@furai: Trading API endpoints
        - performance-optimizer@claude-code: High-frequency trading optimization

  phase_3_backtesting_validation:
    coordinator: quant-analyst@wshobson
    parallel_validation:
      - data-scientist@wshobson: Statistical backtesting analysis
      - risk-manager@wshobson: Risk metrics validation
      - performance-engineer@lst97: Performance benchmarking
      - test-automator@lst97: Automated test suite for edge cases

  phase_4_integration:
    coordinator: agent-organizer@lst97
    integration_tasks:
      - Backend integration with existing portfolio system
      - Real-time data feed integration
      - Risk monitoring dashboard integration
      - Compliance audit trail implementation

  final_synthesis:
    deliverables:
      - Production trading algorithm
      - Comprehensive backtesting report  
      - Risk management validation
      - Performance benchmarks
      - Regulatory compliance documentation
```

### Example 10: Comprehensive Market Data Pipeline
**User Request**: "Build complete market data pipeline from multiple sources with real-time processing and caching"

**Agent Selection**:
- **Teams**: Data Pipeline Team + Performance Team + DevOps Team + Backend Development Team  
- **Complexity**: 9/10 (Enterprise data infrastructure)

**Execution Plan**:
```yaml
enterprise_data_pipeline:
  project_coordinator: agent-organizer@lst97
  technical_lead: data-engineer@lst97

  phase_1_architecture_design:
    parallel_design:
      data_architecture:
        - data-engineer@lst97: Overall data pipeline architecture
        - kafka-expert@furai: Real-time streaming architecture  
        - data-engineer@voltagent: Multi-source integration strategy
      
      performance_architecture:
        - performance-optimizer@claude-code: Caching and optimization strategy
        - redis-expert@furai: Real-time caching design
        - database-optimizer@wshobson: Data storage optimization
      
      infrastructure_architecture:
        - deployment-engineer@lst97: Container orchestration design
        - cloud-architect@wshobson: Scalable cloud infrastructure
        - kubernetes-expert@furai: Auto-scaling configuration

    synthesis: Unified data pipeline architecture

  phase_2_source_integration:
    parallel_api_clients:
      - python-pro@wshobson: Alpha Vantage client with rate limiting
      - fastapi-expert@furai: Finnhub real-time client  
      - api-designer@claude-code: Polygon.io integration
      - data-scientist@wshobson: SEC EDGAR data extraction
    
    coordination: data-engineer@lst97
    validation: Unified data format and quality standards

  phase_3_processing_pipeline:
    stream_processing:
      - kafka-expert@furai: Real-time data streaming setup
      - python-pro@wshobson: Data transformation and cleaning
      - performance-optimizer@claude-code: Processing optimization
    
    batch_processing:
      - data-engineer@lst97: Airflow DAG implementation
      - data-engineer@voltagent: Historical data processing
      - postgres-pro@lst97: Batch data storage optimization

  phase_4_caching_performance:
    coordinator: performance-optimizer@claude-code
    parallel_implementation:
      - redis-expert@furai: Multi-tier caching implementation
      - performance-engineer@lst97: Performance monitoring setup
      - database-optimizer@wshobson: Database query optimization
      - cloud-architect@wshobson: Auto-scaling configuration

  phase_5_monitoring_deployment:
    coordinator: deployment-engineer@lst97
    parallel_setup:
      - devops-engineer@voltagent: CI/CD pipeline setup
      - kubernetes-expert@furai: Production deployment
      - monitoring-specialist: Real-time monitoring dashboards
      - security-auditor@wshobson: Data security validation

  final_integration:
    coordinator: agent-organizer@lst97
    validation_criteria:
      - Data quality: 99.9% accuracy
      - Performance: <100ms API response time
      - Reliability: 99.9% uptime
      - Scalability: Handle 10,000 req/sec
      - Cost: Stay within $50/month budget
```

These examples demonstrate the sophisticated multi-agent coordination system that leverages all 397 specialized agents across the investment platform development lifecycle, ensuring maximum efficiency through parallel execution and expert specialization.