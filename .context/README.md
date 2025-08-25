# Context Management System

## Overview

This directory contains the context management system for the Investment Analysis Platform. The context manager maintains a comprehensive knowledge graph of the project structure, configurations, and operational state to enable seamless collaboration between distributed agents.

## Architecture

The context management system implements a 3-layer architecture:

1. **L1 Cache**: In-memory context storage (< 100ms retrieval)
2. **L2 Cache**: Redis-based distributed cache (< 200ms retrieval)  
3. **L3 Storage**: File-based persistent storage with database backup

## Key Files

### `context-manager.json`
The main context knowledge graph containing:
- Complete project structure and metadata
- API endpoint definitions and capabilities
- Database schema and relationships
- Caching architecture and policies
- External API configurations
- ML model specifications
- Deployment configurations
- Security and compliance features
- Cost optimization strategies
- Performance benchmarks

### `sync-config.json`
Synchronization system configuration including:
- File watching patterns and priorities
- Sync triggers and frequencies
- Consistency checks and validation
- Conflict resolution strategies
- Performance optimization settings
- Agent-specific synchronization preferences

## Context Access Patterns

### Fast Retrieval Queries
```json
{
  "context_type": "project_structure",
  "filter": {"directory": "backend/"},
  "cache_level": "l1"
}
```

### Comprehensive Analysis
```json
{
  "context_type": "system_architecture", 
  "include": ["api_endpoints", "database_schema", "caching_layers"],
  "format": "detailed"
}
```

### Agent-Specific Context
```json
{
  "requesting_agent": "workflow_orchestrator",
  "context_scope": "process_definitions",
  "update_frequency": "real_time"
}
```

## Performance Targets

- **Retrieval Time**: < 100ms for cached contexts
- **Cache Hit Rate**: > 80%
- **Consistency Score**: 100% (strong consistency)
- **Availability**: > 99.9%

## Synchronization Triggers

1. **File Changes**: Real-time sync for critical files
2. **Git Commits**: Post-commit hooks for version tracking
3. **Deployment Events**: Environment-specific updates
4. **Agent Requests**: On-demand context retrieval
5. **Scheduled Sync**: Full synchronization every 4 hours

## Data Lifecycle

- **Real-time Context**: Immediate updates for critical changes
- **Batch Context**: Accumulated updates for non-critical changes
- **Historical Context**: Version tracking with 30-day retention
- **Archive Context**: Long-term storage for compliance

## Security and Compliance

- **Access Control**: Agent-based permissions
- **Audit Trail**: Complete operation logging
- **Data Privacy**: GDPR-compliant data handling
- **Encryption**: Optional AES-256-GCM encryption
- **Backup**: Automated daily backups with 30-day retention

## Integration Points

The context manager integrates with all platform agents:

- **Agent Organizer**: Project structure and component metadata
- **Multi-Agent Coordinator**: System state and agent status
- **Workflow Orchestrator**: Process definitions and execution logs
- **Task Distributor**: Workload metrics and resource allocation
- **Performance Monitor**: Performance metrics and alerts
- **Error Coordinator**: Error contexts and recovery procedures
- **Knowledge Synthesizer**: Insights and knowledge graphs

## Usage Examples

### Get Project Structure
```bash
curl -X GET "http://localhost:8000/api/context/get/project_structure"
```

### Update Component Status
```bash
curl -X POST "http://localhost:8000/api/context/update/component_status" \
  -H "Content-Type: application/json" \
  -d '{"component": "backend", "status": "healthy", "timestamp": "2025-08-19T00:00:00Z"}'
```

### Search Contexts
```bash
curl -X GET "http://localhost:8000/api/context/search?q=caching&type=configuration"
```

### Trigger Sync
```bash
curl -X POST "http://localhost:8000/api/context/sync/trigger" \
  -H "Content-Type: application/json" \
  -d '{"scope": "full", "priority": "high"}'
```

## Monitoring and Metrics

The context manager exports Prometheus metrics:

- `context_retrieval_duration_seconds`: Histogram of retrieval times
- `context_cache_hit_rate`: Cache hit percentage
- `context_sync_operations_total`: Counter of sync operations
- `context_conflicts_total`: Counter of sync conflicts
- `context_size_bytes`: Current context storage size

## Best Practices

1. **Cache Locality**: Keep frequently accessed contexts in L1 cache
2. **Batch Updates**: Group related updates for efficiency
3. **Conflict Avoidance**: Use atomic operations for critical updates
4. **Performance Monitoring**: Track retrieval times and hit rates
5. **Consistent Schemas**: Validate context structure before updates
6. **Error Handling**: Implement graceful degradation for cache misses

## Troubleshooting

### Common Issues

**Slow Retrieval Times**
- Check cache hit rates
- Verify L1 cache size limits
- Monitor memory usage

**Sync Conflicts**
- Review conflict resolution strategy
- Check file change patterns
- Validate agent update sequences

**Missing Contexts**
- Verify sync trigger configuration
- Check file watching patterns
- Review agent permission settings

**High Memory Usage**
- Adjust L1 cache size limits
- Enable context compression
- Review data retention policies

## Configuration Updates

To modify context management behavior:

1. Update `sync-config.json` for synchronization settings
2. Update `context-manager.json` for structural changes
3. Restart context management services
4. Verify changes with health checks

## Development

For local development and testing:

```bash
# Start context management services
docker-compose up -d redis

# Test context retrieval
python scripts/test_context_manager.py

# Monitor sync operations
tail -f logs/context-manager.log
```

This context management system ensures fast, consistent, and secure access to project information across all distributed agents while maintaining operational efficiency under the $50/month budget constraint.