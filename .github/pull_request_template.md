# Pull Request

## Description

Brief description of changes and why they are needed.

Fixes # (issue number)

## Type of change

Please delete options that are not relevant.

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update
- [ ] Performance improvement
- [ ] Code refactoring (no functional changes)
- [ ] Database migration
- [ ] Configuration change
- [ ] Security enhancement

## Component(s) affected

- [ ] Backend API
- [ ] Frontend Web App
- [ ] Mobile App  
- [ ] Data Pipeline
- [ ] ML/Analytics
- [ ] Database Schema
- [ ] Infrastructure/DevOps
- [ ] Documentation
- [ ] Testing

## Testing

### Test Coverage
- [ ] New tests added for new functionality
- [ ] Existing tests updated for changes
- [ ] All tests pass locally
- [ ] Test coverage maintained/improved

### Manual Testing
Describe the tests you ran to verify your changes:

1. Test A: 
2. Test B:

### Financial/Trading Testing (if applicable)
- [ ] Tested with real market data
- [ ] Verified calculations against known benchmarks
- [ ] Tested edge cases (market holidays, extreme volatility, etc.)
- [ ] Performance tested with large datasets

## Database Changes

- [ ] No database changes
- [ ] Database migration included
- [ ] Migration tested on staging
- [ ] Rollback plan documented
- [ ] Performance impact assessed

## API Changes

- [ ] No API changes
- [ ] Backward compatible changes
- [ ] Breaking changes (version bump required)
- [ ] API documentation updated
- [ ] OpenAPI spec updated

## Security Considerations

- [ ] No security implications
- [ ] Security review completed
- [ ] Input validation added/updated
- [ ] Authentication/authorization updated
- [ ] Secrets/credentials properly managed
- [ ] SQL injection prevention verified
- [ ] XSS prevention verified

## Performance Impact

- [ ] No performance impact
- [ ] Performance improved
- [ ] Potential performance regression (please explain)
- [ ] Load testing completed
- [ ] Memory usage assessed
- [ ] Database query optimization verified

## Deployment Notes

### Environment Variables
List any new environment variables or configuration changes:

```
NEW_VARIABLE=default_value
UPDATED_VARIABLE=new_default
```

### Infrastructure Changes
- [ ] No infrastructure changes required
- [ ] Docker image updates
- [ ] Kubernetes manifests updated
- [ ] Monitoring/alerting updates needed
- [ ] CDN/cache invalidation required

### Third-party Dependencies
- [ ] No new dependencies
- [ ] Dependencies added (list in description)
- [ ] Dependencies updated (security patches)
- [ ] License compatibility verified

## Monitoring and Observability

- [ ] Appropriate logging added
- [ ] Metrics/monitoring updated
- [ ] Error handling improved
- [ ] Health checks updated
- [ ] Grafana dashboards updated

## Documentation

- [ ] README updated
- [ ] API documentation updated
- [ ] Code comments added/updated
- [ ] Architecture documentation updated
- [ ] Deployment guide updated
- [ ] User documentation updated

## Screenshots (if applicable)

### Before
<!-- Add screenshots of the current behavior -->

### After
<!-- Add screenshots of the new behavior -->

## Checklist

### Code Quality
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] My changes generate no new warnings
- [ ] Code is properly formatted (black, prettier)
- [ ] Type annotations added (Python) / TypeScript types defined

### Financial/Business Logic
- [ ] Financial calculations are accurate and tested
- [ ] Edge cases handled (division by zero, missing data, etc.)
- [ ] Data validation implemented
- [ ] Proper error handling for external API failures
- [ ] Rate limiting respected for external APIs

### Security & Compliance
- [ ] No hardcoded secrets or credentials
- [ ] Input validation and sanitization implemented
- [ ] Proper error messages (no sensitive data exposure)
- [ ] GDPR compliance maintained
- [ ] Audit logging added where necessary

### Testing & Quality Assurance
- [ ] Unit tests written and passing
- [ ] Integration tests written and passing
- [ ] End-to-end tests updated (if applicable)
- [ ] Performance tests completed (if applicable)
- [ ] Manual testing completed
- [ ] Edge cases tested

### DevOps & Deployment
- [ ] CI/CD pipeline passes
- [ ] Docker images build successfully
- [ ] Kubernetes manifests valid
- [ ] Database migrations work forward and backward
- [ ] Environment variables documented

## Reviewer Notes

### Areas of Focus
Please pay special attention to:
- 
- 

### Questions for Reviewers
- 
- 

### Breaking Changes
If this is a breaking change, please describe:
- What breaks
- Migration path for users
- Version compatibility

## Risk Assessment

**Risk Level**: Low / Medium / High

**Potential Risks**:
- 
- 

**Mitigation Strategies**:
- 
- 

## Post-Deployment Verification

Steps to verify the deployment was successful:
1. 
2. 
3. 

## Rollback Plan

If this deployment needs to be rolled back:
1. 
2. 
3.