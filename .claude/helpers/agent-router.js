#!/usr/bin/env node
/**
 * Investment Platform Agent Router
 *
 * Provides comprehensive command and keyword routing to the appropriate agents.
 * Supports investment-specific domain routing, pattern matching, and fallback handling.
 *
 * Usage:
 *   const { routeTask, getAgentConfig } = require('./agent-router');
 *   const result = routeTask("Analyze the credit profile for Company XYZ");
 *   // Returns: { agent: 'deal-underwriter', model: 'opus', confidence: 0.85, ... }
 */

const fs = require('fs');
const path = require('path');

// Load agent registry if available
let REGISTRY = null;
const registryPath = path.join(__dirname, '..', 'config', 'agent-registry.json');
try {
  if (fs.existsSync(registryPath)) {
    REGISTRY = JSON.parse(fs.readFileSync(registryPath, 'utf8'));
  }
} catch (e) {
  console.error('[Router] Warning: Could not load agent registry:', e.message);
}

// Explicit command to agent mappings
const COMMAND_MAPPINGS = {
  // Investment commands
  '/underwrite': { agent: 'deal-underwriter', model: 'opus', domain: 'investment' },
  '/model': { agent: 'financial-modeler', model: 'opus', domain: 'investment' },
  '/analyze-structure': { agent: 'deal-underwriter', model: 'opus', domain: 'investment' },
  '/scenario': { agent: 'financial-modeler', model: 'opus', domain: 'investment' },
  '/stock-analysis': { agent: 'financial-analysis-swarm', model: 'opus', domain: 'investment' },

  // Development commands
  '/plan': { agent: 'planner', model: 'opus', domain: 'development' },
  '/build-fix': { agent: 'build-error-resolver', model: 'opus', domain: 'development' },
  '/code-review': { agent: 'code-reviewer', model: 'opus', domain: 'development' },
  '/tdd': { agent: 'tdd-guide', model: 'opus', domain: 'development' },
  '/e2e': { agent: 'e2e-runner', model: 'opus', domain: 'development' },
  '/refactor-clean': { agent: 'refactor-cleaner', model: 'opus', domain: 'development' },
  '/update-docs': { agent: 'doc-updater', model: 'opus', domain: 'development' },
  '/verify': { agent: 'project-quality-swarm', model: 'opus', domain: 'development' },

  // Infrastructure commands
  '/docker': { agent: 'infrastructure-devops-swarm', model: 'opus', domain: 'infrastructure' },
  '/deploy': { agent: 'infrastructure-devops-swarm', model: 'opus', domain: 'infrastructure' },
  '/monitoring': { agent: 'infrastructure-devops-swarm', model: 'opus', domain: 'infrastructure' },

  // GitHub commands
  '/github-swarm': { agent: 'github-swarm-coordinator', model: 'opus', domain: 'github' },

  // SPARC commands
  '/sparc': { agent: 'sparc-coordinator', model: 'opus', domain: 'methodology' },
  '/sparc-architect': { agent: 'architect', model: 'opus', domain: 'methodology' },
  '/sparc-code': { agent: 'coder', model: 'sonnet', domain: 'methodology' }
};

// Domain patterns for keyword-based routing
const DOMAIN_PATTERNS = {
  'investment-underwriting': {
    pattern: /underwrite|credit\s*(analysis|score|risk)|loan|collateral|lien|ucc|security\s*package|covenant|intercreditor/i,
    agents: ['deal-underwriter'],
    model: 'opus',
    priority: 1
  },
  'financial-modeling': {
    pattern: /dcf|discounted\s*cash\s*flow|lbo|leveraged\s*buyout|valuation|financial\s*model|projection|scenario\s*analysis|sensitivity\s*analysis|irr|npv|wacc/i,
    agents: ['financial-modeler'],
    model: 'opus',
    priority: 1
  },
  'portfolio-risk': {
    pattern: /portfolio|risk\s*(assessment|analysis|metrics)|var\b|value\s*at\s*risk|sharpe|sortino|beta|drawdown|volatility|correlation|diversification/i,
    agents: ['risk-assessor', 'portfolio-manager', 'financial-analysis-swarm'],
    model: 'opus',
    priority: 1
  },
  'stock-analysis': {
    pattern: /stock\s*(analysis|recommendation)|market\s*(analysis|research)|equity|fundamental|technical\s*analysis|pe\s*ratio|earnings/i,
    agents: ['financial-analysis-swarm', 'investment-analyst'],
    model: 'opus',
    priority: 1
  },
  'investment-memo': {
    pattern: /investment\s*(memo|thesis|analysis)|deal\s*(analysis|evaluation)|due\s*diligence|market\s*opportunity/i,
    agents: ['investment-analyst'],
    model: 'opus',
    priority: 1
  },
  'infrastructure': {
    pattern: /docker|container|kubernetes|k8s|deploy|deployment|ci\/?cd|github\s*actions|pipeline|prometheus|grafana|alertmanager|monitoring|health\s*check/i,
    agents: ['infrastructure-devops-swarm'],
    model: 'opus',
    priority: 2
  },
  'backend': {
    pattern: /api\s*(endpoint|development)|fastapi|rest\s*api|graphql|database|postgres|timescaledb|redis|celery|worker|queue|websocket/i,
    agents: ['backend-api-swarm'],
    model: 'opus',
    priority: 2
  },
  'frontend': {
    pattern: /react|component|dashboard|chart|visualization|ui\s*(design|component)|ux|material[\s-]?ui|plotly|frontend/i,
    agents: ['ui-visualization-swarm'],
    model: 'opus',
    priority: 2
  },
  'data-ml': {
    pattern: /etl|data\s*pipeline|airflow|dag|machine\s*learning|ml\s*model|training|prophet|xgboost|feature\s*engineering|prediction/i,
    agents: ['data-ml-pipeline-swarm', 'data-science-architect'],
    model: 'opus',
    priority: 2
  },
  'testing': {
    pattern: /test|testing|coverage|pytest|jest|e2e|end[\s-]?to[\s-]?end|integration\s*test|unit\s*test|quality\s*assurance/i,
    agents: ['project-quality-swarm', 'tdd-guide', 'e2e-runner', 'tester'],
    model: 'sonnet',
    priority: 3
  },
  'security': {
    pattern: /security|vulnerability|audit|compliance|gdpr|sec\s*compliance|penetration|authentication|authorization|permission|access\s*control/i,
    agents: ['security-compliance-swarm', 'security-reviewer'],
    model: 'opus',
    priority: 2
  },
  'documentation': {
    pattern: /documentation|readme|api\s*docs|changelog|code\s*comments|docstring/i,
    agents: ['doc-updater', 'documentation-agent'],
    model: 'sonnet',
    priority: 3
  },
  'architecture': {
    pattern: /architecture|system\s*design|scalability|microservice|monolith|design\s*pattern|refactor/i,
    agents: ['architect', 'architecture-reviewer'],
    model: 'opus',
    priority: 2
  }
};

// File pattern routing
const FILE_PATTERNS = {
  '.py': { domain: 'backend', agents: ['backend-api-swarm', 'coder'] },
  '.tsx': { domain: 'frontend', agents: ['ui-visualization-swarm', 'coder'] },
  '.ts': { domain: 'backend', agents: ['backend-api-swarm', 'coder'] },
  'Dockerfile': { domain: 'infrastructure', agents: ['infrastructure-devops-swarm'] },
  'docker-compose': { domain: 'infrastructure', agents: ['infrastructure-devops-swarm'] },
  '.yml': { domain: 'infrastructure', agents: ['infrastructure-devops-swarm'] },
  '.yaml': { domain: 'infrastructure', agents: ['infrastructure-devops-swarm'] },
  'test_': { domain: 'testing', agents: ['project-quality-swarm', 'tester'] },
  '.test.': { domain: 'testing', agents: ['project-quality-swarm', 'tester'] }
};

/**
 * Route a task to the appropriate agent(s)
 * @param {string} input - The user input or command
 * @param {Object} context - Optional context (file paths, current domain, etc.)
 * @returns {Object} Routing result with agent, model, confidence, and metadata
 */
function routeTask(input, context = {}) {
  const inputLower = input.toLowerCase().trim();
  const result = {
    agent: null,
    alternates: [],
    model: 'opus',
    confidence: 0,
    reason: '',
    domain: null,
    parallel: false,
    sequential: false
  };

  // 1. Check explicit command mappings first (highest priority)
  for (const [cmd, config] of Object.entries(COMMAND_MAPPINGS)) {
    if (inputLower.startsWith(cmd.toLowerCase())) {
      return {
        ...result,
        agent: config.agent,
        model: config.model,
        confidence: 1.0,
        reason: `Explicit command mapping: ${cmd}`,
        domain: config.domain
      };
    }
  }

  // 2. Check file patterns if context includes file paths
  if (context.files && context.files.length > 0) {
    for (const file of context.files) {
      for (const [pattern, config] of Object.entries(FILE_PATTERNS)) {
        if (file.includes(pattern)) {
          return {
            ...result,
            agent: config.agents[0],
            alternates: config.agents.slice(1),
            model: 'sonnet',
            confidence: 0.8,
            reason: `File pattern match: ${pattern}`,
            domain: config.domain
          };
        }
      }
    }
  }

  // 3. Check domain patterns (keyword-based routing)
  const matches = [];
  for (const [domain, config] of Object.entries(DOMAIN_PATTERNS)) {
    if (config.pattern.test(inputLower)) {
      matches.push({
        domain,
        agents: config.agents,
        model: config.model,
        priority: config.priority
      });
    }
  }

  // Sort by priority (lower = higher priority)
  matches.sort((a, b) => a.priority - b.priority);

  if (matches.length > 0) {
    const bestMatch = matches[0];
    return {
      ...result,
      agent: bestMatch.agents[0],
      alternates: bestMatch.agents.slice(1),
      model: bestMatch.model,
      confidence: 0.85,
      reason: `Domain pattern match: ${bestMatch.domain}`,
      domain: bestMatch.domain,
      // If multiple domains matched, might need parallel execution
      parallel: matches.length > 1 && matches[0].priority === matches[1].priority
    };
  }

  // 4. Check for multi-agent orchestration keywords
  if (/orchestrat|coordinat|multi-?(agent|phase)|complex\s*workflow/i.test(inputLower)) {
    // Determine which orchestrator based on context
    if (/investment|deal|portfolio|financial/i.test(inputLower)) {
      return {
        ...result,
        agent: 'queen-investment-orchestrator',
        model: 'opus',
        confidence: 0.9,
        reason: 'Investment orchestration required',
        domain: 'investment'
      };
    }
    if (/github|pr|issue|repo/i.test(inputLower)) {
      return {
        ...result,
        agent: 'github-swarm-coordinator',
        model: 'opus',
        confidence: 0.9,
        reason: 'GitHub orchestration required',
        domain: 'github'
      };
    }
    return {
      ...result,
      agent: 'team-coordinator',
      model: 'opus',
      confidence: 0.8,
      reason: 'General orchestration required',
      domain: 'general'
    };
  }

  // 5. Default fallback to team-coordinator
  return {
    ...result,
    agent: 'team-coordinator',
    model: 'opus',
    confidence: 0.5,
    reason: 'No specific pattern matched - routing to team coordinator for optimal agent selection',
    domain: 'general'
  };
}

/**
 * Get configuration for a specific agent
 * @param {string} agentName - The agent name
 * @returns {Object|null} Agent configuration or null if not found
 */
function getAgentConfig(agentName) {
  if (REGISTRY) {
    // Check custom investment agents
    const customAgent = REGISTRY.agents.custom_investment?.find(a => a.name === agentName);
    if (customAgent) return customAgent;

    // Check imported agents
    const importedAgent = REGISTRY.agents.imported_everything_claude_code?.find(a => a.name === agentName);
    if (importedAgent) return importedAgent;

    // Check github swarm
    const githubAgent = REGISTRY.agents.github_swarm?.agents?.find(a => a.name === agentName);
    if (githubAgent) return githubAgent;
  }

  return null;
}

/**
 * Get all agents for a specific domain
 * @param {string} domain - The domain name
 * @returns {Array} List of agents in the domain
 */
function getAgentsByDomain(domain) {
  if (!REGISTRY) return [];

  const swarmConfig = REGISTRY.swarm_configurations?.[domain];
  if (swarmConfig) {
    return [swarmConfig.coordinator, ...swarmConfig.agents];
  }

  return [];
}

/**
 * Determine if agents should run in parallel or sequential
 * @param {Array} agents - List of agents to coordinate
 * @param {string} taskType - Type of task
 * @returns {Object} Execution plan
 */
function getExecutionPlan(agents, taskType) {
  const parallelPairs = [
    ['code-reviewer', 'security-reviewer'],
    ['test-agent', 'documentation-agent'],
    ['financial-modeler', 'risk-assessor'],
    ['investment-analyst', 'deal-underwriter']
  ];

  const sequentialChains = [
    ['architect', 'coder', 'tester'],
    ['planner', 'coder', 'code-reviewer'],
    ['investment-analyst', 'deal-underwriter', 'financial-modeler', 'investment-analyst']
  ];

  // Check if agents match a parallel pair
  for (const pair of parallelPairs) {
    if (pair.every(a => agents.includes(a))) {
      return {
        mode: 'parallel',
        agents: pair,
        reason: 'Independent analyses can run concurrently'
      };
    }
  }

  // Check if agents match a sequential chain
  for (const chain of sequentialChains) {
    if (chain.some(a => agents.includes(a))) {
      const orderedAgents = chain.filter(a => agents.includes(a));
      return {
        mode: 'sequential',
        agents: orderedAgents,
        reason: 'Output of one agent feeds into the next'
      };
    }
  }

  // Default to parallel for efficiency
  return {
    mode: 'parallel',
    agents,
    reason: 'No specific dependency detected'
  };
}

// Export functions
module.exports = {
  routeTask,
  getAgentConfig,
  getAgentsByDomain,
  getExecutionPlan,
  COMMAND_MAPPINGS,
  DOMAIN_PATTERNS,
  FILE_PATTERNS
};

// CLI usage
if (require.main === module) {
  const args = process.argv.slice(2);
  if (args.length === 0) {
    console.log('Usage: node agent-router.js "<task description>"');
    console.log('Example: node agent-router.js "Analyze credit profile for Company XYZ"');
    process.exit(1);
  }

  const input = args.join(' ');
  const result = routeTask(input);
  console.log(JSON.stringify(result, null, 2));
}
