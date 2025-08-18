#!/usr/bin/env python3
"""
Comprehensive Agent Analysis Script
Analyzes all agents across 7 repositories to identify duplicates, capabilities, and team assignments.
"""

import os
import json
from pathlib import Path
from collections import defaultdict
import re

def parse_agent_file(file_path):
    """Parse an agent markdown file to extract metadata and capabilities."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract YAML frontmatter
        frontmatter_match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        frontmatter = {}
        if frontmatter_match:
            fm_content = frontmatter_match.group(1)
            for line in fm_content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    frontmatter[key.strip()] = value.strip()
        
        # Extract capabilities from content
        capabilities = []
        focus_areas = []
        
        # Look for ## Focus Areas or ## Approach sections
        focus_match = re.search(r'## Focus Areas\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
        if focus_match:
            focus_areas = [line.strip('- ').strip() for line in focus_match.group(1).split('\n') 
                          if line.strip() and line.strip().startswith('-')]
        
        # Look for key capabilities
        caps_match = re.search(r'(?:Key Capabilities|Expertise|Specializations?)\s*[:\n](.*?)(?=\n##|\Z)', content, re.DOTALL)
        if caps_match:
            caps_text = caps_match.group(1)
            capabilities = [line.strip('- ').strip() for line in caps_text.split('\n') 
                          if line.strip() and (line.strip().startswith('-') or line.strip().startswith('*'))]
        
        return {
            'name': frontmatter.get('name', Path(file_path).stem),
            'description': frontmatter.get('description', ''),
            'model': frontmatter.get('model', ''),
            'tools': frontmatter.get('tools', ''),
            'focus_areas': focus_areas[:5],  # Top 5 focus areas
            'capabilities': capabilities[:5],  # Top 5 capabilities
            'file_path': file_path,
            'repository': get_repo_from_path(file_path)
        }
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None

def get_repo_from_path(file_path):
    """Extract repository name from file path."""
    path_parts = Path(file_path).parts
    for part in path_parts:
        if any(repo in part for repo in ['claude-code-sub-agents', 'awesome-claude-code-agents', 
                                        'furai-subagents', 'voltagent-subagents', 'lst97-subagents', 
                                        'nuttall-agents', 'wshobson-agents']):
            return part
    return 'unknown'

def categorize_agent(agent_data):
    """Categorize agent based on name and capabilities."""
    name = agent_data['name'].lower()
    desc = agent_data['description'].lower()
    caps = ' '.join(agent_data['capabilities']).lower()
    focus = ' '.join(agent_data['focus_areas']).lower()
    
    all_text = f"{name} {desc} {caps} {focus}"
    
    # Data Pipeline Team
    if any(term in all_text for term in ['data-engineer', 'etl', 'airflow', 'kafka', 'streaming', 'pipeline']):
        return 'Data Pipeline Team'
    
    # ML/AI Team
    if any(term in all_text for term in ['ml-engineer', 'ai-engineer', 'machine learning', 'pytorch', 'tensorflow', 'scikit', 'model']):
        return 'ML/AI Team'
    
    # Financial Analysis Team
    if any(term in all_text for term in ['quant', 'financial', 'trading', 'risk', 'portfolio', 'fintech']):
        return 'Financial Analysis Team'
    
    # API Integration Team
    if any(term in all_text for term in ['api-designer', 'fastapi', 'rest', 'graphql', 'openapi', 'swagger']):
        return 'API Integration Team'
    
    # Backend Development Team
    if any(term in all_text for term in ['backend', 'python-pro', 'django', 'flask', 'node', 'express', 'server']):
        return 'Backend Development Team'
    
    # Frontend Team
    if any(term in all_text for term in ['frontend', 'react', 'vue', 'angular', 'ui', 'ux', 'mobile']):
        return 'Frontend Team'
    
    # Database Team
    if any(term in all_text for term in ['database', 'postgres', 'mysql', 'mongodb', 'sql', 'schema']):
        return 'Database Team'
    
    # DevOps Team
    if any(term in all_text for term in ['devops', 'docker', 'kubernetes', 'deployment', 'cloud', 'infrastructure']):
        return 'DevOps Team'
    
    # Security Team
    if any(term in all_text for term in ['security', 'auth', 'oauth', 'encryption', 'vulnerability', 'audit']):
        return 'Security Team'
    
    # Performance Team
    if any(term in all_text for term in ['performance', 'optimization', 'caching', 'monitoring']):
        return 'Performance Team'
    
    # Testing Team
    if any(term in all_text for term in ['test', 'qa', 'quality', 'debugging', 'jest', 'pytest']):
        return 'Testing Team'
    
    # Documentation Team
    if any(term in all_text for term in ['documentation', 'docs', 'technical writing', 'api-documenter']):
        return 'Documentation Team'
    
    return 'Uncategorized'

def analyze_agents():
    """Main analysis function."""
    agents_dir = Path('.claude/agents')
    if not agents_dir.exists():
        print("Error: .claude/agents directory not found")
        return
    
    all_agents = []
    repo_counts = defaultdict(int)
    
    # Find all agent files
    for md_file in agents_dir.rglob('*.md'):
        # Skip README and other non-agent files
        if md_file.stem.upper() in ['README', 'CONTRIBUTING', 'LICENSE', 'CLAUDE']:
            continue
        
        agent_data = parse_agent_file(md_file)
        if agent_data:
            all_agents.append(agent_data)
            repo_counts[agent_data['repository']] += 1
    
    print(f"=== AGENT INVENTORY ANALYSIS ===")
    print(f"Total agents found: {len(all_agents)}")
    print()
    
    print("=== REPOSITORY BREAKDOWN ===")
    for repo, count in sorted(repo_counts.items()):
        print(f"{repo}: {count} agents")
    print()
    
    # Find duplicates
    name_to_agents = defaultdict(list)
    for agent in all_agents:
        name_to_agents[agent['name']].append(agent)
    
    duplicates = {name: agents for name, agents in name_to_agents.items() if len(agents) > 1}
    
    print(f"=== DUPLICATE AGENTS ===")
    print(f"Found {len(duplicates)} duplicate agent names across repositories:")
    for name, agents in sorted(duplicates.items()):
        print(f"\n{name} ({len(agents)} versions):")
        for agent in agents:
            print(f"  - {agent['repository']}")
            if agent['description']:
                print(f"    Description: {agent['description'][:100]}...")
    print()
    
    # Categorize agents
    team_assignments = defaultdict(list)
    for agent in all_agents:
        category = categorize_agent(agent)
        team_assignments[category].append(agent)
    
    print("=== TEAM ASSIGNMENTS ===")
    for team, agents in sorted(team_assignments.items()):
        print(f"\n{team} ({len(agents)} agents):")
        agent_names = sorted(set(agent['name'] for agent in agents))
        for name in agent_names[:10]:  # Show first 10
            count = len([a for a in agents if a['name'] == name])
            suffix = f" ({count} versions)" if count > 1 else ""
            print(f"  - {name}{suffix}")
        if len(agent_names) > 10:
            print(f"  ... and {len(agent_names) - 10} more")
    
    # Generate recommendations
    print("\n=== REPOSITORY PRIORITIZATION RECOMMENDATIONS ===")
    repo_quality_scores = {
        'wshobson-agents': 95,  # Financial focus, perfect for investment platform
        'lst97-subagents': 90,  # Well-organized, agent-organizer, context-manager
        'voltagent-subagents': 85,  # Enterprise patterns, good categorization
        'furai-subagents': 80,  # Technology specialists, comprehensive coverage
        'claude-code-sub-agents': 75,  # Core development tools
        'awesome-claude-code-agents': 70,  # Quality but limited scope
        'nuttall-agents': 65   # Modern practices but small collection
    }
    
    print("Recommended priority order for duplicate resolution:")
    for repo, score in sorted(repo_quality_scores.items(), key=lambda x: x[1], reverse=True):
        count = repo_counts.get(repo, 0)
        print(f"1. {repo} (Score: {score}, Agents: {count})")
        
        # Show specializations
        if repo == 'wshobson-agents':
            print("   - Financial analysis (quant-analyst, risk-manager)")
            print("   - Python expertise (python-pro)")
            print("   - ML/AI capabilities (ml-engineer, data-scientist)")
        elif repo == 'lst97-subagents':
            print("   - Meta-coordination (agent-organizer)")
            print("   - Context management (context-manager)")
            print("   - Organized by business function")
        elif repo == 'voltagent-subagents':
            print("   - Enterprise development patterns")
            print("   - Well-categorized by domain")
            print("   - Professional workflows")
    
    # Save detailed analysis
    analysis_result = {
        'total_agents': len(all_agents),
        'repository_counts': dict(repo_counts),
        'duplicate_count': len(duplicates),
        'duplicates': {name: [a['repository'] for a in agents] 
                      for name, agents in duplicates.items()},
        'team_assignments': {team: [a['name'] for a in agents] 
                           for team, agents in team_assignments.items()},
        'priority_repositories': list(repo_quality_scores.keys())
    }
    
    with open('agent_analysis_results.json', 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\nDetailed analysis saved to agent_analysis_results.json")
    return analysis_result

if __name__ == "__main__":
    analyze_agents()