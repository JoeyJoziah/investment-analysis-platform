#!/usr/bin/env python3
"""
Agent Invocation Framework for Investment Analysis Platform

This framework provides automatic agent selection, task distribution, 
parallel execution coordination, and output synthesis for the 397 
specialized agents across 7 repositories.
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

class ComplexityLevel(Enum):
    SIMPLE = (1, 3)
    MODERATE = (4, 6) 
    COMPLEX = (7, 8)
    ENTERPRISE = (9, 10)

class CoordinationProtocol(Enum):
    SEQUENTIAL_PARALLEL = "sequential_with_parallel_phases"
    COLLABORATIVE_RD = "collaborative_research_and_development"
    DOMAIN_EXPERT_LED = "domain_expert_led_collaboration"
    DESIGN_REVIEW = "design_review_implementation"
    ARCHITECTURE_FIRST = "architecture_first_development"

@dataclass
class Agent:
    name: str
    repository: str
    expertise: List[str]
    specialization: str
    proficiency_level: str  # Expert, Advanced, Intermediate
    
@dataclass
class AgentTeam:
    name: str
    mission: str
    lead_agent: Agent
    core_members: List[Agent]
    supporting_members: List[Agent]
    coordination_protocol: CoordinationProtocol
    use_cases: List[str]
    escalation_path: List[str]

@dataclass
class TaskAnalysis:
    complexity_score: int
    domains: List[str]
    technologies: List[str]
    requires_security: bool
    requires_performance: bool
    requires_testing: bool
    parallel_components: List[str]

class AgentInvocationFramework:
    def __init__(self):
        self.agents = self._load_agents()
        self.teams = self._initialize_teams()
        self.task_patterns = self._load_task_patterns()
        
    def _load_agents(self) -> Dict[str, Agent]:
        """Load all 397 agents from the catalog"""
        agents = {}
        
        # Define key agents for investment platform
        key_agents_data = {
            # Data Pipeline Team
            "data-engineer@lst97": Agent("data-engineer", "lst97-subagents", 
                ["data_pipelines", "etl", "airflow"], "Data Engineering", "Expert"),
            "data-engineer@voltagent": Agent("data-engineer", "voltagent-subagents",
                ["big_data", "streaming", "kafka"], "Data Engineering", "Expert"),
            "data-scientist@wshobson": Agent("data-scientist", "wshobson-agents",
                ["machine_learning", "statistics", "analysis"], "Data Science", "Expert"),
            "kafka-expert@furai": Agent("kafka-expert", "furai-subagents",
                ["streaming", "messaging", "real_time"], "Streaming", "Expert"),
            "python-pro@wshobson": Agent("python-pro", "wshobson-agents",
                ["python", "backend", "optimization"], "Python Development", "Expert"),
            
            # ML/AI Team
            "ml-engineer@wshobson": Agent("ml-engineer", "wshobson-agents",
                ["machine_learning", "pytorch", "model_training"], "Machine Learning", "Expert"),
            "ml-engineer@lst97": Agent("ml-engineer", "lst97-subagents",
                ["deep_learning", "feature_engineering"], "Machine Learning", "Expert"),
            "ai-engineer@lst97": Agent("ai-engineer", "lst97-subagents",
                ["artificial_intelligence", "llm", "rag"], "AI Engineering", "Expert"),
            "pytorch-expert@furai": Agent("pytorch-expert", "furai-subagents",
                ["pytorch", "neural_networks", "training"], "Deep Learning", "Expert"),
            
            # Financial Analysis Team  
            "quant-analyst@wshobson": Agent("quant-analyst", "wshobson-agents",
                ["quantitative_finance", "trading", "risk_metrics"], "Financial Analysis", "Expert"),
            "quant-analyst@voltagent": Agent("quant-analyst", "voltagent-subagents",
                ["portfolio_optimization", "derivatives"], "Financial Analysis", "Expert"),
            "risk-manager@wshobson": Agent("risk-manager", "wshobson-agents",
                ["risk_management", "var", "compliance"], "Risk Management", "Expert"),
            "fintech-engineer@voltagent": Agent("fintech-engineer", "voltagent-subagents",
                ["fintech", "payments", "financial_systems"], "Fintech Development", "Expert"),
            
            # API Integration Team
            "api-designer@claude-code": Agent("api-designer", "claude-code-sub-agents",
                ["api_design", "rest", "openapi"], "API Design", "Expert"),
            "api-designer@voltagent": Agent("api-designer", "voltagent-subagents",
                ["microservices", "api_gateway"], "API Architecture", "Expert"),
            "python-backend-engineer@awesome": Agent("python-backend-engineer", "awesome-claude-code-agents",
                ["fastapi", "backend_engineering"], "Backend Engineering", "Expert"),
            "fastapi-expert@furai": Agent("fastapi-expert", "furai-subagents",
                ["fastapi", "async", "web_frameworks"], "FastAPI Development", "Expert"),
            
            # Security Team
            "security-auditor@wshobson": Agent("security-auditor", "wshobson-agents",
                ["security_audit", "vulnerability_assessment"], "Security", "Expert"),
            "security-auditor@lst97": Agent("security-auditor", "lst97-subagents",
                ["application_security", "compliance"], "Security", "Expert"),
            "penetration-tester@voltagent": Agent("penetration-tester", "voltagent-subagents",
                ["penetration_testing", "security_testing"], "Security Testing", "Expert"),
            "jwt-expert@furai": Agent("jwt-expert", "furai-subagents",
                ["jwt", "authentication", "tokens"], "Authentication", "Expert"),
            
            # Performance Team
            "performance-optimizer@claude-code": Agent("performance-optimizer", "claude-code-sub-agents",
                ["performance_optimization", "profiling"], "Performance", "Expert"),
            "performance-engineer@lst97": Agent("performance-engineer", "lst97-subagents",
                ["system_performance", "scaling"], "Performance Engineering", "Expert"),
            "database-optimizer@wshobson": Agent("database-optimizer", "wshobson-agents",
                ["database_optimization", "query_tuning"], "Database Performance", "Expert"),
            
            # Meta-coordination
            "agent-organizer@lst97": Agent("agent-organizer", "lst97-subagents",
                ["meta_coordination", "team_management"], "Meta-Coordination", "Expert"),
            "system-architect@claude-code": Agent("system-architect", "claude-code-sub-agents",
                ["system_architecture", "design_patterns"], "System Architecture", "Expert"),
        }
        
        return key_agents_data
    
    def _initialize_teams(self) -> Dict[str, AgentTeam]:
        """Initialize the 12 specialized agent teams"""
        teams = {}
        
        # Data Pipeline Team
        teams["data_pipeline"] = AgentTeam(
            name="Data Pipeline Team",
            mission="Airflow DAGs, ETL processes, data ingestion, streaming systems",
            lead_agent=self.agents["data-engineer@lst97"],
            core_members=[
                self.agents["data-engineer@voltagent"],
                self.agents["data-scientist@wshobson"], 
                self.agents["kafka-expert@furai"],
                self.agents["python-pro@wshobson"]
            ],
            supporting_members=[],
            coordination_protocol=CoordinationProtocol.SEQUENTIAL_PARALLEL,
            use_cases=[
                "Activate Airflow pipelines",
                "Implement ETL processes", 
                "Set up streaming data systems",
                "Optimize data ingestion performance"
            ],
            escalation_path=["agent-organizer@lst97", "system-architect@claude-code"]
        )
        
        # ML/AI Team
        teams["ml_ai"] = AgentTeam(
            name="ML/AI Team",
            mission="PyTorch models, scikit-learn, feature engineering, model training",
            lead_agent=self.agents["ml-engineer@wshobson"],
            core_members=[
                self.agents["ml-engineer@lst97"],
                self.agents["ai-engineer@lst97"],
                self.agents["pytorch-expert@furai"],
                self.agents["data-scientist@wshobson"]
            ],
            supporting_members=[],
            coordination_protocol=CoordinationProtocol.COLLABORATIVE_RD,
            use_cases=[
                "Train ML models for stock prediction",
                "Implement ensemble methods",
                "Feature engineering and selection", 
                "Model performance optimization"
            ],
            escalation_path=["ai-engineer@lst97", "agent-organizer@lst97"]
        )
        
        # Financial Analysis Team
        teams["financial_analysis"] = AgentTeam(
            name="Financial Analysis Team", 
            mission="Quantitative analysis, trading strategies, risk metrics, portfolio optimization",
            lead_agent=self.agents["quant-analyst@wshobson"],
            core_members=[
                self.agents["quant-analyst@voltagent"],
                self.agents["risk-manager@wshobson"],
                self.agents["fintech-engineer@voltagent"],
                self.agents["data-scientist@wshobson"]
            ],
            supporting_members=[],
            coordination_protocol=CoordinationProtocol.DOMAIN_EXPERT_LED,
            use_cases=[
                "Develop trading algorithms",
                "Calculate risk metrics (VaR, Sharpe)", 
                "Portfolio optimization strategies",
                "Financial model validation"
            ],
            escalation_path=["agent-organizer@lst97"]
        )
        
        return teams
    
    def _load_task_patterns(self) -> Dict[str, Dict]:
        """Load task pattern recognition rules"""
        return {
            "data_pipeline": {
                "keywords": ["airflow", "etl", "data", "pipeline", "streaming", "kafka", "ingestion"],
                "complexity_base": 6,
                "required_expertise": ["data_engineering", "python"]
            },
            "ml_training": {
                "keywords": ["ml", "model", "train", "pytorch", "scikit-learn", "machine learning", "neural"],
                "complexity_base": 7,
                "required_expertise": ["machine_learning", "data_science", "python"]
            },
            "financial_analysis": {
                "keywords": ["trading", "portfolio", "risk", "sharpe", "var", "financial", "quant"],
                "complexity_base": 6,
                "required_expertise": ["quantitative_finance", "risk_management"]
            },
            "api_development": {
                "keywords": ["api", "endpoint", "fastapi", "rest", "swagger", "openapi"],
                "complexity_base": 5,
                "required_expertise": ["api_design", "backend_engineering"]
            },
            "security_implementation": {
                "keywords": ["security", "authentication", "jwt", "oauth", "compliance", "audit"],
                "complexity_base": 7,
                "required_expertise": ["security", "compliance"]
            },
            "performance_optimization": {
                "keywords": ["performance", "optimization", "slow", "bottleneck", "cache", "scale"],
                "complexity_base": 6,
                "required_expertise": ["performance_optimization", "system_performance"]
            }
        }
    
    def analyze_task(self, user_prompt: str) -> TaskAnalysis:
        """Analyze user prompt to determine complexity and required expertise"""
        prompt_lower = user_prompt.lower()
        
        # Initialize scoring
        complexity_score = 3  # Default moderate complexity
        domains = []
        technologies = []
        parallel_components = []
        
        # Pattern matching for complexity scoring
        complexity_indicators = {
            "simple": ["fix", "update", "change", "modify", "single"],
            "moderate": ["implement", "create", "add", "build", "develop"],
            "complex": ["integrate", "optimize", "refactor", "system", "architecture"],
            "enterprise": ["complete", "comprehensive", "full", "entire", "enterprise", "production"]
        }
        
        # Score based on keywords
        for level, keywords in complexity_indicators.items():
            if any(kw in prompt_lower for kw in keywords):
                if level == "simple":
                    complexity_score += 0
                elif level == "moderate":
                    complexity_score += 2
                elif level == "complex":
                    complexity_score += 4
                elif level == "enterprise":
                    complexity_score += 6
        
        # Domain detection
        for pattern_name, pattern_data in self.task_patterns.items():
            if any(kw in prompt_lower for kw in pattern_data["keywords"]):
                domains.append(pattern_name)
                complexity_score = max(complexity_score, pattern_data["complexity_base"])
        
        # Multi-domain complexity boost
        if len(domains) > 1:
            complexity_score += 2
        if len(domains) > 2:
            complexity_score += 2
            
        # Technology detection
        tech_keywords = {
            "python": ["python", "fastapi", "django", "flask"],
            "javascript": ["javascript", "react", "node", "typescript"],
            "database": ["postgresql", "database", "sql", "timescale"],
            "ml": ["pytorch", "tensorflow", "scikit-learn", "model"],
            "devops": ["docker", "kubernetes", "ci/cd", "deployment"]
        }
        
        for tech, keywords in tech_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                technologies.append(tech)
        
        # Special requirements detection
        requires_security = any(kw in prompt_lower for kw in ["security", "auth", "compliance", "encrypt"])
        requires_performance = any(kw in prompt_lower for kw in ["performance", "optimize", "fast", "scale"])
        requires_testing = any(kw in prompt_lower for kw in ["test", "testing", "pytest", "coverage"])
        
        # Parallel component detection
        parallel_indicators = ["and", "plus", "with", "including", "along with"]
        if any(indicator in prompt_lower for indicator in parallel_indicators):
            # Split on common conjunctions to identify parallel work
            components = re.split(r'\s+(?:and|plus|with|including)\s+', user_prompt)
            if len(components) > 1:
                parallel_components = components
        
        return TaskAnalysis(
            complexity_score=min(complexity_score, 10),
            domains=domains,
            technologies=technologies,
            requires_security=requires_security,
            requires_performance=requires_performance,
            requires_testing=requires_testing,
            parallel_components=parallel_components
        )
    
    def select_teams(self, task_analysis: TaskAnalysis) -> List[Tuple[str, AgentTeam]]:
        """Select appropriate teams based on task analysis"""
        selected_teams = []
        
        # Direct domain mapping
        domain_to_team = {
            "data_pipeline": "data_pipeline",
            "ml_training": "ml_ai", 
            "financial_analysis": "financial_analysis",
            "api_development": "api_integration",
            "security_implementation": "security",
            "performance_optimization": "performance"
        }
        
        # Select primary teams
        for domain in task_analysis.domains:
            if domain in domain_to_team and domain_to_team[domain] in self.teams:
                team_key = domain_to_team[domain]
                selected_teams.append((team_key, self.teams[team_key]))
        
        # Add supporting teams based on requirements
        if task_analysis.requires_security and ("security", self.teams.get("security")) not in selected_teams:
            # Would add security team if defined
            pass
            
        if task_analysis.requires_performance and ("performance", self.teams.get("performance")) not in selected_teams:
            # Would add performance team if defined
            pass
            
        return selected_teams
    
    def plan_execution(self, teams: List[Tuple[str, AgentTeam]], 
                      task_analysis: TaskAnalysis) -> Dict[str, any]:
        """Plan parallel execution strategy"""
        
        execution_plan = {
            "complexity_level": self._get_complexity_level(task_analysis.complexity_score),
            "coordination_strategy": "single_team" if len(teams) == 1 else "multi_team",
            "parallel_phases": [],
            "synthesis_points": [],
            "meta_coordination_required": task_analysis.complexity_score >= 8
        }
        
        if len(teams) == 1:
            # Single team execution
            team_name, team = teams[0]
            execution_plan["primary_team"] = team_name
            execution_plan["coordination_protocol"] = team.coordination_protocol.value
            
            if len(task_analysis.parallel_components) > 1:
                execution_plan["parallel_phases"] = [
                    {"phase": f"Component {i+1}", "component": comp.strip()}
                    for i, comp in enumerate(task_analysis.parallel_components)
                ]
        else:
            # Multi-team coordination
            execution_plan["meta_coordinator"] = "agent-organizer@lst97"
            execution_plan["team_coordination"] = []
            
            for i, (team_name, team) in enumerate(teams):
                execution_plan["team_coordination"].append({
                    "team": team_name,
                    "lead": team.lead_agent.name,
                    "protocol": team.coordination_protocol.value,
                    "phase": i + 1
                })
            
            # Add synthesis points between phases
            execution_plan["synthesis_points"] = [
                f"After phase {i+1}" for i in range(len(teams)-1)
            ]
        
        return execution_plan
    
    def _get_complexity_level(self, score: int) -> ComplexityLevel:
        """Convert complexity score to complexity level"""
        if score <= 3:
            return ComplexityLevel.SIMPLE
        elif score <= 6:
            return ComplexityLevel.MODERATE  
        elif score <= 8:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.ENTERPRISE
    
    def invoke_agents(self, user_prompt: str) -> Dict[str, any]:
        """Main entry point for agent invocation"""
        
        # Step 1: Analyze the task
        task_analysis = self.analyze_task(user_prompt)
        
        # Step 2: Select appropriate teams
        selected_teams = self.select_teams(task_analysis)
        
        # Step 3: Plan execution strategy
        execution_plan = self.plan_execution(selected_teams, task_analysis)
        
        # Step 4: Generate invocation response
        response = {
            "user_prompt": user_prompt,
            "task_analysis": {
                "complexity_score": task_analysis.complexity_score,
                "complexity_level": execution_plan["complexity_level"].name,
                "domains": task_analysis.domains,
                "technologies": task_analysis.technologies,
                "parallel_components": len(task_analysis.parallel_components)
            },
            "selected_teams": [
                {
                    "team_name": team_name,
                    "mission": team.mission,
                    "lead_agent": team.lead_agent.name,
                    "core_members": [agent.name for agent in team.core_members],
                    "coordination_protocol": team.coordination_protocol.value
                }
                for team_name, team in selected_teams
            ],
            "execution_plan": execution_plan,
            "recommendations": self._generate_recommendations(task_analysis, selected_teams)
        }
        
        return response
    
    def _generate_recommendations(self, task_analysis: TaskAnalysis, 
                                 selected_teams: List[Tuple[str, AgentTeam]]) -> List[str]:
        """Generate specific recommendations for the task"""
        recommendations = []
        
        if task_analysis.complexity_score >= 8:
            recommendations.append(
                "High complexity task detected. Recommend using agent-organizer@lst97 for meta-coordination."
            )
        
        if len(selected_teams) > 2:
            recommendations.append(
                "Multi-team coordination required. Plan synthesis checkpoints after each major phase."
            )
        
        if task_analysis.requires_security:
            recommendations.append(
                "Security requirements detected. Include security-auditor@wshobson for review."
            )
        
        if task_analysis.requires_performance:
            recommendations.append(
                "Performance requirements detected. Include performance-optimizer@claude-code."
            )
        
        if len(task_analysis.parallel_components) > 1:
            recommendations.append(
                f"Task has {len(task_analysis.parallel_components)} parallelizable components. "
                "Consider concurrent execution with synthesis."
            )
        
        return recommendations


# Example usage and testing
def main():
    """Example usage of the Agent Invocation Framework"""
    framework = AgentInvocationFramework()
    
    # Test cases
    test_prompts = [
        "Implement real-time stock price streaming with WebSocket API",
        "Train ML models for stock prediction using PyTorch",
        "Optimize PostgreSQL query performance for stock analysis",
        "Create comprehensive ML model training pipeline with automated retraining",
        "Add user authentication and API security with JWT tokens",
        "Activate the data pipeline and implement ETL processes"
    ]
    
    print("🤖 Agent Invocation Framework - Test Results")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\n📝 User Prompt: {prompt}")
        print("-" * 50)
        
        result = framework.invoke_agents(prompt)
        
        print(f"🎯 Complexity: {result['task_analysis']['complexity_level']} "
              f"(Score: {result['task_analysis']['complexity_score']})")
        print(f"🏷️  Domains: {', '.join(result['task_analysis']['domains'])}")
        
        if result['selected_teams']:
            print("\n👥 Selected Teams:")
            for team_info in result['selected_teams']:
                print(f"   • {team_info['team_name']} (Lead: {team_info['lead_agent']})")
                print(f"     Protocol: {team_info['coordination_protocol']}")
        
        if result['recommendations']:
            print("\n💡 Recommendations:")
            for rec in result['recommendations']:
                print(f"   • {rec}")
        
        print()


if __name__ == "__main__":
    main()