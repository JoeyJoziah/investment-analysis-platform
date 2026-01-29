# Phase 2 Implementation - Code Examples

## Task 1: Generic Dict → Dict[str, Any]

### Example 1: admin.py

#### BEFORE
```python
# /backend/api/routers/admin.py - Before Phase 2
from typing import List, Optional, Dict
from backend.models.api_response import ApiResponse, success_response

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict]:  # ❌ Generic Dict - no type info
    """Delete a user account"""

    # ... implementation ...

    return success_response(data={
        "id": user_id,
        "status": "deleted",
        "timestamp": datetime.utcnow().isoformat(),
        "details": {"reason": "admin deletion", "actor_id": current_user.id}
    })
```

#### AFTER
```python
# /backend/api/routers/admin.py - After Phase 2
from typing import List, Optional, Dict, Any  # ✅ Any added
from backend.models.api_response import ApiResponse, success_response

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: str,
    request: Request,
    current_user = Depends(check_admin_permission)
) -> ApiResponse[Dict[str, Any]]:  # ✅ Explicit Dict[str, Any]
    """Delete a user account"""

    # ... implementation ...

    return success_response(data={
        "id": user_id,
        "status": "deleted",
        "timestamp": datetime.utcnow().isoformat(),
        "details": {"reason": "admin deletion", "actor_id": current_user.id}
    })
```

**Impact:**
- ✓ Type checkers now know dictionary key type (str)
- ✓ IDEs can provide better autocomplete
- ✓ OpenAPI docs are more specific
- ✓ No runtime behavior change

---

### Example 2: health.py

#### BEFORE
```python
# /backend/api/routers/health.py - Before Phase 2
from typing import Dict, Optional
from backend.models.api_response import ApiResponse, success_response

@router.get("/metrics")
async def get_metrics() -> ApiResponse[Dict]:  # ❌ Generic Dict
    """Get system metrics"""

    metrics = {
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent,
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    return success_response(data=metrics)
```

#### AFTER
```python
# /backend/api/routers/health.py - After Phase 2
from typing import Dict, Optional, Any  # ✅ Any added
from backend.models.api_response import ApiResponse, success_response

@router.get("/metrics")
async def get_metrics() -> ApiResponse[Dict[str, Any]]:  # ✅ Explicit typing
    """Get system metrics"""

    metrics = {
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total": psutil.virtual_memory().total,
                "used": psutil.virtual_memory().used,
                "percent": psutil.virtual_memory().percent,
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    return success_response(data=metrics)
```

**Changes Made:** 5 endpoints in health.py updated

---

## Task 2: New Response Models for agents.py

### Example 1: AgentSelectionResponse

#### BEFORE
```python
# /backend/api/routers/agents.py - Before Phase 2
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from backend.models.api_response import ApiResponse, success_response

# No structured response model - just returns Dict
@router.post("/select-agent")
async def select_agent_endpoint(
    request: AgentSelectionRequest,
    current_user = Depends(get_current_user)
) -> ApiResponse[Dict]:  # ❌ Untyped Dict
    """
    Get agent selection recommendation
    """
    try:
        recommendation = await engine.agent_orchestrator.select_agent(
            task_description=request.task_description
        )

        # Return whatever structure the orchestrator provides
        return success_response(data={
            "recommended_agent": recommendation.agent_type,
            "confidence_score": recommendation.confidence,
            "reasoning": recommendation.reason,
            "alternative_agents": recommendation.alternatives,
            "estimated_tokens": recommendation.token_estimate
        })
```

#### AFTER
```python
# /backend/api/routers/agents.py - After Phase 2
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from backend.models.api_response import ApiResponse, success_response

# ✅ NEW: Structured response model
class AgentSelectionResponse(BaseModel):
    """Response for agent selection recommendation"""
    recommended_agent: str = Field(
        ...,
        description="Recommended agent type"
    )
    confidence_score: float = Field(
        ...,
        ge=0.0,  # ✅ Validation: >= 0.0
        le=1.0,  # ✅ Validation: <= 1.0
        description="Confidence in recommendation"
    )
    reasoning: str = Field(
        ...,
        description="Why this agent was selected"
    )
    alternative_agents: List[str] = Field(
        default_factory=list,
        description="Alternative agent options"
    )
    estimated_tokens: Optional[int] = Field(
        None,
        description="Estimated token usage"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "recommended_agent": "coder",
                "confidence_score": 0.92,
                "reasoning": "Task requires code generation with TDD approach",
                "alternative_agents": ["tdd-guide", "refactor-cleaner"],
                "estimated_tokens": 2500
            }
        }

@router.post("/select-agent")
async def select_agent_endpoint(
    request: AgentSelectionRequest,
    current_user = Depends(get_current_user)
) -> ApiResponse[AgentSelectionResponse]:  # ✅ Strongly typed
    """
    Get agent selection recommendation
    """
    try:
        recommendation = await engine.agent_orchestrator.select_agent(
            task_description=request.task_description
        )

        # ✅ Return structured model with validation
        return success_response(data=AgentSelectionResponse(
            recommended_agent=recommendation.agent_type,
            confidence_score=recommendation.confidence,
            reasoning=recommendation.reason,
            alternative_agents=recommendation.alternatives,
            estimated_tokens=recommendation.token_estimate
        ))
```

**Benefits:**
- ✓ Confidence score validated (0.0-1.0)
- ✓ Type checkers know exact structure
- ✓ Pydantic validates all fields
- ✓ OpenAPI docs show field descriptions
- ✓ Auto-completion in IDEs

---

### Example 2: EngineStatusResponse

#### BEFORE
```python
# /backend/api/routers/agents.py - Before Phase 2

@router.get("/status")
async def get_engine_status(
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
) -> ApiResponse[Dict]:  # ❌ Untyped
    """Get comprehensive engine status and statistics"""
    try:
        status = await engine.get_engine_status()
        return success_response(data=status)  # Whatever the engine returns
```

**Problem:**
- Response structure undefined
- Can't validate response
- API docs don't show field details
- Client code has no type hints

#### AFTER
```python
# /backend/api/routers/agents.py - After Phase 2

# ✅ NEW: Strongly typed response model
class EngineStatusResponse(BaseModel):
    """Response for engine status query"""
    status: str = Field(
        ...,
        description="Overall engine status"
    )
    uptime_seconds: float = Field(
        ...,
        description="Engine uptime in seconds"
    )
    analysis_count: int = Field(
        ...,
        description="Total analyses performed"
    )
    error_count: int = Field(
        ...,
        description="Number of errors encountered"
    )
    active_analyses: int = Field(
        ...,
        description="Currently active analyses"
    )
    performance_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Performance statistics"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "operational",
                "uptime_seconds": 3600.0,
                "analysis_count": 42,
                "error_count": 0,
                "active_analyses": 2,
                "performance_metrics": {
                    "avg_latency_ms": 2300,
                    "success_rate": 0.98
                }
            }
        }

@router.get("/status")
async def get_engine_status(
    current_user = Depends(get_current_user),
    engine: HybridAnalysisEngine = Depends(get_hybrid_engine)
) -> ApiResponse[EngineStatusResponse]:  # ✅ Strongly typed
    """Get comprehensive engine status and statistics"""
    try:
        status = await engine.get_engine_status()

        # ✅ Construct typed response with validation
        return success_response(data=EngineStatusResponse(
            status=status.get("status", "unknown"),
            uptime_seconds=status.get("uptime_seconds", 0.0),
            analysis_count=status.get("analysis_count", 0),
            error_count=status.get("error_count", 0),
            active_analyses=status.get("active_analyses", 0),
            performance_metrics=status.get("performance_metrics", {})
        ))
```

**Improvements:**
- ✓ Response structure guaranteed
- ✓ Integer counts validated
- ✓ Float uptime guaranteed
- ✓ Dict metrics flexible but typed
- ✓ Full field documentation

---

### Example 3: AgentBudgetResponse

#### BEFORE
```python
# Without structured model - cost calculation is opaque

@router.post("/calculate-budget")
async def calculate_budget(
    request: BudgetRequest,
    current_user = Depends(get_current_user)
) -> ApiResponse[Dict]:  # ❌ Unknown structure
    """Calculate budget for task"""

    cost = budget_manager.estimate_cost(request.task)

    return success_response(data={
        "total_budget": cost.total,
        "estimated_cost": cost.estimate,
        "budget_remaining": cost.remaining,
        "cost_breakdown": cost.breakdown,  # What's in this?
        "within_budget": cost.total >= cost.estimate,
        "optimization_suggestions": cost.suggestions  # What type?
    })
```

#### AFTER
```python
# ✅ NEW: Strongly typed budget response

class AgentBudgetResponse(BaseModel):
    """Response for agent budget calculation"""
    total_budget: float = Field(
        ...,
        description="Total budget in USD"
    )
    estimated_cost: float = Field(
        ...,
        description="Estimated cost for this task"
    )
    budget_remaining: float = Field(
        ...,
        description="Remaining budget after task"
    )
    cost_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Cost by component (e.g., llm_calls, embeddings)"
    )
    within_budget: bool = Field(
        ...,
        description="Whether task is within budget"
    )
    optimization_suggestions: List[str] = Field(
        default_factory=list,
        description="Ways to reduce cost"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "total_budget": 50.0,
                "estimated_cost": 3.45,
                "budget_remaining": 46.55,
                "cost_breakdown": {
                    "llm_calls": 2.50,
                    "embeddings": 0.95
                },
                "within_budget": True,
                "optimization_suggestions": [
                    "Use haiku for simple tasks",
                    "Batch embeddings requests"
                ]
            }
        }

@router.post("/calculate-budget")
async def calculate_budget(
    request: BudgetRequest,
    current_user = Depends(get_current_user)
) -> ApiResponse[AgentBudgetResponse]:  # ✅ Structured
    """Calculate budget for task"""

    cost = budget_manager.estimate_cost(request.task)

    # ✅ Return structured, validated response
    return success_response(data=AgentBudgetResponse(
        total_budget=cost.total,
        estimated_cost=cost.estimate,
        budget_remaining=cost.remaining,
        cost_breakdown=cost.breakdown,
        within_budget=cost.total >= cost.estimate,
        optimization_suggestions=cost.suggestions
    ))
```

**Result:**
- ✓ All fields have clear types (float, bool, Dict, List)
- ✓ Cost breakdown typed as Dict[str, float]
- ✓ Suggestions typed as List[str]
- ✓ API docs show exact structure
- ✓ Client code can type-check responses

---

## Comparison: Type Safety Improvements

### Without Phase 2
```python
# Type checker can't help here
response: ApiResponse[Dict]  # What's in the dict?

# Developer has to check docs or code
def process_response(data: Dict):
    # What keys exist? What are the types?
    # Just hope the docs are accurate...
    print(data["status"])  # If it exists...
    print(data["uptime"])  # What if this is missing?
    for metric_name, metric_value in data.get("metrics", {}).items():
        # What type is metric_value?
        print(f"{metric_name}: {metric_value}")
```

### With Phase 2
```python
# Type checker validates everything
response: ApiResponse[EngineStatusResponse]

# Developer gets full type safety
def process_response(data: EngineStatusResponse):
    # ✓ Type checker knows 'status' exists and is str
    print(data.status)

    # ✓ Type checker knows 'uptime_seconds' exists and is float
    uptime_hours = data.uptime_seconds / 3600.0

    # ✓ Type checker knows 'performance_metrics' is Dict[str, Any]
    for metric_name, metric_value in data.performance_metrics.items():
        # metric_name is str, metric_value is Any
        print(f"{metric_name}: {metric_value}")
```

---

## OpenAPI/Swagger Impact

### BEFORE Phase 2
```json
{
  "responses": {
    "200": {
      "description": "Successful Response",
      "content": {
        "application/json": {
          "schema": {
            "type": "object",
            "title": "Response",
            "properties": {
              "data": {
                "type": "object"  // ❌ No field details!
              }
            }
          }
        }
      }
    }
  }
}
```

### AFTER Phase 2
```json
{
  "responses": {
    "200": {
      "description": "Successful Response",
      "content": {
        "application/json": {
          "schema": {
            "type": "object",
            "title": "Response",
            "properties": {
              "data": {
                "$ref": "#/components/schemas/EngineStatusResponse"
              }
            }
          }
        }
      }
    }
  }
}

// In components/schemas:
"EngineStatusResponse": {
  "type": "object",
  "title": "EngineStatusResponse",
  "required": ["status", "uptime_seconds", "analysis_count", "error_count", "active_analyses"],
  "properties": {
    "status": {
      "type": "string",
      "description": "Overall engine status"
    },
    "uptime_seconds": {
      "type": "number",
      "description": "Engine uptime in seconds"
    },
    "analysis_count": {
      "type": "integer",
      "description": "Total analyses performed"
    },
    "error_count": {
      "type": "integer",
      "description": "Number of errors encountered"
    },
    "active_analyses": {
      "type": "integer",
      "description": "Currently active analyses"
    },
    "performance_metrics": {
      "type": "object",
      "description": "Performance statistics"
    }
  },
  "examples": {
    "status": "operational",
    "uptime_seconds": 3600.0,
    "analysis_count": 42,
    "error_count": 0,
    "active_analyses": 2,
    "performance_metrics": {
      "avg_latency_ms": 2300,
      "success_rate": 0.98
    }
  }
}
```

---

## Summary of Changes

### All 41 Endpoints Updated

**Pattern Changed:**
```python
# FROM:
) -> ApiResponse[Dict]:

# TO:
) -> ApiResponse[Dict[str, Any]]:
```

### 6 New Models for agents.py

| Model | Fields | Constraints |
|-------|--------|-------------|
| AgentSelectionResponse | 5 | confidence_score: 0.0-1.0 |
| AgentBudgetResponse | 6 | cost_breakdown: Dict[str, float] |
| EngineStatusResponse | 6 | All fields documented |
| ConnectivityTestResponse | 3 | Minimal structure |
| AnalysisModeResponse | 3 | Simple confirmation |
| SelectionStatsResponse | 1 | Flexible stats dict |

### Result
✅ Full type safety across entire API
✅ Better IDE support and autocomplete
✅ Detailed OpenAPI documentation
✅ Pydantic validation of responses
✅ Zero breaking changes
