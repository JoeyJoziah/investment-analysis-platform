"""
ML Operations Cost Monitoring
Integrates ML operations with cost monitoring to stay within $50/month budget
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from collections import defaultdict
import threading
import asyncio

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of ML resources"""
    COMPUTE_CPU = "compute_cpu"
    COMPUTE_GPU = "compute_gpu"
    MEMORY = "memory"
    STORAGE = "storage"
    API_CALLS = "api_calls"
    DATA_TRANSFER = "data_transfer"
    MODEL_INFERENCE = "model_inference"
    TRAINING = "training"
    FEATURE_COMPUTATION = "feature_computation"


class CostCategory(Enum):
    """Cost categories for ML operations"""
    INFRASTRUCTURE = "infrastructure"
    COMPUTE = "compute"
    STORAGE = "storage"
    API_USAGE = "api_usage"
    DATA_PROCESSING = "data_processing"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"


@dataclass
class ResourceUsage:
    """Resource usage record"""
    timestamp: datetime
    resource_type: ResourceType
    usage_amount: float
    unit: str  # hours, GB, requests, etc.
    cost_per_unit: float
    total_cost: float
    operation: str  # training, inference, monitoring, etc.
    model_name: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['resource_type'] = self.resource_type.value
        return data


@dataclass
class CostAlert:
    """Cost monitoring alert"""
    timestamp: datetime
    alert_type: str
    severity: str
    current_cost: float
    budget_limit: float
    utilization_percent: float
    projected_monthly_cost: float
    message: str
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation"""
    category: str
    priority: str  # high, medium, low
    estimated_savings: float
    implementation_effort: str  # easy, medium, hard
    description: str
    action_items: List[str]
    impact_on_performance: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class MLCostTracker:
    """Tracks costs for ML operations"""
    
    def __init__(self, monthly_budget: float = 50.0):
        self.monthly_budget = monthly_budget
        
        # Cost tracking
        self.usage_records: List[ResourceUsage] = []
        self.daily_costs: Dict[str, float] = defaultdict(float)  # date -> cost
        self.category_costs: Dict[CostCategory, float] = defaultdict(float)
        
        # Pricing configuration
        self.pricing = self._load_pricing_config()
        
        # Alerts and thresholds
        self.alert_thresholds = {
            'warning': 0.7,   # 70% of budget
            'critical': 0.85, # 85% of budget
            'emergency': 0.95 # 95% of budget
        }
        
        self.lock = threading.Lock()
        
        # Load existing data
        self._load_usage_history()
        
        logger.info(f"ML Cost Tracker initialized with ${monthly_budget}/month budget")
    
    def _load_pricing_config(self) -> Dict[str, Dict[str, float]]:
        """Load pricing configuration for various resources"""
        
        # Realistic pricing based on cloud providers (in USD)
        pricing = {
            ResourceType.COMPUTE_CPU.value: {
                'cost_per_hour': 0.08,  # $0.08 per CPU hour
                'unit': 'hours'
            },
            ResourceType.COMPUTE_GPU.value: {
                'cost_per_hour': 0.75,  # $0.75 per GPU hour (basic GPU)
                'unit': 'hours'
            },
            ResourceType.MEMORY.value: {
                'cost_per_gb_hour': 0.01,  # $0.01 per GB-hour
                'unit': 'gb_hours'
            },
            ResourceType.STORAGE.value: {
                'cost_per_gb_month': 0.05,  # $0.05 per GB per month
                'unit': 'gb_months'
            },
            ResourceType.API_CALLS.value: {
                'cost_per_1000_calls': 0.10,  # $0.10 per 1000 API calls
                'unit': 'thousands'
            },
            ResourceType.DATA_TRANSFER.value: {
                'cost_per_gb': 0.09,  # $0.09 per GB transferred
                'unit': 'gb'
            },
            ResourceType.MODEL_INFERENCE.value: {
                'cost_per_1000_predictions': 0.02,  # $0.02 per 1000 predictions
                'unit': 'thousands'
            },
            ResourceType.TRAINING.value: {
                'cost_per_hour': 0.15,  # $0.15 per training hour
                'unit': 'hours'
            },
            ResourceType.FEATURE_COMPUTATION.value: {
                'cost_per_1000_features': 0.01,  # $0.01 per 1000 feature computations
                'unit': 'thousands'
            }
        }
        
        return pricing
    
    def record_usage(self,
                    resource_type: ResourceType,
                    usage_amount: float,
                    operation: str,
                    model_name: Optional[str] = None,
                    metadata: Dict[str, Any] = None) -> float:
        """Record resource usage and calculate cost"""
        
        if resource_type.value not in self.pricing:
            logger.warning(f"No pricing info for resource type: {resource_type.value}")
            return 0.0
        
        pricing_info = self.pricing[resource_type.value]
        
        # Calculate cost based on resource type
        if resource_type == ResourceType.COMPUTE_CPU:
            cost_per_unit = pricing_info['cost_per_hour']
            total_cost = usage_amount * cost_per_unit
            unit = 'hours'
            
        elif resource_type == ResourceType.COMPUTE_GPU:
            cost_per_unit = pricing_info['cost_per_hour']
            total_cost = usage_amount * cost_per_unit
            unit = 'hours'
            
        elif resource_type == ResourceType.MEMORY:
            cost_per_unit = pricing_info['cost_per_gb_hour']
            total_cost = usage_amount * cost_per_unit
            unit = 'gb_hours'
            
        elif resource_type == ResourceType.STORAGE:
            cost_per_unit = pricing_info['cost_per_gb_month']
            total_cost = usage_amount * cost_per_unit
            unit = 'gb_months'
            
        elif resource_type == ResourceType.API_CALLS:
            cost_per_unit = pricing_info['cost_per_1000_calls']
            total_cost = (usage_amount / 1000) * cost_per_unit
            unit = 'calls'
            
        elif resource_type == ResourceType.MODEL_INFERENCE:
            cost_per_unit = pricing_info['cost_per_1000_predictions']
            total_cost = (usage_amount / 1000) * cost_per_unit
            unit = 'predictions'
            
        elif resource_type == ResourceType.FEATURE_COMPUTATION:
            cost_per_unit = pricing_info['cost_per_1000_features']
            total_cost = (usage_amount / 1000) * cost_per_unit
            unit = 'features'
            
        else:
            cost_per_unit = 0.01  # Default
            total_cost = usage_amount * cost_per_unit
            unit = 'units'
        
        # Create usage record
        usage_record = ResourceUsage(
            timestamp=datetime.utcnow(),
            resource_type=resource_type,
            usage_amount=usage_amount,
            unit=unit,
            cost_per_unit=cost_per_unit,
            total_cost=total_cost,
            operation=operation,
            model_name=model_name,
            metadata=metadata or {}
        )
        
        with self.lock:
            self.usage_records.append(usage_record)
            
            # Update daily costs
            date_key = usage_record.timestamp.strftime('%Y-%m-%d')
            self.daily_costs[date_key] += total_cost
            
            # Update category costs
            category = self._categorize_operation(operation)
            self.category_costs[category] += total_cost
        
        # Check for cost alerts
        self._check_cost_alerts()
        
        logger.debug(f"Recorded usage: {resource_type.value} - {usage_amount} {unit} = ${total_cost:.4f}")
        
        return total_cost
    
    def _categorize_operation(self, operation: str) -> CostCategory:
        """Categorize operation for cost tracking"""
        
        operation_lower = operation.lower()
        
        if any(keyword in operation_lower for keyword in ['train', 'fit', 'learn']):
            return CostCategory.COMPUTE
        elif any(keyword in operation_lower for keyword in ['inference', 'predict', 'score']):
            return CostCategory.COMPUTE
        elif any(keyword in operation_lower for keyword in ['storage', 'save', 'load', 'persist']):
            return CostCategory.STORAGE
        elif any(keyword in operation_lower for keyword in ['api', 'request', 'call']):
            return CostCategory.API_USAGE
        elif any(keyword in operation_lower for keyword in ['feature', 'preprocess', 'transform']):
            return CostCategory.DATA_PROCESSING
        elif any(keyword in operation_lower for keyword in ['monitor', 'alert', 'drift']):
            return CostCategory.MONITORING
        elif any(keyword in operation_lower for keyword in ['optimize', 'cache', 'compress']):
            return CostCategory.OPTIMIZATION
        else:
            return CostCategory.INFRASTRUCTURE
    
    def get_current_month_cost(self) -> float:
        """Get total cost for current month"""
        
        current_month = datetime.utcnow().strftime('%Y-%m')
        
        with self.lock:
            month_cost = sum(
                cost for date, cost in self.daily_costs.items()
                if date.startswith(current_month)
            )
        
        return month_cost
    
    def get_projected_monthly_cost(self) -> float:
        """Project monthly cost based on current usage"""
        
        current_date = datetime.utcnow()
        days_in_month = self._get_days_in_current_month()
        days_elapsed = current_date.day
        
        if days_elapsed == 0:
            return 0.0
        
        current_month_cost = self.get_current_month_cost()
        daily_average = current_month_cost / days_elapsed
        
        projected_cost = daily_average * days_in_month
        
        return projected_cost
    
    def _get_days_in_current_month(self) -> int:
        """Get number of days in current month"""
        
        current_date = datetime.utcnow()
        if current_date.month == 12:
            next_month = datetime(current_date.year + 1, 1, 1)
        else:
            next_month = datetime(current_date.year, current_date.month + 1, 1)
        
        last_day = next_month - timedelta(days=1)
        return last_day.day
    
    def _check_cost_alerts(self):
        """Check if cost thresholds are exceeded"""
        
        current_cost = self.get_current_month_cost()
        projected_cost = self.get_projected_monthly_cost()
        utilization = projected_cost / self.monthly_budget if self.monthly_budget > 0 else 0
        
        alert_triggered = None
        
        if utilization >= self.alert_thresholds['emergency']:
            alert_triggered = 'emergency'
        elif utilization >= self.alert_thresholds['critical']:
            alert_triggered = 'critical'
        elif utilization >= self.alert_thresholds['warning']:
            alert_triggered = 'warning'
        
        if alert_triggered:
            self._create_cost_alert(
                alert_type=f"budget_{alert_triggered}",
                severity=alert_triggered,
                current_cost=current_cost,
                projected_cost=projected_cost,
                utilization_percent=utilization * 100
            )
    
    def _create_cost_alert(self,
                          alert_type: str,
                          severity: str,
                          current_cost: float,
                          projected_cost: float,
                          utilization_percent: float):
        """Create cost monitoring alert"""
        
        recommendations = self._generate_cost_recommendations(utilization_percent)
        
        alert = CostAlert(
            timestamp=datetime.utcnow(),
            alert_type=alert_type,
            severity=severity,
            current_cost=current_cost,
            budget_limit=self.monthly_budget,
            utilization_percent=utilization_percent,
            projected_monthly_cost=projected_cost,
            message=f"Budget utilization at {utilization_percent:.1f}% - projected monthly cost: ${projected_cost:.2f}",
            recommendations=recommendations
        )
        
        # Log the alert
        logger.warning(f"Cost Alert [{severity.upper()}]: {alert.message}")
        
        # In production, would send to monitoring system
        self._handle_cost_alert(alert)
    
    def _generate_cost_recommendations(self, utilization_percent: float) -> List[str]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        
        if utilization_percent > 95:
            recommendations.extend([
                "EMERGENCY: Disable non-critical ML operations immediately",
                "Switch to cached predictions only",
                "Pause model training and retraining",
                "Enable emergency cost controls"
            ])
        elif utilization_percent > 85:
            recommendations.extend([
                "Reduce model training frequency",
                "Increase caching for predictions",
                "Optimize feature computation",
                "Consider model quantization"
            ])
        elif utilization_percent > 70:
            recommendations.extend([
                "Review and optimize expensive operations",
                "Increase cache TTL for features",
                "Consider batch processing optimizations",
                "Monitor inference costs closely"
            ])
        
        return recommendations
    
    def _handle_cost_alert(self, alert: CostAlert):
        """Handle cost alert with appropriate actions"""
        
        # Emergency actions
        if alert.severity == 'emergency':
            self._enable_emergency_mode()
        
        # Save alert for reporting
        self._save_cost_alert(alert)
    
    def _enable_emergency_mode(self):
        """Enable emergency cost control mode"""
        
        logger.critical("EMERGENCY MODE ACTIVATED - ML operations restricted to essential only")
        
        # Set environment variable to signal emergency mode
        os.environ['ML_EMERGENCY_MODE'] = 'true'
        
        # In production, would:
        # - Disable expensive operations
        # - Switch to cached predictions
        # - Notify administrators
        # - Scale down resources
    
    def is_operation_allowed(self,
                           operation: str,
                           estimated_cost: float,
                           force: bool = False) -> Tuple[bool, str]:
        """Check if an operation is allowed given current budget"""
        
        if force:
            return True, "Forced execution"
        
        # Check emergency mode
        if os.getenv('ML_EMERGENCY_MODE', 'false').lower() == 'true':
            essential_operations = ['cached_prediction', 'monitoring', 'alert']
            if operation not in essential_operations:
                return False, "Emergency mode active - operation not essential"
        
        current_cost = self.get_current_month_cost()
        projected_cost = self.get_projected_monthly_cost()
        
        # Check if operation would exceed budget
        new_projected_cost = projected_cost + estimated_cost
        
        if new_projected_cost > self.monthly_budget:
            return False, f"Operation would exceed budget: ${new_projected_cost:.2f} > ${self.monthly_budget:.2f}"
        
        # Check if operation would trigger critical threshold
        utilization = new_projected_cost / self.monthly_budget
        if utilization > self.alert_thresholds['critical']:
            return False, f"Operation would trigger critical threshold: {utilization*100:.1f}%"
        
        return True, "Operation allowed"
    
    def get_cost_breakdown(self, days_back: int = 30) -> Dict[str, Any]:
        """Get detailed cost breakdown"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        with self.lock:
            recent_records = [
                record for record in self.usage_records
                if record.timestamp >= cutoff_date
            ]
        
        # Cost by resource type
        resource_costs = defaultdict(float)
        for record in recent_records:
            resource_costs[record.resource_type.value] += record.total_cost
        
        # Cost by operation
        operation_costs = defaultdict(float)
        for record in recent_records:
            operation_costs[record.operation] += record.total_cost
        
        # Cost by model
        model_costs = defaultdict(float)
        for record in recent_records:
            if record.model_name:
                model_costs[record.model_name] += record.total_cost
        
        # Daily costs
        daily_breakdown = defaultdict(float)
        for record in recent_records:
            date_key = record.timestamp.strftime('%Y-%m-%d')
            daily_breakdown[date_key] += record.total_cost
        
        total_cost = sum(record.total_cost for record in recent_records)
        current_month_cost = self.get_current_month_cost()
        projected_cost = self.get_projected_monthly_cost()
        
        return {
            'period_days': days_back,
            'total_cost': total_cost,
            'current_month_cost': current_month_cost,
            'projected_monthly_cost': projected_cost,
            'budget_utilization': (projected_cost / self.monthly_budget * 100) if self.monthly_budget > 0 else 0,
            'monthly_budget': self.monthly_budget,
            'cost_by_resource': dict(resource_costs),
            'cost_by_operation': dict(operation_costs),
            'cost_by_model': dict(model_costs),
            'cost_by_category': {cat.value: cost for cat, cost in self.category_costs.items()},
            'daily_costs': dict(daily_breakdown),
            'record_count': len(recent_records)
        }
    
    def generate_optimization_recommendations(self) -> List[OptimizationRecommendation]:
        """Generate cost optimization recommendations"""
        
        recommendations = []
        cost_breakdown = self.get_cost_breakdown(days_back=30)
        
        # Analyze resource usage patterns
        resource_costs = cost_breakdown['cost_by_resource']
        operation_costs = cost_breakdown['cost_by_operation']
        model_costs = cost_breakdown['cost_by_model']
        
        # High compute costs
        if resource_costs.get(ResourceType.COMPUTE_CPU.value, 0) > 20:
            recommendations.append(OptimizationRecommendation(
                category="compute",
                priority="high",
                estimated_savings=resource_costs.get(ResourceType.COMPUTE_CPU.value, 0) * 0.3,
                implementation_effort="medium",
                description="High CPU compute costs detected",
                action_items=[
                    "Implement model quantization to reduce compute requirements",
                    "Optimize batch sizes for better CPU utilization",
                    "Consider caching frequently computed features",
                    "Use more efficient algorithms where possible"
                ],
                impact_on_performance="Low to moderate - may slightly reduce accuracy"
            ))
        
        # High inference costs
        if operation_costs.get('inference', 0) > 15:
            recommendations.append(OptimizationRecommendation(
                category="inference",
                priority="medium",
                estimated_savings=operation_costs.get('inference', 0) * 0.4,
                implementation_effort="easy",
                description="High inference costs from frequent predictions",
                action_items=[
                    "Increase prediction caching TTL",
                    "Batch multiple inference requests",
                    "Use optimized model formats (ONNX, TensorRT)",
                    "Implement prediction result reuse for similar inputs"
                ],
                impact_on_performance="Minimal - may improve latency"
            ))
        
        # High training costs
        if operation_costs.get('training', 0) > 10:
            recommendations.append(OptimizationRecommendation(
                category="training",
                priority="medium",
                estimated_savings=operation_costs.get('training', 0) * 0.25,
                implementation_effort="medium",
                description="High model training costs",
                action_items=[
                    "Reduce training frequency for stable models",
                    "Implement incremental learning where appropriate",
                    "Use early stopping to avoid overtraining",
                    "Optimize hyperparameter search efficiency"
                ],
                impact_on_performance="Low - may improve generalization"
            ))
        
        # High storage costs
        if resource_costs.get(ResourceType.STORAGE.value, 0) > 5:
            recommendations.append(OptimizationRecommendation(
                category="storage",
                priority="low",
                estimated_savings=resource_costs.get(ResourceType.STORAGE.value, 0) * 0.5,
                implementation_effort="easy",
                description="High storage costs from data and model artifacts",
                action_items=[
                    "Clean up old model versions and artifacts",
                    "Compress stored data and models",
                    "Implement data retention policies",
                    "Use efficient storage formats (Parquet, HDF5)"
                ],
                impact_on_performance="None"
            ))
        
        # Model-specific recommendations
        if model_costs:
            most_expensive_model = max(model_costs.items(), key=lambda x: x[1])
            if most_expensive_model[1] > 10:
                recommendations.append(OptimizationRecommendation(
                    category="model_optimization",
                    priority="high",
                    estimated_savings=most_expensive_model[1] * 0.2,
                    implementation_effort="medium",
                    description=f"Model '{most_expensive_model[0]}' has high operational costs",
                    action_items=[
                        "Analyze model complexity and simplify if possible",
                        "Implement model distillation for faster inference",
                        "Consider ensemble pruning to reduce model count",
                        "Optimize feature selection to reduce input dimensions"
                    ],
                    impact_on_performance="Low to moderate depending on optimizations"
                ))
        
        return recommendations
    
    def _load_usage_history(self):
        """Load usage history from storage"""
        
        # In production, would load from database
        # For now, keep in memory
        pass
    
    def _save_cost_alert(self, alert: CostAlert):
        """Save cost alert to storage"""
        
        # In production, would save to database and send notifications
        logger.info(f"Cost alert saved: {alert.alert_type} - {alert.message}")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get usage summary for monitoring dashboard"""
        
        current_cost = self.get_current_month_cost()
        projected_cost = self.get_projected_monthly_cost()
        utilization = (projected_cost / self.monthly_budget * 100) if self.monthly_budget > 0 else 0
        
        # Recent usage (last 24 hours)
        cutoff = datetime.utcnow() - timedelta(hours=24)
        recent_usage = sum(
            record.total_cost for record in self.usage_records
            if record.timestamp >= cutoff
        )
        
        return {
            'current_month_cost': current_cost,
            'projected_monthly_cost': projected_cost,
            'monthly_budget': self.monthly_budget,
            'budget_utilization_percent': utilization,
            'remaining_budget': max(0, self.monthly_budget - projected_cost),
            'last_24h_cost': recent_usage,
            'emergency_mode_active': os.getenv('ML_EMERGENCY_MODE', 'false').lower() == 'true',
            'total_usage_records': len(self.usage_records),
            'cost_breakdown': self.get_cost_breakdown(days_back=30)
        }


class MLCostOptimizer:
    """Automatic cost optimization for ML operations"""
    
    def __init__(self, cost_tracker: MLCostTracker):
        self.cost_tracker = cost_tracker
        self.optimization_enabled = True
        
        # Optimization strategies
        self.strategies = {
            'adaptive_caching': True,
            'dynamic_batching': True,
            'model_quantization': True,
            'feature_pruning': True,
            'inference_optimization': True
        }
        
        logger.info("ML Cost Optimizer initialized")
    
    def optimize_operation(self,
                          operation: str,
                          estimated_cost: float,
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Optimize operation to reduce costs"""
        
        if not self.optimization_enabled:
            return {'optimized': False, 'reason': 'optimization_disabled'}
        
        context = context or {}
        optimizations = []
        cost_reduction = 0.0
        
        # Adaptive caching
        if self.strategies['adaptive_caching'] and 'cache' not in context:
            cache_savings = estimated_cost * 0.6  # 60% savings with caching
            optimizations.append({
                'type': 'caching',
                'description': 'Enable result caching',
                'savings': cache_savings
            })
            cost_reduction += cache_savings
        
        # Dynamic batching
        if self.strategies['dynamic_batching'] and context.get('batch_size', 1) == 1:
            batch_savings = estimated_cost * 0.3  # 30% savings with batching
            optimizations.append({
                'type': 'batching',
                'description': 'Use dynamic batching',
                'savings': batch_savings
            })
            cost_reduction += batch_savings
        
        # Model quantization
        if (self.strategies['model_quantization'] and 
            operation == 'inference' and 
            not context.get('quantized', False)):
            quant_savings = estimated_cost * 0.25  # 25% savings with quantization
            optimizations.append({
                'type': 'quantization',
                'description': 'Use quantized model',
                'savings': quant_savings
            })
            cost_reduction += quant_savings
        
        return {
            'optimized': len(optimizations) > 0,
            'optimizations': optimizations,
            'original_cost': estimated_cost,
            'optimized_cost': max(0.1, estimated_cost - cost_reduction),  # Minimum 10% of original
            'total_savings': cost_reduction,
            'savings_percent': (cost_reduction / estimated_cost * 100) if estimated_cost > 0 else 0
        }


# Global instances
_ml_cost_tracker: Optional[MLCostTracker] = None
_ml_cost_optimizer: Optional[MLCostOptimizer] = None

def get_ml_cost_tracker() -> MLCostTracker:
    """Get global ML cost tracker instance"""
    global _ml_cost_tracker
    if _ml_cost_tracker is None:
        monthly_budget = float(os.getenv('ML_MONTHLY_BUDGET', '50.0'))
        _ml_cost_tracker = MLCostTracker(monthly_budget=monthly_budget)
    return _ml_cost_tracker

def get_ml_cost_optimizer() -> MLCostOptimizer:
    """Get global ML cost optimizer instance"""
    global _ml_cost_optimizer
    if _ml_cost_optimizer is None:
        _ml_cost_optimizer = MLCostOptimizer(get_ml_cost_tracker())
    return _ml_cost_optimizer

# Decorators for automatic cost tracking
def track_ml_cost(resource_type: ResourceType, operation: str):
    """Decorator to automatically track ML operation costs"""
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            cost_tracker = get_ml_cost_tracker()
            
            try:
                result = func(*args, **kwargs)
                
                # Calculate usage based on operation time
                end_time = datetime.utcnow()
                duration_hours = (end_time - start_time).total_seconds() / 3600
                
                # Record usage
                cost_tracker.record_usage(
                    resource_type=resource_type,
                    usage_amount=duration_hours,
                    operation=operation,
                    model_name=kwargs.get('model_name'),
                    metadata={
                        'function': func.__name__,
                        'duration_seconds': (end_time - start_time).total_seconds(),
                        'success': True
                    }
                )
                
                return result
                
            except Exception as e:
                # Record failed operation
                end_time = datetime.utcnow()
                duration_hours = (end_time - start_time).total_seconds() / 3600
                
                cost_tracker.record_usage(
                    resource_type=resource_type,
                    usage_amount=duration_hours,
                    operation=f"{operation}_failed",
                    model_name=kwargs.get('model_name'),
                    metadata={
                        'function': func.__name__,
                        'duration_seconds': (end_time - start_time).total_seconds(),
                        'success': False,
                        'error': str(e)
                    }
                )
                
                raise
        
        return wrapper
    return decorator