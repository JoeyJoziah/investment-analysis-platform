"""
Data Lineage Tracking System
Tracks data flow, transformations, and quality throughout the pipeline
"""

import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import asyncio
from collections import defaultdict
import networkx as nx
import pandas as pd

from backend.utils.cache import get_redis
from backend.utils.database import get_db_async
from sqlalchemy import Column, String, DateTime, JSON, Float, Integer, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

logger = logging.getLogger(__name__)

Base = declarative_base()


class DataOperation(Enum):
    """Types of data operations"""
    INGESTION = "ingestion"
    TRANSFORMATION = "transformation"
    AGGREGATION = "aggregation"
    ENRICHMENT = "enrichment"
    VALIDATION = "validation"
    STORAGE = "storage"
    SERVING = "serving"
    ML_FEATURE = "ml_feature"
    ML_PREDICTION = "ml_prediction"


class DataQualityStatus(Enum):
    """Data quality status levels"""
    PRISTINE = "pristine"      # Perfect quality
    GOOD = "good"               # Minor issues
    DEGRADED = "degraded"       # Some quality concerns
    POOR = "poor"               # Significant issues
    FAILED = "failed"           # Quality check failed


@dataclass
class LineageNode:
    """Represents a node in the data lineage graph"""
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operation: DataOperation = DataOperation.INGESTION
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Data identification
    data_source: str = ""
    data_type: str = ""
    ticker: Optional[str] = None
    
    # Operation details
    operation_name: str = ""
    operation_version: str = "1.0"
    operator: str = "system"
    
    # Data characteristics
    row_count: int = 0
    column_count: int = 0
    data_size_bytes: int = 0
    data_hash: str = ""
    schema_hash: str = ""
    
    # Quality metrics
    quality_status: DataQualityStatus = DataQualityStatus.GOOD
    quality_score: float = 1.0
    quality_checks: Dict[str, bool] = field(default_factory=dict)
    anomalies_detected: List[str] = field(default_factory=list)
    
    # Performance metrics
    processing_time_ms: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Lineage relationships
    parent_nodes: List[str] = field(default_factory=list)
    child_nodes: List[str] = field(default_factory=list)
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['operation'] = self.operation.value
        data['quality_status'] = self.quality_status.value
        return data


class DataLineageTracker:
    """
    Comprehensive data lineage tracking system
    """
    
    def __init__(self):
        self.redis = None
        self.lineage_graph = nx.DiGraph()
        self.active_traces = {}
        self.quality_thresholds = {
            'completeness': 0.95,
            'consistency': 0.98,
            'timeliness': 24,  # hours
            'accuracy': 0.99
        }
        
    async def initialize(self):
        """Initialize the lineage tracker"""
        self.redis = await get_redis()
        await self._load_lineage_graph()
        logger.info("Data lineage tracker initialized")
    
    async def start_trace(
        self,
        data_source: str,
        data_type: str,
        ticker: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Start a new data lineage trace
        
        Args:
            data_source: Source of the data (e.g., 'finnhub', 'alpha_vantage')
            data_type: Type of data (e.g., 'prices', 'fundamentals')
            ticker: Stock ticker if applicable
            metadata: Additional metadata
            
        Returns:
            Trace ID for tracking
        """
        trace_id = str(uuid.uuid4())
        
        # Create root node
        root_node = LineageNode(
            operation=DataOperation.INGESTION,
            data_source=data_source,
            data_type=data_type,
            ticker=ticker,
            operation_name=f"ingest_{data_source}_{data_type}",
            metadata=metadata or {}
        )
        
        # Store in active traces
        self.active_traces[trace_id] = {
            'root_node': root_node,
            'current_node': root_node,
            'nodes': [root_node],
            'start_time': datetime.utcnow()
        }
        
        # Add to graph
        self.lineage_graph.add_node(
            root_node.node_id,
            **root_node.to_dict()
        )
        
        # Persist to Redis
        await self._persist_node(root_node)
        
        logger.debug(f"Started lineage trace {trace_id} for {data_source}/{data_type}")
        return trace_id
    
    async def add_transformation(
        self,
        trace_id: str,
        operation_name: str,
        operation_type: DataOperation,
        input_data: Any,
        output_data: Any,
        quality_checks: Optional[Dict[str, bool]] = None,
        metadata: Optional[Dict] = None
    ) -> LineageNode:
        """
        Add a transformation step to the lineage
        
        Args:
            trace_id: Active trace ID
            operation_name: Name of the transformation
            operation_type: Type of operation
            input_data: Input data (for quality/size calculation)
            output_data: Output data
            quality_checks: Quality check results
            metadata: Additional metadata
            
        Returns:
            New lineage node
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Unknown trace ID: {trace_id}")
        
        trace = self.active_traces[trace_id]
        parent_node = trace['current_node']
        
        # Create new node
        new_node = LineageNode(
            operation=operation_type,
            data_source=parent_node.data_source,
            data_type=parent_node.data_type,
            ticker=parent_node.ticker,
            operation_name=operation_name,
            parent_nodes=[parent_node.node_id],
            metadata=metadata or {}
        )
        
        # Calculate data characteristics
        new_node.row_count = self._get_row_count(output_data)
        new_node.column_count = self._get_column_count(output_data)
        new_node.data_size_bytes = self._estimate_size(output_data)
        new_node.data_hash = self._calculate_data_hash(output_data)
        
        # Perform quality assessment
        quality_result = await self._assess_quality(
            input_data,
            output_data,
            quality_checks
        )
        new_node.quality_status = quality_result['status']
        new_node.quality_score = quality_result['score']
        new_node.quality_checks = quality_result['checks']
        new_node.anomalies_detected = quality_result['anomalies']
        
        # Update parent's children
        parent_node.child_nodes.append(new_node.node_id)
        
        # Add to trace
        trace['nodes'].append(new_node)
        trace['current_node'] = new_node
        
        # Add to graph
        self.lineage_graph.add_node(
            new_node.node_id,
            **new_node.to_dict()
        )
        self.lineage_graph.add_edge(
            parent_node.node_id,
            new_node.node_id,
            operation=operation_name
        )
        
        # Persist
        await self._persist_node(new_node)
        await self._update_node(parent_node)
        
        return new_node
    
    async def complete_trace(
        self,
        trace_id: str,
        final_quality_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Complete a lineage trace and generate summary
        
        Args:
            trace_id: Trace ID to complete
            final_quality_score: Override final quality score
            
        Returns:
            Trace summary
        """
        if trace_id not in self.active_traces:
            raise ValueError(f"Unknown trace ID: {trace_id}")
        
        trace = self.active_traces[trace_id]
        end_time = datetime.utcnow()
        duration = (end_time - trace['start_time']).total_seconds()
        
        # Calculate overall quality
        quality_scores = [node.quality_score for node in trace['nodes']]
        overall_quality = final_quality_score or (
            sum(quality_scores) / len(quality_scores) if quality_scores else 0
        )
        
        # Generate summary
        summary = {
            'trace_id': trace_id,
            'start_time': trace['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'node_count': len(trace['nodes']),
            'operations': [node.operation.value for node in trace['nodes']],
            'overall_quality_score': overall_quality,
            'quality_status': self._determine_quality_status(overall_quality),
            'data_source': trace['root_node'].data_source,
            'data_type': trace['root_node'].data_type,
            'ticker': trace['root_node'].ticker,
            'total_rows_processed': sum(node.row_count for node in trace['nodes']),
            'anomalies': [
                anomaly
                for node in trace['nodes']
                for anomaly in node.anomalies_detected
            ]
        }
        
        # Store summary
        await self._store_trace_summary(trace_id, summary)
        
        # Clean up active trace
        del self.active_traces[trace_id]
        
        logger.info(f"Completed trace {trace_id} with quality score {overall_quality:.2f}")
        return summary
    
    async def get_data_lineage(
        self,
        node_id: Optional[str] = None,
        ticker: Optional[str] = None,
        data_type: Optional[str] = None,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Get lineage information for data
        
        Args:
            node_id: Specific node to trace from
            ticker: Filter by ticker
            data_type: Filter by data type
            max_depth: Maximum depth to traverse
            
        Returns:
            Lineage information including graph and quality metrics
        """
        if node_id:
            # Get lineage for specific node
            ancestors = self._get_ancestors(node_id, max_depth)
            descendants = self._get_descendants(node_id, max_depth)
            
            return {
                'node_id': node_id,
                'node': self.lineage_graph.nodes.get(node_id, {}),
                'ancestors': ancestors,
                'descendants': descendants,
                'lineage_path': self._construct_lineage_path(node_id),
                'quality_history': await self._get_quality_history(node_id)
            }
        
        # Get lineage by filters
        matching_nodes = []
        for node_id, node_data in self.lineage_graph.nodes(data=True):
            if ticker and node_data.get('ticker') != ticker:
                continue
            if data_type and node_data.get('data_type') != data_type:
                continue
            matching_nodes.append((node_id, node_data))
        
        return {
            'filter': {'ticker': ticker, 'data_type': data_type},
            'node_count': len(matching_nodes),
            'nodes': matching_nodes[:100],  # Limit results
            'quality_summary': self._calculate_quality_summary(matching_nodes)
        }
    
    async def analyze_impact(
        self,
        node_id: str,
        issue_type: str = 'quality_degradation'
    ) -> Dict[str, Any]:
        """
        Analyze impact of an issue at a specific node
        
        Args:
            node_id: Node where issue occurred
            issue_type: Type of issue
            
        Returns:
            Impact analysis including affected downstream nodes
        """
        # Get all downstream nodes
        affected_nodes = list(nx.descendants(self.lineage_graph, node_id))
        
        # Analyze impact severity
        impact_analysis = {
            'source_node': node_id,
            'issue_type': issue_type,
            'affected_node_count': len(affected_nodes),
            'affected_nodes': affected_nodes[:50],  # Limit for display
            'impact_by_operation': defaultdict(int),
            'impact_by_data_type': defaultdict(int),
            'critical_paths': [],
            'remediation_suggestions': []
        }
        
        # Categorize impact
        for affected_id in affected_nodes:
            node_data = self.lineage_graph.nodes.get(affected_id, {})
            operation = node_data.get('operation', 'unknown')
            data_type = node_data.get('data_type', 'unknown')
            
            impact_analysis['impact_by_operation'][operation] += 1
            impact_analysis['impact_by_data_type'][data_type] += 1
            
            # Check if this is a critical path (leads to ML predictions)
            if operation == 'ml_prediction':
                path = nx.shortest_path(self.lineage_graph, node_id, affected_id)
                impact_analysis['critical_paths'].append(path)
        
        # Generate remediation suggestions
        if issue_type == 'quality_degradation':
            impact_analysis['remediation_suggestions'] = [
                "Reprocess data from last known good checkpoint",
                "Apply additional quality filters",
                "Switch to alternative data source",
                "Notify downstream consumers"
            ]
        elif issue_type == 'missing_data':
            impact_analysis['remediation_suggestions'] = [
                "Use cached/stale data with warning",
                "Interpolate missing values",
                "Fetch from alternative provider",
                "Skip affected time period"
            ]
        
        return impact_analysis
    
    async def get_data_provenance(
        self,
        data_id: str
    ) -> Dict[str, Any]:
        """
        Get complete provenance for a piece of data
        
        Args:
            data_id: Identifier for the data
            
        Returns:
            Complete provenance information
        """
        # Find node by data hash or ID
        target_node = None
        for node_id, node_data in self.lineage_graph.nodes(data=True):
            if node_data.get('data_hash') == data_id:
                target_node = node_id
                break
        
        if not target_node:
            return {'error': 'Data not found in lineage'}
        
        # Get complete ancestry
        ancestors = list(nx.ancestors(self.lineage_graph, target_node))
        
        # Build provenance chain
        provenance_chain = []
        for ancestor_id in ancestors:
            node_data = self.lineage_graph.nodes[ancestor_id]
            provenance_chain.append({
                'node_id': ancestor_id,
                'timestamp': node_data.get('timestamp'),
                'operation': node_data.get('operation'),
                'data_source': node_data.get('data_source'),
                'quality_score': node_data.get('quality_score'),
                'operator': node_data.get('operator')
            })
        
        # Sort by timestamp
        provenance_chain.sort(key=lambda x: x['timestamp'])
        
        return {
            'data_id': data_id,
            'target_node': target_node,
            'provenance_chain': provenance_chain,
            'original_source': provenance_chain[0] if provenance_chain else None,
            'transformation_count': len(provenance_chain),
            'total_quality_degradation': self._calculate_quality_degradation(
                provenance_chain
            )
        }
    
    async def audit_data_flow(
        self,
        start_date: datetime,
        end_date: datetime,
        ticker: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Audit data flow for compliance and quality
        
        Args:
            start_date: Start of audit period
            end_date: End of audit period
            ticker: Optional ticker filter
            
        Returns:
            Audit report
        """
        # Get all nodes in time range
        nodes_in_range = []
        for node_id, node_data in self.lineage_graph.nodes(data=True):
            timestamp = datetime.fromisoformat(node_data.get('timestamp', ''))
            if start_date <= timestamp <= end_date:
                if not ticker or node_data.get('ticker') == ticker:
                    nodes_in_range.append((node_id, node_data))
        
        # Analyze data flow
        audit_report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'ticker': ticker,
            'total_operations': len(nodes_in_range),
            'operations_by_type': defaultdict(int),
            'quality_metrics': {
                'average_quality': 0,
                'failed_quality_checks': 0,
                'anomalies_detected': []
            },
            'data_sources_used': set(),
            'compliance_issues': [],
            'performance_metrics': {
                'average_processing_time_ms': 0,
                'total_data_processed_gb': 0
            }
        }
        
        # Aggregate metrics
        total_quality = 0
        total_processing_time = 0
        total_data_size = 0
        
        for node_id, node_data in nodes_in_range:
            # Operations
            operation = node_data.get('operation', 'unknown')
            audit_report['operations_by_type'][operation] += 1
            
            # Quality
            quality_score = node_data.get('quality_score', 0)
            total_quality += quality_score
            if quality_score < 0.8:
                audit_report['quality_metrics']['failed_quality_checks'] += 1
            
            # Anomalies
            anomalies = node_data.get('anomalies_detected', [])
            audit_report['quality_metrics']['anomalies_detected'].extend(anomalies)
            
            # Data sources
            source = node_data.get('data_source')
            if source:
                audit_report['data_sources_used'].add(source)
            
            # Performance
            total_processing_time += node_data.get('processing_time_ms', 0)
            total_data_size += node_data.get('data_size_bytes', 0)
            
            # Check compliance
            compliance_issues = self._check_compliance(node_data)
            audit_report['compliance_issues'].extend(compliance_issues)
        
        # Calculate averages
        if nodes_in_range:
            audit_report['quality_metrics']['average_quality'] = (
                total_quality / len(nodes_in_range)
            )
            audit_report['performance_metrics']['average_processing_time_ms'] = (
                total_processing_time / len(nodes_in_range)
            )
        
        audit_report['performance_metrics']['total_data_processed_gb'] = (
            total_data_size / (1024 ** 3)
        )
        
        # Convert sets to lists for JSON serialization
        audit_report['data_sources_used'] = list(audit_report['data_sources_used'])
        
        return audit_report
    
    # Helper methods
    
    async def _assess_quality(
        self,
        input_data: Any,
        output_data: Any,
        quality_checks: Optional[Dict[str, bool]]
    ) -> Dict[str, Any]:
        """Assess data quality"""
        
        checks = quality_checks or {}
        anomalies = []
        
        # Completeness check
        if not checks.get('completeness'):
            completeness = self._check_completeness(output_data)
            checks['completeness'] = completeness > self.quality_thresholds['completeness']
            if not checks['completeness']:
                anomalies.append(f"Completeness {completeness:.2%} below threshold")
        
        # Consistency check
        if not checks.get('consistency'):
            consistency = self._check_consistency(input_data, output_data)
            checks['consistency'] = consistency > self.quality_thresholds['consistency']
            if not checks['consistency']:
                anomalies.append(f"Consistency {consistency:.2%} below threshold")
        
        # Calculate overall score
        passed_checks = sum(1 for v in checks.values() if v)
        total_checks = len(checks) if checks else 1
        score = passed_checks / total_checks
        
        # Determine status
        if score >= 0.95:
            status = DataQualityStatus.PRISTINE
        elif score >= 0.8:
            status = DataQualityStatus.GOOD
        elif score >= 0.6:
            status = DataQualityStatus.DEGRADED
        elif score >= 0.4:
            status = DataQualityStatus.POOR
        else:
            status = DataQualityStatus.FAILED
        
        return {
            'status': status,
            'score': score,
            'checks': checks,
            'anomalies': anomalies
        }
    
    def _check_completeness(self, data: Any) -> float:
        """Check data completeness"""
        if isinstance(data, pd.DataFrame):
            total_values = data.size
            non_null_values = data.count().sum()
            return non_null_values / total_values if total_values > 0 else 0
        elif isinstance(data, dict):
            total_keys = len(data)
            non_null_keys = sum(1 for v in data.values() if v is not None)
            return non_null_keys / total_keys if total_keys > 0 else 0
        return 1.0
    
    def _check_consistency(self, input_data: Any, output_data: Any) -> float:
        """Check data consistency between input and output"""
        # Simplified check - in production, implement domain-specific rules
        if input_data is None or output_data is None:
            return 1.0
        
        # Check row count consistency (allowing for filtering)
        input_rows = self._get_row_count(input_data)
        output_rows = self._get_row_count(output_data)
        
        if input_rows > 0:
            # Output should not have more rows than input (unless aggregation)
            return min(1.0, output_rows / input_rows)
        
        return 1.0
    
    def _get_row_count(self, data: Any) -> int:
        """Get row count from various data types"""
        if isinstance(data, pd.DataFrame):
            return len(data)
        elif isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Assume dict of lists/arrays
            for value in data.values():
                if isinstance(value, (list, pd.Series)):
                    return len(value)
        return 0
    
    def _get_column_count(self, data: Any) -> int:
        """Get column count from various data types"""
        if isinstance(data, pd.DataFrame):
            return len(data.columns)
        elif isinstance(data, dict):
            return len(data)
        return 0
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate data size in bytes"""
        if isinstance(data, pd.DataFrame):
            return data.memory_usage(deep=True).sum()
        elif isinstance(data, dict):
            return len(json.dumps(data, default=str))
        elif isinstance(data, str):
            return len(data.encode('utf-8'))
        return 0
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate hash of data for tracking"""
        if isinstance(data, pd.DataFrame):
            # Hash of shape and sample of data
            hash_input = f"{data.shape}_{data.head(10).to_json()}"
        elif isinstance(data, dict):
            hash_input = json.dumps(data, sort_keys=True, default=str)
        else:
            hash_input = str(data)
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def _determine_quality_status(self, score: float) -> str:
        """Determine quality status from score"""
        if score >= 0.95:
            return DataQualityStatus.PRISTINE.value
        elif score >= 0.8:
            return DataQualityStatus.GOOD.value
        elif score >= 0.6:
            return DataQualityStatus.DEGRADED.value
        elif score >= 0.4:
            return DataQualityStatus.POOR.value
        else:
            return DataQualityStatus.FAILED.value
    
    def _get_ancestors(self, node_id: str, max_depth: int) -> List[Dict]:
        """Get ancestor nodes up to max depth"""
        ancestors = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            node_data = self.lineage_graph.nodes.get(current_id, {})
            
            if current_id != node_id:  # Don't include self
                ancestors.append({
                    'node_id': current_id,
                    'depth': depth,
                    **node_data
                })
            
            # Add parents to queue
            for parent_id in node_data.get('parent_nodes', []):
                queue.append((parent_id, depth + 1))
        
        return ancestors
    
    def _get_descendants(self, node_id: str, max_depth: int) -> List[Dict]:
        """Get descendant nodes up to max depth"""
        descendants = []
        visited = set()
        queue = [(node_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            if depth > max_depth or current_id in visited:
                continue
            
            visited.add(current_id)
            node_data = self.lineage_graph.nodes.get(current_id, {})
            
            if current_id != node_id:  # Don't include self
                descendants.append({
                    'node_id': current_id,
                    'depth': depth,
                    **node_data
                })
            
            # Add children to queue
            for child_id in node_data.get('child_nodes', []):
                queue.append((child_id, depth + 1))
        
        return descendants
    
    def _construct_lineage_path(self, node_id: str) -> List[str]:
        """Construct the main lineage path from root to node"""
        try:
            # Find root nodes (nodes with no parents)
            roots = [n for n in self.lineage_graph.nodes() 
                    if self.lineage_graph.in_degree(n) == 0]
            
            # Find path from a root to this node
            for root in roots:
                if nx.has_path(self.lineage_graph, root, node_id):
                    return nx.shortest_path(self.lineage_graph, root, node_id)
        except:
            pass
        
        return [node_id]
    
    async def _get_quality_history(self, node_id: str) -> List[Dict]:
        """Get quality history for a lineage path"""
        path = self._construct_lineage_path(node_id)
        history = []
        
        for path_node_id in path:
            node_data = self.lineage_graph.nodes.get(path_node_id, {})
            history.append({
                'node_id': path_node_id,
                'timestamp': node_data.get('timestamp'),
                'operation': node_data.get('operation'),
                'quality_score': node_data.get('quality_score', 0),
                'quality_status': node_data.get('quality_status', 'unknown')
            })
        
        return history
    
    def _calculate_quality_summary(
        self,
        nodes: List[Tuple[str, Dict]]
    ) -> Dict[str, Any]:
        """Calculate quality summary for a set of nodes"""
        if not nodes:
            return {}
        
        quality_scores = [n[1].get('quality_score', 0) for n in nodes]
        
        return {
            'average_quality': sum(quality_scores) / len(quality_scores),
            'min_quality': min(quality_scores),
            'max_quality': max(quality_scores),
            'pristine_count': sum(1 for s in quality_scores if s >= 0.95),
            'good_count': sum(1 for s in quality_scores if 0.8 <= s < 0.95),
            'degraded_count': sum(1 for s in quality_scores if 0.6 <= s < 0.8),
            'poor_count': sum(1 for s in quality_scores if 0.4 <= s < 0.6),
            'failed_count': sum(1 for s in quality_scores if s < 0.4)
        }
    
    def _calculate_quality_degradation(
        self,
        provenance_chain: List[Dict]
    ) -> float:
        """Calculate total quality degradation through transformations"""
        if not provenance_chain:
            return 0.0
        
        initial_quality = provenance_chain[0].get('quality_score', 1.0)
        final_quality = provenance_chain[-1].get('quality_score', 1.0)
        
        return max(0, initial_quality - final_quality)
    
    def _check_compliance(self, node_data: Dict) -> List[str]:
        """Check for compliance issues"""
        issues = []
        
        # Check data retention compliance
        timestamp = datetime.fromisoformat(node_data.get('timestamp', ''))
        age_days = (datetime.utcnow() - timestamp).days
        
        if age_days > 2555:  # 7 years for financial data
            issues.append(f"Data exceeds retention period ({age_days} days)")
        
        # Check for PII in metadata
        metadata = node_data.get('metadata', {})
        if any(key in str(metadata).lower() for key in ['ssn', 'email', 'phone']):
            issues.append("Potential PII detected in metadata")
        
        # Check data quality for regulatory reporting
        if node_data.get('quality_score', 0) < 0.95:
            if 'regulatory' in node_data.get('tags', []):
                issues.append("Quality below regulatory threshold")
        
        return issues
    
    async def _persist_node(self, node: LineageNode):
        """Persist node to Redis"""
        key = f"lineage:node:{node.node_id}"
        await self.redis.set(
            key,
            json.dumps(node.to_dict()),
            ex=86400 * 30  # 30 days TTL
        )
    
    async def _update_node(self, node: LineageNode):
        """Update existing node in Redis"""
        await self._persist_node(node)
    
    async def _store_trace_summary(self, trace_id: str, summary: Dict):
        """Store trace summary"""
        key = f"lineage:trace:{trace_id}"
        await self.redis.set(
            key,
            json.dumps(summary),
            ex=86400 * 90  # 90 days TTL
        )
    
    async def _load_lineage_graph(self):
        """Load lineage graph from storage"""
        # In production, load from database
        # For now, start with empty graph
        pass


# Database models for persistent storage
class LineageNodeDB(Base):
    """Database model for lineage nodes"""
    __tablename__ = 'lineage_nodes'
    
    node_id = Column(String, primary_key=True)
    operation = Column(String)
    timestamp = Column(DateTime)
    data_source = Column(String)
    data_type = Column(String)
    ticker = Column(String, nullable=True)
    operation_name = Column(String)
    operation_version = Column(String)
    operator = Column(String)
    row_count = Column(Integer)
    column_count = Column(Integer)
    data_size_bytes = Column(Integer)
    data_hash = Column(String)
    schema_hash = Column(String)
    quality_status = Column(String)
    quality_score = Column(Float)
    quality_checks = Column(JSON)
    anomalies_detected = Column(JSON)
    processing_time_ms = Column(Integer)
    parent_nodes = Column(JSON)
    child_nodes = Column(JSON)
    meta_data = Column(JSON)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    tags = Column(JSON)


class LineageTraceSummary(Base):
    """Database model for trace summaries"""
    __tablename__ = 'lineage_trace_summaries'
    
    trace_id = Column(String, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    duration_seconds = Column(Float)
    node_count = Column(Integer)
    overall_quality_score = Column(Float)
    quality_status = Column(String)
    data_source = Column(String)
    data_type = Column(String)
    ticker = Column(String, nullable=True)
    total_rows_processed = Column(Integer)
    anomalies = Column(JSON)


# Global instance
_lineage_tracker: Optional[DataLineageTracker] = None


async def get_lineage_tracker() -> DataLineageTracker:
    """Get or create the global lineage tracker"""
    global _lineage_tracker
    if _lineage_tracker is None:
        _lineage_tracker = DataLineageTracker()
        await _lineage_tracker.initialize()
    return _lineage_tracker