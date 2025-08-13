"""
Intelligent Stock Distribution System with Pagination
Optimizes distribution of stocks across processing resources
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from collections import defaultdict
import heapq

from backend.utils.cache import get_redis
from backend.utils.enhanced_cost_monitor import StockPriority
from backend.config.settings import settings

logger = logging.getLogger(__name__)


class DistributionStrategy(Enum):
    """Stock distribution strategies"""
    ROUND_ROBIN = "round_robin"
    WEIGHTED = "weighted"
    HASH_BASED = "hash_based"
    PRIORITY_BASED = "priority_based"
    ADAPTIVE = "adaptive"
    GEOGRAPHIC = "geographic"


@dataclass
class ProcessingNode:
    """Represents a processing node/worker"""
    node_id: str
    capacity: int  # Max stocks it can process
    current_load: int = 0
    processing_speed: float = 1.0  # Relative speed
    specialization: List[str] = field(default_factory=list)  # e.g., ['tech', 'healthcare']
    geographic_region: str = ""
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    error_rate: float = 0.0
    average_latency_ms: float = 0.0
    
    @property
    def available_capacity(self) -> int:
        return max(0, self.capacity - self.current_load)
    
    @property
    def load_factor(self) -> float:
        return self.current_load / self.capacity if self.capacity > 0 else 1.0


@dataclass
class StockBatch:
    """Batch of stocks for processing"""
    batch_id: str
    stocks: List[Dict[str, Any]]
    priority: StockPriority
    assigned_node: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    processing_started: Optional[datetime] = None
    processing_completed: Optional[datetime] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PaginationState:
    """Pagination state for large result sets"""
    total_items: int
    page_size: int
    current_page: int = 1
    sort_field: str = "ticker"
    sort_order: str = "asc"
    filters: Dict[str, Any] = field(default_factory=dict)
    cursor: Optional[str] = None  # For cursor-based pagination
    
    @property
    def total_pages(self) -> int:
        return math.ceil(self.total_items / self.page_size)
    
    @property
    def offset(self) -> int:
        return (self.current_page - 1) * self.page_size
    
    @property
    def has_next(self) -> bool:
        return self.current_page < self.total_pages
    
    @property
    def has_previous(self) -> bool:
        return self.current_page > 1


class StockDistributor:
    """
    Intelligent stock distribution system with pagination support
    """
    
    def __init__(self):
        self.redis = None
        self.nodes: Dict[str, ProcessingNode] = {}
        self.distribution_history = defaultdict(list)
        self.batch_queue = asyncio.Queue()
        self.failed_batches = []
        self.distribution_metrics = defaultdict(lambda: {
            'total_distributed': 0,
            'success_rate': 0.0,
            'average_latency': 0.0
        })
        
    async def initialize(self):
        """Initialize the distributor"""
        self.redis = await get_redis()
        await self._discover_nodes()
        logger.info(f"Stock distributor initialized with {len(self.nodes)} nodes")
    
    async def distribute_stocks(
        self,
        stocks: List[Dict[str, Any]],
        strategy: DistributionStrategy = DistributionStrategy.ADAPTIVE,
        batch_size: int = 100,
        priority_field: str = "priority"
    ) -> List[StockBatch]:
        """
        Distribute stocks across processing nodes
        
        Args:
            stocks: List of stock dictionaries
            strategy: Distribution strategy to use
            batch_size: Size of each batch
            priority_field: Field to use for priority
            
        Returns:
            List of stock batches with node assignments
        """
        if not self.nodes:
            raise RuntimeError("No processing nodes available")
        
        # Group stocks by priority
        priority_groups = self._group_by_priority(stocks, priority_field)
        
        # Create batches
        all_batches = []
        
        for priority, priority_stocks in priority_groups.items():
            batches = self._create_batches(priority_stocks, batch_size, priority)
            
            # Distribute batches based on strategy
            if strategy == DistributionStrategy.ROUND_ROBIN:
                distributed = await self._distribute_round_robin(batches)
            elif strategy == DistributionStrategy.WEIGHTED:
                distributed = await self._distribute_weighted(batches)
            elif strategy == DistributionStrategy.HASH_BASED:
                distributed = await self._distribute_hash_based(batches)
            elif strategy == DistributionStrategy.PRIORITY_BASED:
                distributed = await self._distribute_priority_based(batches)
            elif strategy == DistributionStrategy.ADAPTIVE:
                distributed = await self._distribute_adaptive(batches)
            elif strategy == DistributionStrategy.GEOGRAPHIC:
                distributed = await self._distribute_geographic(batches)
            else:
                distributed = await self._distribute_round_robin(batches)
            
            all_batches.extend(distributed)
        
        # Track distribution
        await self._track_distribution(all_batches)
        
        return all_batches
    
    async def paginate_stocks(
        self,
        stocks: List[Dict[str, Any]],
        page: int = 1,
        page_size: int = 100,
        sort_field: str = "ticker",
        sort_order: str = "asc",
        filters: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[Dict], PaginationState]:
        """
        Paginate stock list with filtering and sorting
        
        Args:
            stocks: Complete list of stocks
            page: Current page number (1-indexed)
            page_size: Items per page
            sort_field: Field to sort by
            sort_order: Sort order ('asc' or 'desc')
            filters: Optional filters to apply
            
        Returns:
            Tuple of (paginated stocks, pagination state)
        """
        # Apply filters
        filtered_stocks = stocks
        if filters:
            filtered_stocks = self._apply_filters(stocks, filters)
        
        # Sort stocks
        sorted_stocks = self._sort_stocks(
            filtered_stocks,
            sort_field,
            sort_order
        )
        
        # Create pagination state
        pagination = PaginationState(
            total_items=len(sorted_stocks),
            page_size=page_size,
            current_page=page,
            sort_field=sort_field,
            sort_order=sort_order,
            filters=filters or {}
        )
        
        # Get page of stocks
        start_idx = pagination.offset
        end_idx = min(start_idx + page_size, len(sorted_stocks))
        page_stocks = sorted_stocks[start_idx:end_idx]
        
        # Generate cursor for next page
        if page_stocks and pagination.has_next:
            last_item = page_stocks[-1]
            pagination.cursor = self._generate_cursor(last_item, sort_field)
        
        return page_stocks, pagination
    
    async def cursor_paginate_stocks(
        self,
        stocks: List[Dict[str, Any]],
        cursor: Optional[str] = None,
        limit: int = 100,
        sort_field: str = "ticker",
        sort_order: str = "asc"
    ) -> Tuple[List[Dict], Optional[str]]:
        """
        Cursor-based pagination for efficient large dataset handling
        
        Args:
            stocks: Complete list of stocks
            cursor: Cursor from previous page
            limit: Number of items to return
            sort_field: Field to sort by
            sort_order: Sort order
            
        Returns:
            Tuple of (page stocks, next cursor)
        """
        # Sort stocks
        sorted_stocks = self._sort_stocks(stocks, sort_field, sort_order)
        
        # Find starting position based on cursor
        start_idx = 0
        if cursor:
            decoded_cursor = self._decode_cursor(cursor)
            for i, stock in enumerate(sorted_stocks):
                if stock.get(sort_field) > decoded_cursor['value']:
                    start_idx = i
                    break
        
        # Get page of stocks
        page_stocks = sorted_stocks[start_idx:start_idx + limit]
        
        # Generate next cursor
        next_cursor = None
        if len(page_stocks) == limit and start_idx + limit < len(sorted_stocks):
            last_item = page_stocks[-1]
            next_cursor = self._generate_cursor(last_item, sort_field)
        
        return page_stocks, next_cursor
    
    async def stream_stocks(
        self,
        stocks: List[Dict[str, Any]],
        chunk_size: int = 100,
        process_func: Optional[callable] = None
    ):
        """
        Stream stocks in chunks for memory-efficient processing
        
        Args:
            stocks: List of stocks
            chunk_size: Size of each chunk
            process_func: Optional async function to process each chunk
            
        Yields:
            Chunks of stocks
        """
        for i in range(0, len(stocks), chunk_size):
            chunk = stocks[i:i + chunk_size]
            
            if process_func:
                chunk = await process_func(chunk)
            
            yield chunk
            
            # Small delay to prevent overwhelming downstream
            await asyncio.sleep(0.01)
    
    async def redistribute_failed_batches(self) -> List[StockBatch]:
        """
        Redistribute failed batches to healthy nodes
        
        Returns:
            List of redistributed batches
        """
        if not self.failed_batches:
            return []
        
        # Get healthy nodes
        healthy_nodes = [
            node for node in self.nodes.values()
            if node.error_rate < 0.1 and node.available_capacity > 0
        ]
        
        if not healthy_nodes:
            logger.warning("No healthy nodes available for redistribution")
            return []
        
        redistributed = []
        
        for batch in self.failed_batches[:]:  # Copy to avoid modification during iteration
            # Increment retry count
            batch.retry_count += 1
            
            # Skip if too many retries
            if batch.retry_count > 3:
                logger.error(f"Batch {batch.batch_id} exceeded max retries")
                continue
            
            # Find best node for redistribution
            best_node = min(healthy_nodes, key=lambda n: n.load_factor)
            
            # Assign to node
            batch.assigned_node = best_node.node_id
            best_node.current_load += len(batch.stocks)
            
            redistributed.append(batch)
            self.failed_batches.remove(batch)
            
            logger.info(f"Redistributed batch {batch.batch_id} to node {best_node.node_id}")
        
        return redistributed
    
    async def optimize_distribution(
        self,
        current_distribution: List[StockBatch]
    ) -> List[StockBatch]:
        """
        Optimize existing distribution based on node performance
        
        Args:
            current_distribution: Current batch distribution
            
        Returns:
            Optimized distribution
        """
        # Calculate node efficiency scores
        node_scores = {}
        for node_id, node in self.nodes.items():
            efficiency = (
                node.processing_speed * (1 - node.error_rate) /
                (1 + node.average_latency_ms / 1000)
            )
            node_scores[node_id] = efficiency
        
        # Identify imbalanced batches
        optimized = []
        
        for batch in current_distribution:
            if not batch.assigned_node:
                continue
            
            current_node = self.nodes.get(batch.assigned_node)
            if not current_node:
                continue
            
            # Check if batch should be moved
            current_score = node_scores[batch.assigned_node]
            
            # Find better node
            better_nodes = [
                (node_id, score) for node_id, score in node_scores.items()
                if score > current_score * 1.2 and  # At least 20% better
                self.nodes[node_id].available_capacity >= len(batch.stocks)
            ]
            
            if better_nodes:
                # Move to best available node
                best_node_id = max(better_nodes, key=lambda x: x[1])[0]
                
                # Update loads
                current_node.current_load -= len(batch.stocks)
                self.nodes[best_node_id].current_load += len(batch.stocks)
                
                # Reassign batch
                batch.assigned_node = best_node_id
                logger.debug(f"Optimized batch {batch.batch_id}: {current_node.node_id} -> {best_node_id}")
            
            optimized.append(batch)
        
        return optimized
    
    # Distribution strategies
    
    async def _distribute_round_robin(
        self,
        batches: List[StockBatch]
    ) -> List[StockBatch]:
        """Round-robin distribution across nodes"""
        node_list = list(self.nodes.values())
        node_index = 0
        
        for batch in batches:
            # Find next available node
            attempts = 0
            while attempts < len(node_list):
                node = node_list[node_index]
                if node.available_capacity >= len(batch.stocks):
                    batch.assigned_node = node.node_id
                    node.current_load += len(batch.stocks)
                    break
                
                node_index = (node_index + 1) % len(node_list)
                attempts += 1
            
            if not batch.assigned_node:
                logger.warning(f"Could not assign batch {batch.batch_id}")
        
        return batches
    
    async def _distribute_weighted(
        self,
        batches: List[StockBatch]
    ) -> List[StockBatch]:
        """Weighted distribution based on node capacity"""
        # Calculate weights
        total_capacity = sum(node.capacity for node in self.nodes.values())
        
        for batch in batches:
            # Select node based on weighted probability
            rand_val = hash(batch.batch_id) % total_capacity
            cumulative = 0
            
            for node in self.nodes.values():
                cumulative += node.capacity
                if rand_val < cumulative and node.available_capacity >= len(batch.stocks):
                    batch.assigned_node = node.node_id
                    node.current_load += len(batch.stocks)
                    break
        
        return batches
    
    async def _distribute_hash_based(
        self,
        batches: List[StockBatch]
    ) -> List[StockBatch]:
        """Hash-based distribution for consistency"""
        node_list = list(self.nodes.values())
        
        for batch in batches:
            # Hash batch ID to determine node
            hash_val = int(hashlib.md5(batch.batch_id.encode()).hexdigest(), 16)
            node_index = hash_val % len(node_list)
            
            # Try to assign to hashed node
            node = node_list[node_index]
            if node.available_capacity >= len(batch.stocks):
                batch.assigned_node = node.node_id
                node.current_load += len(batch.stocks)
            else:
                # Fallback to least loaded node
                available_nodes = [
                    n for n in node_list
                    if n.available_capacity >= len(batch.stocks)
                ]
                if available_nodes:
                    best_node = min(available_nodes, key=lambda n: n.load_factor)
                    batch.assigned_node = best_node.node_id
                    best_node.current_load += len(batch.stocks)
        
        return batches
    
    async def _distribute_priority_based(
        self,
        batches: List[StockBatch]
    ) -> List[StockBatch]:
        """Priority-based distribution - high priority to best nodes"""
        # Sort batches by priority
        sorted_batches = sorted(
            batches,
            key=lambda b: b.priority.value if b.priority else 999
        )
        
        # Sort nodes by performance
        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (n.error_rate, n.average_latency_ms)
        )
        
        for batch in sorted_batches:
            # Assign high-priority batches to best nodes
            for node in sorted_nodes:
                if node.available_capacity >= len(batch.stocks):
                    batch.assigned_node = node.node_id
                    node.current_load += len(batch.stocks)
                    break
        
        return batches
    
    async def _distribute_adaptive(
        self,
        batches: List[StockBatch]
    ) -> List[StockBatch]:
        """Adaptive distribution based on real-time metrics"""
        for batch in batches:
            # Calculate scores for each node
            node_scores = []
            
            for node_id, node in self.nodes.items():
                if node.available_capacity < len(batch.stocks):
                    continue
                
                # Multi-factor scoring
                capacity_score = node.available_capacity / node.capacity
                performance_score = node.processing_speed * (1 - node.error_rate)
                latency_score = 1 / (1 + node.average_latency_ms / 100)
                
                # Check specialization match
                specialization_score = 1.0
                if node.specialization and batch.metadata.get('sector'):
                    if batch.metadata['sector'] in node.specialization:
                        specialization_score = 1.5
                
                total_score = (
                    capacity_score * 0.3 +
                    performance_score * 0.3 +
                    latency_score * 0.2 +
                    specialization_score * 0.2
                )
                
                heapq.heappush(node_scores, (-total_score, node_id))
            
            # Assign to best scoring node
            if node_scores:
                _, best_node_id = heapq.heappop(node_scores)
                batch.assigned_node = best_node_id
                self.nodes[best_node_id].current_load += len(batch.stocks)
        
        return batches
    
    async def _distribute_geographic(
        self,
        batches: List[StockBatch]
    ) -> List[StockBatch]:
        """Geographic distribution for regional optimization"""
        # Group nodes by region
        regional_nodes = defaultdict(list)
        for node in self.nodes.values():
            regional_nodes[node.geographic_region].append(node)
        
        for batch in batches:
            # Determine batch region (e.g., based on exchange)
            batch_region = batch.metadata.get('region', 'default')
            
            # Try to assign to same region
            if batch_region in regional_nodes:
                available_nodes = [
                    n for n in regional_nodes[batch_region]
                    if n.available_capacity >= len(batch.stocks)
                ]
                
                if available_nodes:
                    best_node = min(available_nodes, key=lambda n: n.load_factor)
                    batch.assigned_node = best_node.node_id
                    best_node.current_load += len(batch.stocks)
                    continue
            
            # Fallback to any available node
            all_available = [
                n for n in self.nodes.values()
                if n.available_capacity >= len(batch.stocks)
            ]
            
            if all_available:
                best_node = min(all_available, key=lambda n: n.load_factor)
                batch.assigned_node = best_node.node_id
                best_node.current_load += len(batch.stocks)
        
        return batches
    
    # Helper methods
    
    def _group_by_priority(
        self,
        stocks: List[Dict],
        priority_field: str
    ) -> Dict[StockPriority, List[Dict]]:
        """Group stocks by priority"""
        groups = defaultdict(list)
        
        for stock in stocks:
            priority_value = stock.get(priority_field, StockPriority.MEDIUM.value)
            
            # Convert to StockPriority enum
            if isinstance(priority_value, str):
                try:
                    priority = StockPriority[priority_value.upper()]
                except KeyError:
                    priority = StockPriority.MEDIUM
            elif isinstance(priority_value, int):
                # Map numeric priorities
                if priority_value <= 1:
                    priority = StockPriority.CRITICAL
                elif priority_value <= 2:
                    priority = StockPriority.HIGH
                elif priority_value <= 3:
                    priority = StockPriority.MEDIUM
                elif priority_value <= 4:
                    priority = StockPriority.LOW
                else:
                    priority = StockPriority.MINIMAL
            else:
                priority = StockPriority.MEDIUM
            
            groups[priority].append(stock)
        
        return dict(groups)
    
    def _create_batches(
        self,
        stocks: List[Dict],
        batch_size: int,
        priority: StockPriority
    ) -> List[StockBatch]:
        """Create batches from stock list"""
        batches = []
        
        for i in range(0, len(stocks), batch_size):
            batch_stocks = stocks[i:i + batch_size]
            batch = StockBatch(
                batch_id=f"{priority.value}_{datetime.utcnow().timestamp()}_{i}",
                stocks=batch_stocks,
                priority=priority,
                metadata={
                    'batch_index': i // batch_size,
                    'total_batches': math.ceil(len(stocks) / batch_size)
                }
            )
            batches.append(batch)
        
        return batches
    
    def _apply_filters(
        self,
        stocks: List[Dict],
        filters: Dict[str, Any]
    ) -> List[Dict]:
        """Apply filters to stock list"""
        filtered = stocks
        
        for field, value in filters.items():
            if isinstance(value, dict):
                # Range filter
                if 'min' in value:
                    filtered = [
                        s for s in filtered
                        if s.get(field, 0) >= value['min']
                    ]
                if 'max' in value:
                    filtered = [
                        s for s in filtered
                        if s.get(field, float('inf')) <= value['max']
                    ]
            elif isinstance(value, list):
                # In filter
                filtered = [
                    s for s in filtered
                    if s.get(field) in value
                ]
            else:
                # Exact match
                filtered = [
                    s for s in filtered
                    if s.get(field) == value
                ]
        
        return filtered
    
    def _sort_stocks(
        self,
        stocks: List[Dict],
        sort_field: str,
        sort_order: str
    ) -> List[Dict]:
        """Sort stocks by field"""
        reverse = sort_order.lower() == 'desc'
        
        return sorted(
            stocks,
            key=lambda x: x.get(sort_field, ''),
            reverse=reverse
        )
    
    def _generate_cursor(self, item: Dict, sort_field: str) -> str:
        """Generate cursor for pagination"""
        cursor_data = {
            'field': sort_field,
            'value': item.get(sort_field),
            'id': item.get('ticker', item.get('id'))
        }
        
        import base64
        import json
        
        cursor_json = json.dumps(cursor_data)
        cursor_bytes = cursor_json.encode('utf-8')
        cursor_b64 = base64.b64encode(cursor_bytes).decode('utf-8')
        
        return cursor_b64
    
    def _decode_cursor(self, cursor: str) -> Dict:
        """Decode pagination cursor"""
        import base64
        import json
        
        try:
            cursor_bytes = base64.b64decode(cursor.encode('utf-8'))
            cursor_json = cursor_bytes.decode('utf-8')
            return json.loads(cursor_json)
        except:
            return {}
    
    async def _discover_nodes(self):
        """Discover available processing nodes"""
        # In production, this would discover from service registry
        # For now, create mock nodes
        self.nodes = {
            'node1': ProcessingNode(
                node_id='node1',
                capacity=1000,
                processing_speed=1.2,
                specialization=['technology', 'healthcare'],
                geographic_region='us-east'
            ),
            'node2': ProcessingNode(
                node_id='node2',
                capacity=800,
                processing_speed=1.0,
                specialization=['finance', 'energy'],
                geographic_region='us-west'
            ),
            'node3': ProcessingNode(
                node_id='node3',
                capacity=1200,
                processing_speed=1.5,
                specialization=['consumer', 'industrial'],
                geographic_region='eu-central'
            )
        }
    
    async def _track_distribution(self, batches: List[StockBatch]):
        """Track distribution metrics"""
        for batch in batches:
            if batch.assigned_node:
                self.distribution_history[batch.assigned_node].append({
                    'batch_id': batch.batch_id,
                    'timestamp': datetime.utcnow(),
                    'stock_count': len(batch.stocks),
                    'priority': batch.priority.value
                })
                
                # Update metrics
                node_metrics = self.distribution_metrics[batch.assigned_node]
                node_metrics['total_distributed'] += len(batch.stocks)


# Global instance
_stock_distributor: Optional[StockDistributor] = None


async def get_stock_distributor() -> StockDistributor:
    """Get or create the global stock distributor"""
    global _stock_distributor
    if _stock_distributor is None:
        _stock_distributor = StockDistributor()
        await _stock_distributor.initialize()
    return _stock_distributor