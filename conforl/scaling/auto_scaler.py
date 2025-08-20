"""
Auto-scaling and load balancing for ConfoRL production deployments.
Generation 3: Intelligent scaling based on performance metrics and load.
"""

import time
import threading
import queue
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict

from ..utils.monitoring import MetricsCollector, record_metric, AlertLevel
from ..utils.circuit_breaker import CircuitBreaker


class ScalingDirection(Enum):
    """Scaling direction."""
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"


@dataclass
class WorkerNode:
    """Worker node in the ConfoRL cluster."""
    node_id: str
    capacity: int = 100
    current_load: int = 0
    response_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    last_health_check: float = 0.0
    status: str = "healthy"  # healthy, unhealthy, draining
    circuit_breaker: CircuitBreaker = field(default_factory=CircuitBreaker)
    
    def get_load_percentage(self) -> float:
        """Get current load percentage."""
        return (self.current_load / self.capacity) * 100 if self.capacity > 0 else 100
    
    def get_average_response_time(self) -> float:
        """Get average response time."""
        return sum(self.response_times) / len(self.response_times) if self.response_times else 0
    
    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return (
            self.status == "healthy" and
            self.get_load_percentage() < 90 and
            self.error_count < 5 and
            time.time() - self.last_health_check < 60
        )


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_replicas: int = 3
    max_replicas: int = 100
    target_cpu_utilization: float = 70.0
    target_memory_utilization: float = 80.0
    target_response_time_ms: float = 100.0
    scale_up_threshold: float = 85.0
    scale_down_threshold: float = 50.0
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    evaluation_window_seconds: int = 300


class LoadBalancer:
    """High-performance load balancer for ConfoRL workers."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
        """
        self.strategy = strategy
        self.workers = {}
        self.request_queue = queue.Queue()
        
        # Strategy-specific state
        self.round_robin_index = 0
        
        # Performance tracking
        self.metrics = MetricsCollector()
        self.request_counts = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        
        # Thread safety
        self._lock = threading.RLock()
    
    def add_worker(self, worker: WorkerNode) -> None:
        """Add worker to the load balancer.
        
        Args:
            worker: Worker node to add
        """
        with self._lock:
            self.workers[worker.node_id] = worker
            record_metric("workers_total", len(self.workers))
    
    def remove_worker(self, node_id: str) -> None:
        """Remove worker from the load balancer.
        
        Args:
            node_id: ID of worker to remove
        """
        with self._lock:
            if node_id in self.workers:
                del self.workers[node_id]
                record_metric("workers_total", len(self.workers))
    
    def get_next_worker(self) -> Optional[WorkerNode]:
        """Get next worker based on load balancing strategy.
        
        Returns:
            Selected worker or None if no healthy workers available
        """
        with self._lock:
            healthy_workers = [w for w in self.workers.values() if w.is_healthy()]
            
            if not healthy_workers:
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(healthy_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
                return self._least_response_time_select(healthy_workers)
            else:
                return healthy_workers[0]  # Fallback
    
    def _round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Round-robin worker selection."""
        worker = workers[self.round_robin_index % len(workers)]
        self.round_robin_index += 1
        return worker
    
    def _least_connections_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Least connections worker selection."""
        return min(workers, key=lambda w: w.current_load)
    
    def _weighted_round_robin_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Weighted round-robin based on capacity."""
        # Simple weighted selection based on available capacity
        weights = [max(1, w.capacity - w.current_load) for w in workers]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return workers[0]
        
        # Select based on weight
        import random
        rand_val = random.uniform(0, total_weight)
        cumulative = 0
        
        for worker, weight in zip(workers, weights):
            cumulative += weight
            if rand_val <= cumulative:
                return worker
        
        return workers[-1]  # Fallback
    
    def _least_response_time_select(self, workers: List[WorkerNode]) -> WorkerNode:
        """Least response time worker selection."""
        return min(workers, key=lambda w: w.get_average_response_time())
    
    def route_request(self, request_func: Callable, *args, **kwargs) -> Any:
        """Route request to best available worker.
        
        Args:
            request_func: Function to execute on worker
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Request result
        """
        start_time = time.time()
        
        worker = self.get_next_worker()
        if not worker:
            raise RuntimeError("No healthy workers available")
        
        try:
            # Track request
            worker.current_load += 1
            self.request_counts[worker.node_id] += 1
            
            # Execute request
            result = worker.circuit_breaker.call(request_func, *args, **kwargs)
            
            # Track success
            response_time = time.time() - start_time
            worker.response_times.append(response_time)
            self.response_times.append(response_time)
            
            record_metric("request_success_total", 1)
            record_metric("request_duration_seconds", response_time)
            
            return result
            
        except Exception as e:
            # Track error
            worker.error_count += 1
            record_metric("request_error_total", 1)
            raise
            
        finally:
            # Release load
            worker.current_load = max(0, worker.current_load - 1)
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            total_requests = sum(self.request_counts.values())
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            worker_stats = {}
            for worker in self.workers.values():
                worker_stats[worker.node_id] = {
                    'load_percentage': worker.get_load_percentage(),
                    'average_response_time': worker.get_average_response_time(),
                    'error_count': worker.error_count,
                    'status': worker.status,
                    'request_count': self.request_counts[worker.node_id]
                }
            
            return {
                'strategy': self.strategy.value,
                'total_workers': len(self.workers),
                'healthy_workers': len([w for w in self.workers.values() if w.is_healthy()]),
                'total_requests': total_requests,
                'average_response_time': avg_response_time,
                'worker_stats': worker_stats
            }


class AutoScaler:
    """Intelligent auto-scaler for ConfoRL deployments."""
    
    def __init__(self, policy: ScalingPolicy, load_balancer: LoadBalancer):
        """Initialize auto-scaler.
        
        Args:
            policy: Scaling policy
            load_balancer: Load balancer instance
        """
        self.policy = policy
        self.load_balancer = load_balancer
        
        # Scaling state
        self.current_replicas = policy.min_replicas
        self.last_scale_up_time = 0
        self.last_scale_down_time = 0
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=policy.evaluation_window_seconds)
        self.scaling_decisions = deque(maxlen=100)
        
        # Control
        self.is_running = False
        self._thread = None
        self._lock = threading.Lock()
    
    def start(self) -> None:
        """Start auto-scaling."""
        with self._lock:
            if self.is_running:
                return
            
            self.is_running = True
            self._thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self._thread.start()
    
    def stop(self) -> None:
        """Stop auto-scaling."""
        with self._lock:
            self.is_running = False
            if self._thread:
                self._thread.join(timeout=5)
    
    def _scaling_loop(self) -> None:
        """Main scaling loop."""
        while self.is_running:
            try:
                self._evaluate_scaling()
                time.sleep(30)  # Evaluate every 30 seconds
            except Exception as e:
                record_metric("autoscaler_error_total", 1)
                time.sleep(60)  # Back off on error
    
    def _evaluate_scaling(self) -> None:
        """Evaluate if scaling is needed."""
        # Collect current metrics
        current_metrics = self._collect_metrics()
        self.metrics_history.append(current_metrics)
        
        # Need enough data to make decisions
        if len(self.metrics_history) < 10:
            return
        
        # Calculate averages over evaluation window
        recent_metrics = list(self.metrics_history)[-min(10, len(self.metrics_history)):]
        
        avg_cpu = sum(m['cpu_utilization'] for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m['memory_utilization'] for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m['response_time_ms'] for m in recent_metrics) / len(recent_metrics)
        avg_error_rate = sum(m['error_rate'] for m in recent_metrics) / len(recent_metrics)
        
        # Determine scaling direction
        scaling_direction = self._determine_scaling_direction(
            avg_cpu, avg_memory, avg_response_time, avg_error_rate
        )
        
        # Execute scaling decision
        if scaling_direction == ScalingDirection.UP:
            self._scale_up()
        elif scaling_direction == ScalingDirection.DOWN:
            self._scale_down()
        
        # Record decision
        self.scaling_decisions.append({
            'timestamp': time.time(),
            'direction': scaling_direction.value,
            'replicas': self.current_replicas,
            'cpu': avg_cpu,
            'memory': avg_memory,
            'response_time': avg_response_time,
            'error_rate': avg_error_rate
        })
    
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # Get load balancer stats
        lb_stats = self.load_balancer.get_load_balancer_stats()
        
        # Calculate utilization metrics
        workers = self.load_balancer.workers.values()
        
        if workers:
            avg_cpu = sum(w.get_load_percentage() for w in workers) / len(workers)
            avg_memory = avg_cpu  # Simplified - in production, use actual memory metrics
            avg_response_time = sum(w.get_average_response_time() for w in workers) / len(workers)
            error_rate = sum(w.error_count for w in workers) / len(workers)
        else:
            avg_cpu = avg_memory = avg_response_time = error_rate = 0
        
        return {
            'cpu_utilization': avg_cpu,
            'memory_utilization': avg_memory,
            'response_time_ms': avg_response_time * 1000,  # Convert to ms
            'error_rate': error_rate,
            'healthy_workers': lb_stats['healthy_workers'],
            'total_workers': lb_stats['total_workers']
        }
    
    def _determine_scaling_direction(
        self, 
        cpu: float, 
        memory: float, 
        response_time: float, 
        error_rate: float
    ) -> ScalingDirection:
        """Determine scaling direction based on metrics.
        
        Args:
            cpu: CPU utilization percentage
            memory: Memory utilization percentage
            response_time: Average response time in ms
            error_rate: Error rate
            
        Returns:
            Scaling direction
        """
        current_time = time.time()
        
        # Check scale-up conditions
        scale_up_needed = (
            cpu > self.policy.scale_up_threshold or
            memory > self.policy.scale_up_threshold or
            response_time > self.policy.target_response_time_ms * 2 or
            error_rate > 0.05  # 5% error rate
        )
        
        # Check scale-down conditions
        scale_down_safe = (
            cpu < self.policy.scale_down_threshold and
            memory < self.policy.scale_down_threshold and
            response_time < self.policy.target_response_time_ms and
            error_rate < 0.01  # 1% error rate
        )
        
        # Apply cooldown periods
        can_scale_up = (
            current_time - self.last_scale_up_time > self.policy.scale_up_cooldown_seconds
        )
        can_scale_down = (
            current_time - self.last_scale_down_time > self.policy.scale_down_cooldown_seconds
        )
        
        # Make scaling decision
        if scale_up_needed and can_scale_up and self.current_replicas < self.policy.max_replicas:
            return ScalingDirection.UP
        elif scale_down_safe and can_scale_down and self.current_replicas > self.policy.min_replicas:
            return ScalingDirection.DOWN
        else:
            return ScalingDirection.STABLE
    
    def _scale_up(self) -> None:
        """Scale up the deployment."""
        if self.current_replicas >= self.policy.max_replicas:
            return
        
        # Calculate new replica count (scale by 20% or at least 1)
        new_replicas = max(
            self.current_replicas + 1,
            int(self.current_replicas * 1.2)
        )
        new_replicas = min(new_replicas, self.policy.max_replicas)
        
        # Add new workers
        for i in range(new_replicas - self.current_replicas):
            worker_id = f"worker_{int(time.time())}_{i}"
            worker = WorkerNode(node_id=worker_id, capacity=100)
            self.load_balancer.add_worker(worker)
        
        self.current_replicas = new_replicas
        self.last_scale_up_time = time.time()
        
        record_metric("autoscaler_scale_up_total", 1)
        record_metric("autoscaler_replicas", self.current_replicas)
    
    def _scale_down(self) -> None:
        """Scale down the deployment."""
        if self.current_replicas <= self.policy.min_replicas:
            return
        
        # Calculate new replica count (scale by 10% or at least 1)
        new_replicas = max(
            self.current_replicas - 1,
            int(self.current_replicas * 0.9)
        )
        new_replicas = max(new_replicas, self.policy.min_replicas)
        
        # Remove workers (preferably unhealthy ones first)
        workers_to_remove = self.current_replicas - new_replicas
        all_workers = list(self.load_balancer.workers.values())
        
        # Sort by health and load (remove unhealthy/high-load workers first)
        all_workers.sort(key=lambda w: (w.is_healthy(), -w.get_load_percentage()))
        
        for i in range(workers_to_remove):
            if i < len(all_workers):
                self.load_balancer.remove_worker(all_workers[i].node_id)
        
        self.current_replicas = new_replicas
        self.last_scale_down_time = time.time()
        
        record_metric("autoscaler_scale_down_total", 1)
        record_metric("autoscaler_replicas", self.current_replicas)
    
    def get_autoscaler_stats(self) -> Dict[str, Any]:
        """Get auto-scaler statistics."""
        return {
            'current_replicas': self.current_replicas,
            'min_replicas': self.policy.min_replicas,
            'max_replicas': self.policy.max_replicas,
            'last_scale_up_time': self.last_scale_up_time,
            'last_scale_down_time': self.last_scale_down_time,
            'recent_decisions': list(self.scaling_decisions)[-10:],
            'is_running': self.is_running,
            'policy': {
                'target_cpu_utilization': self.policy.target_cpu_utilization,
                'scale_up_threshold': self.policy.scale_up_threshold,
                'scale_down_threshold': self.policy.scale_down_threshold,
                'scale_up_cooldown_seconds': self.policy.scale_up_cooldown_seconds,
                'scale_down_cooldown_seconds': self.policy.scale_down_cooldown_seconds
            }
        }


def create_production_cluster(
    min_replicas: int = 3,
    max_replicas: int = 50,
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS
) -> Tuple[LoadBalancer, AutoScaler]:
    """Create a production-ready ConfoRL cluster.
    
    Args:
        min_replicas: Minimum number of replicas
        max_replicas: Maximum number of replicas
        strategy: Load balancing strategy
        
    Returns:
        Tuple of (load_balancer, auto_scaler)
    """
    # Create load balancer
    load_balancer = LoadBalancer(strategy)
    
    # Add initial workers
    for i in range(min_replicas):
        worker = WorkerNode(node_id=f"worker_initial_{i}", capacity=100)
        load_balancer.add_worker(worker)
    
    # Create scaling policy
    policy = ScalingPolicy(
        min_replicas=min_replicas,
        max_replicas=max_replicas,
        target_cpu_utilization=70.0,
        scale_up_threshold=85.0,
        scale_down_threshold=40.0
    )
    
    # Create auto-scaler
    auto_scaler = AutoScaler(policy, load_balancer)
    
    return load_balancer, auto_scaler