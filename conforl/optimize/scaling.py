"""Auto-scaling and load balancing utilities for production deployment."""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from collections import deque
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..utils.logging import get_logger

logger = get_logger(__name__)


class ScalingAction(Enum):
    """Enumeration of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    cpu_usage: float
    memory_usage: float
    request_rate: float
    response_time: float
    error_rate: float
    queue_length: int
    timestamp: float


@dataclass
class ScalingRule:
    """Rule for auto-scaling decisions."""
    metric_name: str
    threshold_up: float
    threshold_down: float
    duration: float  # seconds to maintain threshold before scaling
    cooldown: float  # seconds to wait after scaling action


class AutoScaler:
    """Automatic scaling system for ConfoRL deployments."""
    
    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        default_rules: bool = True,
        custom_rules: Optional[List[ScalingRule]] = None
    ):
        """Initialize auto-scaler.
        
        Args:
            min_instances: Minimum number of instances
            max_instances: Maximum number of instances
            default_rules: Whether to use default scaling rules
            custom_rules: Custom scaling rules
        """
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        
        # Scaling rules
        self.rules = custom_rules or []
        if default_rules:
            self._add_default_rules()
        
        # Metrics tracking
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = []
        
        # State tracking
        self.rule_timers = {}  # Track how long thresholds have been exceeded
        self.last_scaling_action = 0  # Timestamp of last scaling action
        self.scaling_lock = threading.Lock()
        
        # Active monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitoring_interval = 30.0  # seconds
        
        logger.info(f"Auto-scaler initialized: {min_instances}-{max_instances} instances")
    
    def _add_default_rules(self):
        """Add default scaling rules."""
        default_rules = [
            ScalingRule(
                metric_name="cpu_usage",
                threshold_up=80.0,
                threshold_down=30.0,
                duration=60.0,  # 1 minute
                cooldown=300.0  # 5 minutes
            ),
            ScalingRule(
                metric_name="memory_usage",
                threshold_up=85.0,
                threshold_down=40.0,
                duration=90.0,  # 1.5 minutes
                cooldown=300.0
            ),
            ScaringRule(
                metric_name="response_time",
                threshold_up=2.0,  # 2 seconds
                threshold_down=0.5,  # 0.5 seconds
                duration=120.0,  # 2 minutes
                cooldown=180.0  # 3 minutes
            ),
            ScalingRule(
                metric_name="queue_length",
                threshold_up=50,
                threshold_down=10,
                duration=30.0,  # 30 seconds
                cooldown=120.0  # 2 minutes
            )
        ]
        
        self.rules.extend(default_rules)
        logger.info(f"Added {len(default_rules)} default scaling rules")
    
    def add_rule(self, rule: ScalingRule):
        """Add custom scaling rule.
        
        Args:
            rule: Scaling rule to add
        """
        self.rules.append(rule)
        logger.info(f"Added scaling rule for {rule.metric_name}")
    
    def update_metrics(self, metrics: ScalingMetrics):
        """Update scaling metrics.
        
        Args:
            metrics: Current system metrics
        """
        self.metrics_history.append(metrics)
        
        # Check if scaling action is needed
        with self.scaling_lock:
            scaling_action = self._evaluate_scaling_rules(metrics)
            
            if scaling_action != ScalingAction.NO_ACTION:
                self._execute_scaling_action(scaling_action, metrics)
    
    def _evaluate_scaling_rules(self, metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate scaling rules against current metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            Recommended scaling action
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scaling_action < min(rule.cooldown for rule in self.rules):
            return ScalingAction.NO_ACTION
        
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in self.rules:
            metric_value = getattr(metrics, rule.metric_name, None)
            if metric_value is None:
                continue
            
            rule_key = f"{rule.metric_name}_{rule.threshold_up}_{rule.threshold_down}"
            
            # Check scale-up condition
            if metric_value > rule.threshold_up:
                if rule_key not in self.rule_timers:
                    self.rule_timers[rule_key] = current_time
                elif current_time - self.rule_timers[rule_key] >= rule.duration:
                    scale_up_votes += 1
            
            # Check scale-down condition
            elif metric_value < rule.threshold_down:
                if rule_key not in self.rule_timers:
                    self.rule_timers[rule_key] = current_time
                elif current_time - self.rule_timers[rule_key] >= rule.duration:
                    scale_down_votes += 1
            
            else:
                # Reset timer if threshold not exceeded
                if rule_key in self.rule_timers:
                    del self.rule_timers[rule_key]
        
        # Determine scaling action
        if scale_up_votes > 0 and self.current_instances < self.max_instances:
            return ScalingAction.SCALE_UP
        elif scale_down_votes > 0 and self.current_instances > self.min_instances:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION
    
    def _execute_scaling_action(self, action: ScalingAction, metrics: ScalingMetrics):
        """Execute scaling action.
        
        Args:
            action: Scaling action to execute
            metrics: Current metrics that triggered action
        """
        old_instances = self.current_instances
        
        if action == ScalingAction.SCALE_UP:
            self.current_instances = min(self.current_instances + 1, self.max_instances)
        elif action == ScalingAction.SCALE_DOWN:
            self.current_instances = max(self.current_instances - 1, self.min_instances)
        
        if self.current_instances != old_instances:
            # Record scaling event
            scaling_event = {
                'timestamp': time.time(),
                'action': action.value,
                'old_instances': old_instances,
                'new_instances': self.current_instances,
                'trigger_metrics': {
                    'cpu_usage': metrics.cpu_usage,
                    'memory_usage': metrics.memory_usage,
                    'response_time': metrics.response_time,
                    'queue_length': metrics.queue_length
                }
            }
            
            self.scaling_history.append(scaling_event)
            self.last_scaling_action = time.time()
            
            # Clear rule timers after scaling
            self.rule_timers.clear()
            
            logger.info(f"Scaled {action.value}: {old_instances} -> {self.current_instances} instances")
    
    def get_current_scale(self) -> int:
        """Get current number of instances.
        
        Returns:
            Current instance count
        """
        return self.current_instances
    
    def get_scaling_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get scaling history.
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of scaling events
        """
        history = self.scaling_history.copy()
        if limit:
            history = history[-limit:]
        return history
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics.
        
        Returns:
            Dictionary with scaling statistics
        """
        if not self.scaling_history:
            return {
                'total_scaling_events': 0,
                'scale_up_events': 0,
                'scale_down_events': 0,
                'current_instances': self.current_instances,
                'min_instances': self.min_instances,
                'max_instances': self.max_instances
            }
        
        scale_up_count = sum(1 for event in self.scaling_history if event['action'] == 'scale_up')
        scale_down_count = sum(1 for event in self.scaling_history if event['action'] == 'scale_down')
        
        return {
            'total_scaling_events': len(self.scaling_history),
            'scale_up_events': scale_up_count,
            'scale_down_events': scale_down_count,
            'current_instances': self.current_instances,
            'min_instances': self.min_instances,
            'max_instances': self.max_instances,
            'last_scaling_time': self.scaling_history[-1]['timestamp'] if self.scaling_history else None
        }


class LoadBalancer:
    """Load balancer for distributing work across multiple instances."""
    
    def __init__(
        self,
        balancing_strategy: str = "round_robin",
        health_check_interval: float = 30.0
    ):
        """Initialize load balancer.
        
        Args:
            balancing_strategy: Strategy for load balancing ('round_robin', 'least_connections', 'weighted')
            health_check_interval: Interval for health checks in seconds
        """
        self.balancing_strategy = balancing_strategy
        self.health_check_interval = health_check_interval
        
        # Instance management
        self.instances = {}  # instance_id -> instance_info
        self.instance_health = {}  # instance_id -> health_status
        self.instance_load = {}  # instance_id -> current_load
        
        # Round-robin state
        self.round_robin_index = 0
        
        # Health checking
        self.health_check_thread = None
        self.health_check_active = False
        
        # Load balancing statistics
        self.request_count = 0
        self.routing_stats = {}
        
        logger.info(f"Load balancer initialized with {balancing_strategy} strategy")
    
    def register_instance(
        self,
        instance_id: str,
        instance_info: Dict[str, Any],
        weight: float = 1.0
    ):
        """Register a new instance.
        
        Args:
            instance_id: Unique identifier for instance
            instance_info: Instance configuration and connection info
            weight: Weight for weighted load balancing
        """
        self.instances[instance_id] = {
            'info': instance_info,
            'weight': weight,
            'registered_at': time.time()
        }
        
        self.instance_health[instance_id] = True  # Assume healthy initially
        self.instance_load[instance_id] = 0
        self.routing_stats[instance_id] = {
            'requests_routed': 0,
            'last_request_time': None,
            'avg_response_time': 0.0,
            'error_count': 0
        }
        
        logger.info(f"Registered instance: {instance_id}")
    
    def unregister_instance(self, instance_id: str):
        """Unregister an instance.
        
        Args:
            instance_id: ID of instance to unregister
        """
        if instance_id in self.instances:
            del self.instances[instance_id]
            del self.instance_health[instance_id]
            del self.instance_load[instance_id]
            del self.routing_stats[instance_id]
            
            logger.info(f"Unregistered instance: {instance_id}")
    
    def get_next_instance(self) -> Optional[str]:
        """Get next instance ID for request routing.
        
        Returns:
            Instance ID to route request to, or None if no healthy instances
        """
        healthy_instances = [
            instance_id for instance_id, health in self.instance_health.items()
            if health
        ]
        
        if not healthy_instances:
            logger.warning("No healthy instances available")
            return None
        
        if self.balancing_strategy == "round_robin":
            return self._round_robin_selection(healthy_instances)
        elif self.balancing_strategy == "least_connections":
            return self._least_connections_selection(healthy_instances)
        elif self.balancing_strategy == "weighted":
            return self._weighted_selection(healthy_instances)
        else:
            # Default to round-robin
            return self._round_robin_selection(healthy_instances)
    
    def _round_robin_selection(self, healthy_instances: List[str]) -> str:
        """Select instance using round-robin strategy."""
        if not healthy_instances:
            return None
        
        selected = healthy_instances[self.round_robin_index % len(healthy_instances)]
        self.round_robin_index += 1
        return selected
    
    def _least_connections_selection(self, healthy_instances: List[str]) -> str:
        """Select instance with least current connections."""
        if not healthy_instances:
            return None
        
        return min(healthy_instances, key=lambda x: self.instance_load.get(x, 0))
    
    def _weighted_selection(self, healthy_instances: List[str]) -> str:
        """Select instance using weighted random selection."""
        if not healthy_instances:
            return None
        
        weights = [self.instances[instance_id]['weight'] for instance_id in healthy_instances]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return healthy_instances[0]
        
        # Weighted random selection
        random_value = np.random.random() * total_weight
        cumulative_weight = 0
        
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if random_value <= cumulative_weight:
                return healthy_instances[i]
        
        return healthy_instances[-1]  # Fallback
    
    def update_instance_load(self, instance_id: str, load: int):
        """Update current load for an instance.
        
        Args:
            instance_id: Instance ID
            load: Current load (e.g., number of active connections)
        """
        if instance_id in self.instance_load:
            self.instance_load[instance_id] = load
    
    def update_instance_health(self, instance_id: str, is_healthy: bool):
        """Update health status for an instance.
        
        Args:
            instance_id: Instance ID
            is_healthy: Whether instance is healthy
        """
        if instance_id in self.instance_health:
            old_health = self.instance_health[instance_id]
            self.instance_health[instance_id] = is_healthy
            
            if old_health != is_healthy:
                status = "healthy" if is_healthy else "unhealthy"
                logger.info(f"Instance {instance_id} marked as {status}")
    
    def record_request(
        self,
        instance_id: str,
        response_time: float,
        success: bool = True
    ):
        """Record request statistics for an instance.
        
        Args:
            instance_id: Instance that handled the request
            response_time: Response time in seconds
            success: Whether request was successful
        """
        if instance_id not in self.routing_stats:
            return
        
        stats = self.routing_stats[instance_id]
        stats['requests_routed'] += 1
        stats['last_request_time'] = time.time()
        
        # Update average response time
        if stats['avg_response_time'] == 0:
            stats['avg_response_time'] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            stats['avg_response_time'] = (
                alpha * response_time + (1 - alpha) * stats['avg_response_time']
            )
        
        if not success:
            stats['error_count'] += 1
        
        self.request_count += 1
    
    def get_load_balancing_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics.
        
        Returns:
            Dictionary with load balancing statistics
        """
        total_requests = sum(
            stats['requests_routed'] for stats in self.routing_stats.values()
        )
        
        instance_stats = {}
        for instance_id, stats in self.routing_stats.items():
            instance_stats[instance_id] = {
                'requests_routed': stats['requests_routed'],
                'request_percentage': (
                    stats['requests_routed'] / total_requests * 100
                    if total_requests > 0 else 0
                ),
                'avg_response_time': stats['avg_response_time'],
                'error_count': stats['error_count'],
                'current_load': self.instance_load.get(instance_id, 0),
                'is_healthy': self.instance_health.get(instance_id, False)
            }
        
        return {
            'total_instances': len(self.instances),
            'healthy_instances': sum(self.instance_health.values()),
            'total_requests': total_requests,
            'balancing_strategy': self.balancing_strategy,
            'instance_stats': instance_stats
        }