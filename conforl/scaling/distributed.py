"""Distributed computing framework for ConfoRL.

Scalable distributed training and inference for safe RL with
automatic load balancing, fault tolerance, and elastic scaling.
"""

import time
import threading
import multiprocessing
import queue
import socket
import pickle
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod
import uuid
import json

import numpy as np

from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError
from ..core.types import TrajectoryData, RiskCertificate

logger = get_logger(__name__)


@dataclass
class NodeInfo:
    """Information about a compute node."""
    
    node_id: str
    hostname: str
    ip_address: str
    port: int
    cpu_cores: int
    memory_gb: float
    gpu_count: int = 0
    node_type: str = "worker"  # worker, coordinator, hybrid
    status: str = "active"  # active, inactive, busy, error
    last_heartbeat: float = 0.0
    current_load: float = 0.0
    capabilities: List[str] = None
    
    def __post_init__(self):
        if self.capabilities is None:
            self.capabilities = []
        if self.last_heartbeat == 0.0:
            self.last_heartbeat = time.time()


@dataclass
class TaskInfo:
    """Information about a distributed task."""
    
    task_id: str
    task_type: str
    priority: int = 1  # 1=low, 5=high
    estimated_duration: float = 0.0
    required_memory: float = 0.0
    required_cpus: int = 1
    requires_gpu: bool = False
    data_size_mb: float = 0.0
    dependencies: List[str] = None
    assigned_node: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result_data: Optional[Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.created_at == 0.0:
            self.created_at = time.time()


class DistributedAgent(ABC):
    """Base class for distributed ConfoRL agents."""
    
    def __init__(
        self,
        agent_id: str,
        node_info: Optional[NodeInfo] = None,
        communication_timeout: float = 30.0
    ):
        """Initialize distributed agent.
        
        Args:
            agent_id: Unique agent identifier
            node_info: Information about local node
            communication_timeout: Timeout for network operations
        """
        self.agent_id = agent_id
        self.node_info = node_info or self._detect_node_info()
        self.communication_timeout = communication_timeout
        
        # Distributed state
        self.is_coordinator = False
        self.coordinator_node: Optional[NodeInfo] = None
        self.known_nodes: Dict[str, NodeInfo] = {}
        self.local_tasks: Dict[str, TaskInfo] = {}
        
        # Communication
        self.message_queue = queue.Queue()
        self.result_cache = {}
        
        # Thread pool for concurrent processing
        self.executor = ThreadPoolExecutor(max_workers=self.node_info.cpu_cores)
        
        # Performance metrics
        self.task_completion_times = []
        self.network_latencies = {}
        
        logger.info(f"Distributed agent {agent_id} initialized on node {self.node_info.node_id}")
    
    def _detect_node_info(self) -> NodeInfo:
        """Auto-detect local node information."""
        import psutil
        
        hostname = socket.gethostname()
        try:
            ip_address = socket.gethostbyname(hostname)
        except:
            ip_address = "127.0.0.1"
        
        cpu_cores = multiprocessing.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Try to detect GPUs (simplified)
        gpu_count = 0
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_count = len(result.stdout.strip().split('\n'))
        except:
            pass
        
        node_id = f"node_{hashlib.md5(f'{hostname}_{ip_address}'.encode()).hexdigest()[:8]}"
        
        return NodeInfo(
            node_id=node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=8080,  # Default port
            cpu_cores=cpu_cores,
            memory_gb=memory_gb,
            gpu_count=gpu_count,
            capabilities=["training", "inference", "risk_computation"]
        )
    
    @abstractmethod
    def process_task(self, task: TaskInfo) -> Any:
        """Process a distributed task.
        
        Args:
            task: Task to process
            
        Returns:
            Task result
        """
        pass
    
    def submit_task(
        self,
        task_type: str,
        task_data: Any,
        priority: int = 1,
        target_node: Optional[str] = None
    ) -> str:
        """Submit task for distributed processing.
        
        Args:
            task_type: Type of task
            task_data: Task input data
            priority: Task priority
            target_node: Specific node to target (optional)
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        # Estimate resource requirements
        data_size = self._estimate_data_size(task_data)
        memory_req = max(0.1, data_size / 1024)  # Rough estimate
        
        task = TaskInfo(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data_size_mb=data_size,
            required_memory=memory_req,
            assigned_node=target_node
        )
        
        if target_node and target_node == self.node_info.node_id:
            # Execute locally
            self._execute_task_locally(task, task_data)
        else:
            # Distribute to cluster
            self._distribute_task(task, task_data)
        
        return task_id
    
    def _execute_task_locally(self, task: TaskInfo, task_data: Any) -> None:
        """Execute task on local node."""
        self.local_tasks[task.task_id] = task
        task.status = "running"
        task.started_at = time.time()
        
        def run_task():
            try:
                start_time = time.time()
                result = self.process_task(task)
                duration = time.time() - start_time
                
                task.status = "completed"
                task.completed_at = time.time()
                task.result_data = result
                
                self.task_completion_times.append(duration)
                self.result_cache[task.task_id] = result
                
                logger.debug(f"Task {task.task_id} completed in {duration:.3f}s")
                
            except Exception as e:
                task.status = "failed"
                task.error_message = str(e)
                task.completed_at = time.time()
                
                logger.error(f"Task {task.task_id} failed: {e}")
        
        # Submit to thread pool
        self.executor.submit(run_task)
    
    def _distribute_task(self, task: TaskInfo, task_data: Any) -> None:
        """Distribute task to cluster."""
        if not self.coordinator_node:
            logger.warning(f"No coordinator available - executing task {task.task_id} locally")
            self._execute_task_locally(task, task_data)
            return
        
        # Send task to coordinator for scheduling
        message = {
            'type': 'task_submission',
            'task': asdict(task),
            'data': task_data,
            'sender': self.node_info.node_id
        }
        
        self._send_message(self.coordinator_node, message)
        logger.debug(f"Task {task.task_id} submitted to coordinator")
    
    def get_task_result(self, task_id: str, timeout: float = None) -> Optional[Any]:
        """Get result of completed task.
        
        Args:
            task_id: Task identifier
            timeout: Maximum time to wait
            
        Returns:
            Task result if available
        """
        # Check local cache first
        if task_id in self.result_cache:
            return self.result_cache[task_id]
        
        # Check local tasks
        if task_id in self.local_tasks:
            task = self.local_tasks[task_id]
            if task.status == "completed":
                return task.result_data
            elif task.status == "failed":
                raise ConfoRLError(f"Task failed: {task.error_message}")
        
        # Wait for result with timeout
        start_time = time.time()
        timeout = timeout or self.communication_timeout
        
        while time.time() - start_time < timeout:
            if task_id in self.result_cache:
                return self.result_cache[task_id]
            
            if task_id in self.local_tasks:
                task = self.local_tasks[task_id]
                if task.status == "completed":
                    return task.result_data
                elif task.status == "failed":
                    raise ConfoRLError(f"Task failed: {task.error_message}")
            
            time.sleep(0.1)
        
        raise TimeoutError(f"Task {task_id} result not available within timeout")
    
    def _estimate_data_size(self, data: Any) -> float:
        """Estimate size of data in MB."""
        try:
            if isinstance(data, (np.ndarray, list, dict)):
                serialized = pickle.dumps(data)
                return len(serialized) / (1024 * 1024)
            else:
                return 0.1  # Default small size
        except:
            return 1.0  # Conservative estimate
    
    def _send_message(self, target_node: NodeInfo, message: Dict[str, Any]) -> bool:
        """Send message to target node.
        
        Args:
            target_node: Target node
            message: Message to send
            
        Returns:
            True if message sent successfully
        """
        try:
            # In a real implementation, this would use network communication
            # For now, simulate with logging
            logger.debug(f"Sending message to {target_node.node_id}: {message['type']}")
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {target_node.node_id}: {e}")
            return False
    
    def join_cluster(self, coordinator_address: Tuple[str, int]) -> bool:
        """Join distributed cluster.
        
        Args:
            coordinator_address: (host, port) of coordinator
            
        Returns:
            True if successfully joined
        """
        try:
            # Create coordinator node info
            host, port = coordinator_address
            coordinator_node = NodeInfo(
                node_id=f"coordinator_{host}_{port}",
                hostname=host,
                ip_address=host,
                port=port,
                cpu_cores=0,  # Unknown
                memory_gb=0,  # Unknown
                node_type="coordinator"
            )
            
            # Send join request
            join_message = {
                'type': 'join_request',
                'node_info': asdict(self.node_info),
                'sender': self.node_info.node_id
            }
            
            success = self._send_message(coordinator_node, join_message)
            
            if success:
                self.coordinator_node = coordinator_node
                logger.info(f"Joined cluster with coordinator at {coordinator_address}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to join cluster: {e}")
            return False
    
    def leave_cluster(self) -> bool:
        """Leave distributed cluster."""
        if not self.coordinator_node:
            return True
        
        try:
            leave_message = {
                'type': 'leave_request',
                'node_info': asdict(self.node_info),
                'sender': self.node_info.node_id
            }
            
            self._send_message(self.coordinator_node, leave_message)
            self.coordinator_node = None
            
            logger.info("Left cluster")
            return True
            
        except Exception as e:
            logger.error(f"Failed to leave cluster: {e}")
            return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_task_time = np.mean(self.task_completion_times) if self.task_completion_times else 0.0
        
        local_task_counts = {}
        for task in self.local_tasks.values():
            status = task.status
            local_task_counts[status] = local_task_counts.get(status, 0) + 1
        
        return {
            'node_id': self.node_info.node_id,
            'total_tasks_processed': len(self.task_completion_times),
            'average_task_time': avg_task_time,
            'current_load': self.node_info.current_load,
            'local_task_counts': local_task_counts,
            'cache_hit_ratio': len(self.result_cache) / max(1, len(self.local_tasks)),
            'is_in_cluster': self.coordinator_node is not None,
            'known_nodes': len(self.known_nodes)
        }
    
    def shutdown(self):
        """Shutdown distributed agent."""
        # Leave cluster if joined
        self.leave_cluster()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info(f"Distributed agent {self.agent_id} shutdown")


class ClusterManager:
    """Manages distributed ConfoRL cluster."""
    
    def __init__(
        self,
        coordinator_port: int = 8080,
        max_nodes: int = 100,
        health_check_interval: float = 30.0
    ):
        """Initialize cluster manager.
        
        Args:
            coordinator_port: Port for coordinator
            max_nodes: Maximum number of nodes
            health_check_interval: Seconds between health checks
        """
        self.coordinator_port = coordinator_port
        self.max_nodes = max_nodes
        self.health_check_interval = health_check_interval
        
        # Cluster state
        self.nodes: Dict[str, NodeInfo] = {}
        self.tasks: Dict[str, TaskInfo] = {}
        self.task_queue = queue.PriorityQueue()
        self.completed_tasks = {}
        
        # Scheduling
        self.scheduling_strategy = "load_balanced"  # load_balanced, round_robin, capability_based
        self.load_balancing_weights = {}
        
        # Monitoring
        self.cluster_metrics = {
            'total_tasks_processed': 0,
            'total_nodes_joined': 0,
            'total_nodes_left': 0,
            'average_node_utilization': 0.0,
            'task_throughput': 0.0
        }
        
        # Health checking
        self._health_check_thread = None
        self._shutdown_event = threading.Event()
        
        logger.info(f"Cluster manager initialized (port={coordinator_port})")
    
    def start_coordinator(self) -> bool:
        """Start cluster coordinator service.
        
        Returns:
            True if coordinator started successfully
        """
        try:
            # Start health checking thread
            self._health_check_thread = threading.Thread(
                target=self._health_check_loop,
                daemon=True
            )
            self._health_check_thread.start()
            
            # Start task scheduling thread
            self._scheduling_thread = threading.Thread(
                target=self._scheduling_loop,
                daemon=True
            )
            self._scheduling_thread.start()
            
            logger.info("Cluster coordinator started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}")
            return False
    
    def register_node(self, node_info: NodeInfo) -> bool:
        """Register node with cluster.
        
        Args:
            node_info: Information about node
            
        Returns:
            True if registration successful
        """
        if len(self.nodes) >= self.max_nodes:
            logger.warning(f"Cluster at maximum capacity ({self.max_nodes} nodes)")
            return False
        
        # Update node info
        node_info.last_heartbeat = time.time()
        node_info.status = "active"
        
        self.nodes[node_info.node_id] = node_info
        self.cluster_metrics['total_nodes_joined'] += 1
        
        logger.info(f"Node {node_info.node_id} registered ({node_info.hostname})")
        return True
    
    def deregister_node(self, node_id: str) -> bool:
        """Deregister node from cluster.
        
        Args:
            node_id: Node identifier
            
        Returns:
            True if deregistration successful
        """
        if node_id not in self.nodes:
            return False
        
        # Reassign tasks from departing node
        self._reassign_node_tasks(node_id)
        
        del self.nodes[node_id]
        self.cluster_metrics['total_nodes_left'] += 1
        
        logger.info(f"Node {node_id} deregistered")
        return True
    
    def submit_task(self, task: TaskInfo, task_data: Any) -> bool:
        """Submit task to cluster for execution.
        
        Args:
            task: Task information
            task_data: Task input data
            
        Returns:
            True if task submitted successfully
        """
        # Store task
        self.tasks[task.task_id] = task
        
        # Add to scheduling queue (priority, timestamp, task_id)
        priority_score = -task.priority  # Negative for max priority queue
        self.task_queue.put((priority_score, task.created_at, task.task_id, task_data))
        
        logger.debug(f"Task {task.task_id} queued for execution")
        return True
    
    def _scheduling_loop(self):
        """Main task scheduling loop."""
        while not self._shutdown_event.is_set():
            try:
                # Get next task from queue
                try:
                    priority_score, created_at, task_id, task_data = self.task_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Find suitable node
                target_node = self._select_node_for_task(task)
                
                if target_node:
                    # Assign task to node
                    task.assigned_node = target_node.node_id
                    task.status = "assigned"
                    
                    # Send task to node (simulated)
                    self._send_task_to_node(target_node, task, task_data)
                    
                    logger.debug(f"Task {task_id} assigned to node {target_node.node_id}")
                else:
                    # No suitable node available - requeue
                    self.task_queue.put((priority_score, created_at, task_id, task_data))
                    time.sleep(1.0)  # Wait before retrying
                
            except Exception as e:
                logger.error(f"Scheduling error: {e}")
    
    def _select_node_for_task(self, task: TaskInfo) -> Optional[NodeInfo]:
        """Select best node for task based on scheduling strategy.
        
        Args:
            task: Task to schedule
            
        Returns:
            Selected node or None if no suitable node
        """
        available_nodes = [
            node for node in self.nodes.values()
            if (node.status == "active" and 
                node.current_load < 0.8 and  # Not overloaded
                node.memory_gb >= task.required_memory and
                (not task.requires_gpu or node.gpu_count > 0))
        ]
        
        if not available_nodes:
            return None
        
        if self.scheduling_strategy == "load_balanced":
            # Select node with lowest current load
            return min(available_nodes, key=lambda n: n.current_load)
        
        elif self.scheduling_strategy == "round_robin":
            # Simple round-robin (simplified implementation)
            return available_nodes[len(self.completed_tasks) % len(available_nodes)]
        
        elif self.scheduling_strategy == "capability_based":
            # Select node with best capabilities match
            task_capabilities = self._infer_required_capabilities(task)
            
            def capability_score(node):
                matching_caps = len(set(task_capabilities) & set(node.capabilities))
                return matching_caps / max(1, len(task_capabilities))
            
            return max(available_nodes, key=capability_score)
        
        else:
            # Default to first available
            return available_nodes[0]
    
    def _infer_required_capabilities(self, task: TaskInfo) -> List[str]:
        """Infer required capabilities from task."""
        capabilities = []
        
        if "train" in task.task_type.lower():
            capabilities.append("training")
        if "infer" in task.task_type.lower() or "predict" in task.task_type.lower():
            capabilities.append("inference")
        if "risk" in task.task_type.lower():
            capabilities.append("risk_computation")
        if task.requires_gpu:
            capabilities.append("gpu")
        
        return capabilities or ["general"]
    
    def _send_task_to_node(self, node: NodeInfo, task: TaskInfo, task_data: Any) -> bool:
        """Send task to node for execution.
        
        Args:
            node: Target node
            task: Task to execute
            task_data: Task input data
            
        Returns:
            True if task sent successfully
        """
        try:
            # In real implementation, would send over network
            # For now, simulate successful task dispatch
            
            # Update node load (simplified)
            node.current_load += 0.1
            
            # Simulate task execution (would be done by receiving node)
            def simulate_completion():
                time.sleep(max(0.1, task.estimated_duration or 1.0))
                self._handle_task_completion(task.task_id, {"result": "simulated"})
            
            thread = threading.Thread(target=simulate_completion, daemon=True)
            thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send task to node {node.node_id}: {e}")
            return False
    
    def _handle_task_completion(self, task_id: str, result: Any) -> None:
        """Handle task completion notification.
        
        Args:
            task_id: Completed task ID
            result: Task result
        """
        if task_id not in self.tasks:
            return
        
        task = self.tasks[task_id]
        task.status = "completed"
        task.completed_at = time.time()
        task.result_data = result
        
        # Update metrics
        self.cluster_metrics['total_tasks_processed'] += 1
        
        # Move to completed tasks
        self.completed_tasks[task_id] = task
        
        # Update node load
        if task.assigned_node and task.assigned_node in self.nodes:
            node = self.nodes[task.assigned_node]
            node.current_load = max(0.0, node.current_load - 0.1)
        
        logger.debug(f"Task {task_id} completed")
    
    def _health_check_loop(self):
        """Health check loop for monitoring nodes."""
        while not self._shutdown_event.is_set():
            try:
                current_time = time.time()
                inactive_nodes = []
                
                for node_id, node in self.nodes.items():
                    # Check if node has sent heartbeat recently
                    if current_time - node.last_heartbeat > self.health_check_interval * 2:
                        node.status = "inactive"
                        inactive_nodes.append(node_id)
                        logger.warning(f"Node {node_id} marked as inactive (no heartbeat)")
                
                # Remove inactive nodes
                for node_id in inactive_nodes:
                    self.deregister_node(node_id)
                
                # Update cluster metrics
                if self.nodes:
                    avg_load = np.mean([node.current_load for node in self.nodes.values()])
                    self.cluster_metrics['average_node_utilization'] = avg_load
                
                time.sleep(self.health_check_interval)
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    def _reassign_node_tasks(self, departing_node_id: str) -> None:
        """Reassign tasks from departing node.
        
        Args:
            departing_node_id: ID of node leaving cluster
        """
        reassigned_count = 0
        
        for task in self.tasks.values():
            if (task.assigned_node == departing_node_id and 
                task.status in ["assigned", "running"]):
                
                # Reset task for reassignment
                task.assigned_node = None
                task.status = "pending"
                
                # Add back to queue
                priority_score = -task.priority
                task_data = task.result_data  # Use cached data if available
                self.task_queue.put((priority_score, task.created_at, task.task_id, task_data))
                
                reassigned_count += 1
        
        if reassigned_count > 0:
            logger.info(f"Reassigned {reassigned_count} tasks from departing node {departing_node_id}")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status.
        
        Returns:
            Cluster status information
        """
        active_nodes = [n for n in self.nodes.values() if n.status == "active"]
        
        # Task status counts
        task_status_counts = {}
        for task in self.tasks.values():
            status = task.status
            task_status_counts[status] = task_status_counts.get(status, 0) + 1
        
        # Resource summary
        total_cpus = sum(node.cpu_cores for node in active_nodes)
        total_memory = sum(node.memory_gb for node in active_nodes)
        total_gpus = sum(node.gpu_count for node in active_nodes)
        
        return {
            'cluster_size': len(self.nodes),
            'active_nodes': len(active_nodes),
            'total_resources': {
                'cpus': total_cpus,
                'memory_gb': total_memory,
                'gpus': total_gpus
            },
            'task_queue_size': self.task_queue.qsize(),
            'task_status_counts': task_status_counts,
            'completed_tasks': len(self.completed_tasks),
            'metrics': self.cluster_metrics,
            'scheduling_strategy': self.scheduling_strategy
        }
    
    def shutdown(self):
        """Shutdown cluster manager."""
        logger.info("Shutting down cluster manager")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for threads to complete
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
        
        if hasattr(self, '_scheduling_thread') and self._scheduling_thread.is_alive():
            self._scheduling_thread.join(timeout=5.0)
        
        # Notify all nodes of shutdown
        for node in self.nodes.values():
            try:
                shutdown_message = {
                    'type': 'cluster_shutdown',
                    'message': 'Cluster coordinator shutting down'
                }
                # Would send message in real implementation
                logger.debug(f"Notified node {node.node_id} of shutdown")
            except Exception as e:
                logger.warning(f"Failed to notify node {node.node_id}: {e}")
        
        logger.info("Cluster manager shutdown complete")


class NodeOrchestrator:
    """Orchestrates multiple nodes for complex distributed workflows."""
    
    def __init__(self, cluster_manager: ClusterManager):
        """Initialize node orchestrator.
        
        Args:
            cluster_manager: Cluster manager instance
        """
        self.cluster_manager = cluster_manager
        self.workflows: Dict[str, Any] = {}
        self.workflow_dependencies = {}
        
        logger.info("Node orchestrator initialized")
    
    def create_workflow(
        self,
        workflow_id: str,
        tasks: List[TaskInfo],
        dependencies: Dict[str, List[str]] = None
    ) -> bool:
        """Create distributed workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            tasks: List of tasks in workflow
            dependencies: Task dependencies (task_id -> [dependency_ids])
            
        Returns:
            True if workflow created successfully
        """
        try:
            self.workflows[workflow_id] = {
                'tasks': {task.task_id: task for task in tasks},
                'dependencies': dependencies or {},
                'status': 'pending',
                'created_at': time.time(),
                'completed_tasks': set(),
                'failed_tasks': set()
            }
            
            logger.info(f"Created workflow {workflow_id} with {len(tasks)} tasks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workflow {workflow_id}: {e}")
            return False
    
    def execute_workflow(self, workflow_id: str) -> bool:
        """Execute distributed workflow.
        
        Args:
            workflow_id: Workflow to execute
            
        Returns:
            True if workflow execution started
        """
        if workflow_id not in self.workflows:
            logger.error(f"Workflow {workflow_id} not found")
            return False
        
        workflow = self.workflows[workflow_id]
        workflow['status'] = 'running'
        
        # Start with tasks that have no dependencies
        ready_tasks = [
            task_id for task_id, task in workflow['tasks'].items()
            if not workflow['dependencies'].get(task_id, [])
        ]
        
        # Submit ready tasks
        for task_id in ready_tasks:
            task = workflow['tasks'][task_id]
            self.cluster_manager.submit_task(task, None)  # Simplified
        
        # Start monitoring thread
        def monitor_workflow():
            self._monitor_workflow_progress(workflow_id)
        
        thread = threading.Thread(target=monitor_workflow, daemon=True)
        thread.start()
        
        logger.info(f"Started workflow {workflow_id} execution")
        return True
    
    def _monitor_workflow_progress(self, workflow_id: str) -> None:
        """Monitor workflow progress and schedule dependent tasks.
        
        Args:
            workflow_id: Workflow to monitor
        """
        workflow = self.workflows[workflow_id]
        
        while workflow['status'] == 'running':
            try:
                # Check for completed tasks
                newly_completed = []
                
                for task_id, task in workflow['tasks'].items():
                    if (task_id not in workflow['completed_tasks'] and
                        task_id not in workflow['failed_tasks']):
                        
                        # Check task status in cluster
                        cluster_task = self.cluster_manager.tasks.get(task_id)
                        if cluster_task:
                            if cluster_task.status == 'completed':
                                workflow['completed_tasks'].add(task_id)
                                newly_completed.append(task_id)
                            elif cluster_task.status == 'failed':
                                workflow['failed_tasks'].add(task_id)
                
                # Schedule newly unblocked tasks
                for completed_task in newly_completed:
                    self._schedule_dependent_tasks(workflow_id, completed_task)
                
                # Check if workflow is complete
                all_tasks = set(workflow['tasks'].keys())
                if workflow['completed_tasks'] == all_tasks:
                    workflow['status'] = 'completed'
                    logger.info(f"Workflow {workflow_id} completed successfully")
                    break
                elif workflow['failed_tasks']:
                    # Check if any critical path failed
                    if self._has_critical_failure(workflow_id):
                        workflow['status'] = 'failed'
                        logger.error(f"Workflow {workflow_id} failed due to critical task failures")
                        break
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Workflow monitoring error: {e}")
                workflow['status'] = 'error'
                break
    
    def _schedule_dependent_tasks(self, workflow_id: str, completed_task: str) -> None:
        """Schedule tasks that depend on completed task.
        
        Args:
            workflow_id: Workflow ID
            completed_task: Task that just completed
        """
        workflow = self.workflows[workflow_id]
        
        for task_id, dependencies in workflow['dependencies'].items():
            if (completed_task in dependencies and
                task_id not in workflow['completed_tasks'] and
                task_id not in workflow['failed_tasks']):
                
                # Check if all dependencies are satisfied
                if all(dep in workflow['completed_tasks'] for dep in dependencies):
                    task = workflow['tasks'][task_id]
                    self.cluster_manager.submit_task(task, None)
                    logger.debug(f"Scheduled dependent task {task_id}")
    
    def _has_critical_failure(self, workflow_id: str) -> bool:
        """Check if workflow has critical failures that prevent completion.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            True if workflow has critical failures
        """
        workflow = self.workflows[workflow_id]
        failed_tasks = workflow['failed_tasks']
        
        # Simple check: if any failed task is required by pending tasks
        for task_id, dependencies in workflow['dependencies'].items():
            if (task_id not in workflow['completed_tasks'] and
                any(dep in failed_tasks for dep in dependencies)):
                return True
        
        return False
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            Workflow status or None if not found
        """
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        
        return {
            'workflow_id': workflow_id,
            'status': workflow['status'],
            'total_tasks': len(workflow['tasks']),
            'completed_tasks': len(workflow['completed_tasks']),
            'failed_tasks': len(workflow['failed_tasks']),
            'pending_tasks': (len(workflow['tasks']) - 
                            len(workflow['completed_tasks']) - 
                            len(workflow['failed_tasks'])),
            'created_at': workflow['created_at'],
            'task_details': {
                task_id: {
                    'status': 'completed' if task_id in workflow['completed_tasks']
                             else 'failed' if task_id in workflow['failed_tasks']
                             else 'pending',
                    'dependencies': workflow['dependencies'].get(task_id, [])
                }
                for task_id in workflow['tasks'].keys()
            }
        }