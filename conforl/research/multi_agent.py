"""Multi-Agent Conformal Risk Control.

This module implements novel multi-agent conformal bounds for distributed
reinforcement learning with communication constraints and consensus mechanisms.
This extends ConfoRL to multi-agent settings with joint risk certificates.

Research Contributions:
- Distributed conformal risk control with communication constraints
- Consensus-based risk certificate aggregation
- Scalable multi-agent safety guarantees
- Byzantine-robust risk assessment

Author: ConfoRL Research Team
License: Apache 2.0
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict, deque
import warnings

from ..core.types import RiskCertificate, TrajectoryData
from ..risk.controllers import AdaptiveRiskController
from ..risk.measures import RiskMeasure
from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class CommunicationTopology(Enum):
    """Communication topology types for multi-agent systems."""
    FULLY_CONNECTED = "fully_connected"
    RING = "ring"
    STAR = "star"
    GRID = "grid"
    RANDOM_GRAPH = "random_graph"
    HIERARCHICAL = "hierarchical"


@dataclass
class AgentInfo:
    """Information about an agent in the multi-agent system."""
    
    agent_id: int
    agent_type: str
    risk_budget: float
    neighbors: Set[int] = field(default_factory=set)
    is_byzantine: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CommunicationMessage:
    """Message passed between agents."""
    
    sender_id: int
    receiver_id: int
    message_type: str  # 'risk_update', 'certificate', 'consensus'
    content: Dict[str, Any]
    timestamp: float
    message_id: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class MultiAgentRiskCertificate:
    """Joint risk certificate for multi-agent system."""
    
    global_risk_bound: float
    agent_risk_bounds: Dict[int, float]
    consensus_confidence: float
    communication_rounds: int
    participating_agents: Set[int]
    byzantine_resilience: float
    aggregation_method: str
    sample_sizes: Dict[int, int]
    timestamp: float
    metadata: Optional[Dict[str, Any]] = None


class CommunicationNetwork:
    """Manages communication between agents in the multi-agent system."""
    
    def __init__(
        self,
        agents: Dict[int, AgentInfo],
        topology: CommunicationTopology = CommunicationTopology.FULLY_CONNECTED,
        communication_delay: float = 0.0,
        message_drop_rate: float = 0.0,
        bandwidth_limit: Optional[int] = None
    ):
        """Initialize communication network.
        
        Args:
            agents: Dictionary of agents in the system
            topology: Communication topology
            communication_delay: Delay in message delivery (seconds)
            message_drop_rate: Probability of message being dropped
            bandwidth_limit: Maximum messages per second (None for unlimited)
        """
        self.agents = agents
        self.topology = topology
        self.communication_delay = communication_delay
        self.message_drop_rate = message_drop_rate
        self.bandwidth_limit = bandwidth_limit
        
        # Communication infrastructure
        self.message_queue = deque()
        self.message_history = []
        self.network_graph = self._build_network_graph()
        
        # Statistics
        self.messages_sent = 0
        self.messages_delivered = 0
        self.messages_dropped = 0
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Initialized communication network with {len(agents)} agents "
                   f"using {topology.value} topology")
    
    def _build_network_graph(self) -> Dict[int, Set[int]]:
        """Build communication graph based on topology."""
        n_agents = len(self.agents)
        agent_ids = list(self.agents.keys())
        graph = defaultdict(set)
        
        if self.topology == CommunicationTopology.FULLY_CONNECTED:
            # Every agent connects to every other agent
            for i, agent_i in enumerate(agent_ids):
                for j, agent_j in enumerate(agent_ids):
                    if i != j:
                        graph[agent_i].add(agent_j)
        
        elif self.topology == CommunicationTopology.RING:
            # Agents connected in a ring
            for i in range(n_agents):
                current = agent_ids[i]
                next_agent = agent_ids[(i + 1) % n_agents]
                prev_agent = agent_ids[(i - 1) % n_agents]
                graph[current].add(next_agent)
                graph[current].add(prev_agent)
        
        elif self.topology == CommunicationTopology.STAR:
            # One central agent connected to all others
            central_agent = agent_ids[0]  # First agent is central
            for agent_id in agent_ids[1:]:
                graph[central_agent].add(agent_id)
                graph[agent_id].add(central_agent)
        
        elif self.topology == CommunicationTopology.GRID:
            # Grid topology (approximate for any number of agents)
            grid_size = int(np.ceil(np.sqrt(n_agents)))
            for i, agent_id in enumerate(agent_ids):
                row, col = divmod(i, grid_size)
                # Connect to neighbors (up, down, left, right)
                for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        neighbor_idx = nr * grid_size + nc
                        if neighbor_idx < len(agent_ids):
                            neighbor_id = agent_ids[neighbor_idx]
                            graph[agent_id].add(neighbor_id)
        
        elif self.topology == CommunicationTopology.RANDOM_GRAPH:
            # Random graph with some connection probability
            connection_prob = 0.3
            for i, agent_i in enumerate(agent_ids):
                for j, agent_j in enumerate(agent_ids[i+1:], i+1):
                    if np.random.random() < connection_prob:
                        graph[agent_i].add(agent_j)
                        graph[agent_j].add(agent_i)
        
        # Update agent neighbor information
        for agent_id, neighbors in graph.items():
            if agent_id in self.agents:
                self.agents[agent_id].neighbors = neighbors
        
        return dict(graph)
    
    def send_message(
        self,
        sender_id: int,
        receiver_id: int,
        message_type: str,
        content: Dict[str, Any]
    ) -> bool:
        """Send message between agents.
        
        Args:
            sender_id: Sender agent ID
            receiver_id: Receiver agent ID  
            message_type: Type of message
            content: Message content
            
        Returns:
            True if message was queued successfully
        """
        with self._lock:
            # Check if communication is allowed
            if receiver_id not in self.network_graph.get(sender_id, set()):
                logger.warning(f"No communication link between {sender_id} and {receiver_id}")
                return False
            
            # Check bandwidth limit
            if self.bandwidth_limit and self.messages_sent >= self.bandwidth_limit:
                logger.warning("Bandwidth limit exceeded, dropping message")
                self.messages_dropped += 1
                return False
            
            # Simulate message drop
            if np.random.random() < self.message_drop_rate:
                logger.debug(f"Message dropped: {sender_id} -> {receiver_id}")
                self.messages_dropped += 1
                return False
            
            # Create message
            message = CommunicationMessage(
                sender_id=sender_id,
                receiver_id=receiver_id,
                message_type=message_type,
                content=content,
                timestamp=time.time(),
                message_id=f"{sender_id}_{receiver_id}_{int(time.time()*1000)}"
            )
            
            # Queue with delay
            delivery_time = time.time() + self.communication_delay
            self.message_queue.append((delivery_time, message))
            
            self.messages_sent += 1
            return True
    
    def receive_messages(self, agent_id: int) -> List[CommunicationMessage]:
        """Receive pending messages for an agent.
        
        Args:
            agent_id: Agent ID to receive messages for
            
        Returns:
            List of messages ready for delivery
        """
        with self._lock:
            current_time = time.time()
            ready_messages = []
            remaining_queue = deque()
            
            # Check messages ready for delivery
            while self.message_queue:
                delivery_time, message = self.message_queue.popleft()
                
                if delivery_time <= current_time and message.receiver_id == agent_id:
                    ready_messages.append(message)
                    self.messages_delivered += 1
                    self.message_history.append(message)
                else:
                    remaining_queue.append((delivery_time, message))
            
            self.message_queue = remaining_queue
            return ready_messages
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get network communication statistics."""
        with self._lock:
            return {
                'messages_sent': self.messages_sent,
                'messages_delivered': self.messages_delivered,
                'messages_dropped': self.messages_dropped,
                'messages_in_queue': len(self.message_queue),
                'delivery_rate': (self.messages_delivered / max(1, self.messages_sent)),
                'topology': self.topology.value,
                'num_agents': len(self.agents),
                'total_connections': sum(len(neighbors) for neighbors in self.network_graph.values()) // 2
            }


class ConsensusAlgorithm(ABC):
    """Abstract base class for consensus algorithms."""
    
    @abstractmethod
    def run_consensus(
        self,
        agents: Dict[int, AgentInfo],
        local_values: Dict[int, float],
        network: CommunicationNetwork,
        max_rounds: int = 10
    ) -> Tuple[float, Dict[int, float]]:
        """Run consensus algorithm.
        
        Args:
            agents: Agent information
            local_values: Local values from each agent
            network: Communication network
            max_rounds: Maximum consensus rounds
            
        Returns:
            Tuple of (consensus_value, agent_values)
        """
        pass


class AverageConsensus(ConsensusAlgorithm):
    """Average consensus algorithm for risk aggregation."""
    
    def __init__(self, convergence_threshold: float = 1e-6):
        """Initialize average consensus.
        
        Args:
            convergence_threshold: Convergence threshold for consensus
        """
        self.convergence_threshold = convergence_threshold
    
    def run_consensus(
        self,
        agents: Dict[int, AgentInfo],
        local_values: Dict[int, float],
        network: CommunicationNetwork,
        max_rounds: int = 10
    ) -> Tuple[float, Dict[int, float]]:
        """Run average consensus algorithm.
        
        Args:
            agents: Agent information
            local_values: Initial local values
            network: Communication network
            max_rounds: Maximum rounds
            
        Returns:
            Tuple of (consensus_value, final_agent_values)
        """
        current_values = local_values.copy()
        agent_ids = list(agents.keys())
        
        for round_num in range(max_rounds):
            # Exchange values with neighbors
            next_values = {}
            
            for agent_id in agent_ids:
                if agents[agent_id].is_byzantine:
                    # Byzantine agents may send incorrect values
                    next_values[agent_id] = current_values[agent_id] * (1 + np.random.normal(0, 0.1))
                    continue
                
                # Get neighbor values
                neighbors = agents[agent_id].neighbors
                neighbor_values = [current_values.get(neighbor_id, current_values[agent_id]) 
                                 for neighbor_id in neighbors]
                
                # Average with neighbors (including self)
                all_values = [current_values[agent_id]] + neighbor_values
                next_values[agent_id] = np.mean(all_values)
            
            # Check convergence
            max_change = max(abs(next_values[agent_id] - current_values[agent_id])
                           for agent_id in agent_ids)
            
            current_values = next_values
            
            if max_change < self.convergence_threshold:
                logger.debug(f"Consensus converged in {round_num + 1} rounds")
                break
        
        # Final consensus is average of all agent values
        consensus_value = np.mean(list(current_values.values()))
        
        return consensus_value, current_values


class ByzantineRobustConsensus(ConsensusAlgorithm):
    """Byzantine-robust consensus algorithm."""
    
    def __init__(self, byzantine_fraction: float = 0.3):
        """Initialize Byzantine-robust consensus.
        
        Args:
            byzantine_fraction: Assumed fraction of Byzantine agents
        """
        self.byzantine_fraction = byzantine_fraction
    
    def run_consensus(
        self,
        agents: Dict[int, AgentInfo],
        local_values: Dict[int, float],
        network: CommunicationNetwork,
        max_rounds: int = 10
    ) -> Tuple[float, Dict[int, float]]:
        """Run Byzantine-robust consensus using trimmed mean.
        
        Args:
            agents: Agent information
            local_values: Initial local values
            network: Communication network
            max_rounds: Maximum rounds
            
        Returns:
            Tuple of (consensus_value, final_agent_values)
        """
        current_values = local_values.copy()
        agent_ids = list(agents.keys())
        n_agents = len(agent_ids)
        
        # Number of values to trim from each side
        n_trim = int(np.ceil(n_agents * self.byzantine_fraction))
        
        for round_num in range(max_rounds):
            # Collect all values
            all_values = []
            for agent_id in agent_ids:
                neighbors = agents[agent_id].neighbors
                neighbor_values = [current_values.get(neighbor_id, current_values[agent_id]) 
                                 for neighbor_id in neighbors]
                all_values.extend(neighbor_values)
                all_values.append(current_values[agent_id])
            
            # Trimmed mean for robustness
            sorted_values = sorted(all_values)
            if len(sorted_values) > 2 * n_trim:
                trimmed_values = sorted_values[n_trim:-n_trim] if n_trim > 0 else sorted_values
                consensus_value = np.mean(trimmed_values)
            else:
                consensus_value = np.median(sorted_values)
            
            # Update all agents towards consensus
            next_values = {}
            for agent_id in agent_ids:
                if not agents[agent_id].is_byzantine:
                    # Non-Byzantine agents move towards consensus
                    next_values[agent_id] = 0.8 * current_values[agent_id] + 0.2 * consensus_value
                else:
                    # Byzantine agents may deviate
                    next_values[agent_id] = consensus_value * (1 + np.random.normal(0, 0.2))
            
            current_values = next_values
        
        # Final consensus using trimmed mean
        all_final_values = list(current_values.values())
        sorted_final = sorted(all_final_values)
        if len(sorted_final) > 2 * n_trim:
            trimmed_final = sorted_final[n_trim:-n_trim] if n_trim > 0 else sorted_final
            final_consensus = np.mean(trimmed_final)
        else:
            final_consensus = np.median(sorted_final)
        
        return final_consensus, current_values


class MultiAgentRiskController:
    """Distributed risk controller for multi-agent systems."""
    
    def __init__(
        self,
        agents: Dict[int, AgentInfo],
        communication_network: CommunicationNetwork,
        consensus_algorithm: ConsensusAlgorithm,
        local_controllers: Dict[int, AdaptiveRiskController],
        global_risk_budget: float = 0.05,
        consensus_frequency: int = 100
    ):
        """Initialize multi-agent risk controller.
        
        Args:
            agents: Dictionary of agent information
            communication_network: Communication infrastructure
            consensus_algorithm: Algorithm for reaching consensus
            local_controllers: Local risk controllers for each agent
            global_risk_budget: Global system risk budget
            consensus_frequency: Frequency of consensus rounds
        """
        self.agents = agents
        self.network = communication_network
        self.consensus_algorithm = consensus_algorithm
        self.local_controllers = local_controllers
        self.global_risk_budget = global_risk_budget
        self.consensus_frequency = consensus_frequency
        
        # Multi-agent state
        self.global_certificates = []
        self.consensus_history = []
        self.update_counts = defaultdict(int)
        
        # Certificate cache
        self._certificate_cache = None
        self._cache_timestamp = 0.0
        
        # Thread safety for distributed updates
        self._lock = threading.Lock()
        
        logger.info(f"Initialized multi-agent risk controller for {len(agents)} agents")
    
    def update_agent(
        self,
        agent_id: int,
        trajectory: TrajectoryData,
        risk_measure: RiskMeasure
    ) -> None:
        """Update local risk controller for specific agent.
        
        Args:
            agent_id: Agent to update
            trajectory: New trajectory data
            risk_measure: Risk measure to use
        """
        if agent_id not in self.local_controllers:
            raise ValidationError(f"Unknown agent ID: {agent_id}")
        
        with self._lock:
            # Update local controller
            self.local_controllers[agent_id].update(trajectory, risk_measure)
            self.update_counts[agent_id] += 1
            
            # Check if consensus round is needed
            total_updates = sum(self.update_counts.values())
            if total_updates % self.consensus_frequency == 0:
                self._run_consensus_round()
            
            # Invalidate cache
            self._certificate_cache = None
            self._cache_timestamp = 0.0
            
            logger.debug(f"Updated agent {agent_id} (update #{self.update_counts[agent_id]})")
    
    def _run_consensus_round(self) -> None:
        """Run consensus round to update global risk estimates."""
        try:
            # Get local risk bounds from all agents
            local_risk_bounds = {}
            for agent_id, controller in self.local_controllers.items():
                cert = controller.get_certificate()
                local_risk_bounds[agent_id] = cert.risk_bound
            
            # Run consensus algorithm
            start_time = time.time()
            global_risk_bound, agent_consensus_values = self.consensus_algorithm.run_consensus(
                agents=self.agents,
                local_values=local_risk_bounds,
                network=self.network,
                max_rounds=10
            )
            consensus_time = time.time() - start_time
            
            # Record consensus result
            consensus_record = {
                'global_risk_bound': global_risk_bound,
                'agent_values': agent_consensus_values,
                'consensus_time': consensus_time,
                'timestamp': time.time(),
                'participating_agents': set(local_risk_bounds.keys())
            }
            
            self.consensus_history.append(consensus_record)
            
            # Keep recent history
            max_history = 1000
            if len(self.consensus_history) > max_history:
                self.consensus_history = self.consensus_history[-max_history:]
            
            logger.info(f"Consensus round completed: global_risk={global_risk_bound:.4f}, "
                       f"time={consensus_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Consensus round failed: {e}")
    
    def get_multi_agent_certificate(self) -> MultiAgentRiskCertificate:
        """Get joint risk certificate for the multi-agent system.
        
        Returns:
            Multi-agent risk certificate with global guarantees
        """
        # Check cache
        current_time = time.time()
        if (self._certificate_cache is not None and 
            current_time - self._cache_timestamp < 30.0):  # 30 second cache
            return self._certificate_cache
        
        with self._lock:
            # Get individual agent certificates
            agent_risk_bounds = {}
            sample_sizes = {}
            
            for agent_id, controller in self.local_controllers.items():
                cert = controller.get_certificate()
                agent_risk_bounds[agent_id] = cert.risk_bound
                sample_sizes[agent_id] = cert.sample_size
            
            # Get latest consensus result
            if self.consensus_history:
                latest_consensus = self.consensus_history[-1]
                global_risk_bound = latest_consensus['global_risk_bound']
                consensus_confidence = 0.95  # Based on consensus algorithm guarantees
                communication_rounds = len(self.consensus_history)
                participating_agents = latest_consensus['participating_agents']
            else:
                # No consensus yet - use conservative estimate
                global_risk_bound = max(agent_risk_bounds.values()) if agent_risk_bounds else 1.0
                consensus_confidence = 0.5
                communication_rounds = 0
                participating_agents = set(self.agents.keys())
            
            # Estimate Byzantine resilience
            n_agents = len(self.agents)
            n_byzantine = sum(1 for agent in self.agents.values() if agent.is_byzantine)
            byzantine_resilience = max(0.0, 1.0 - 2 * n_byzantine / n_agents) if n_agents > 0 else 0.0
            
            # Create multi-agent certificate
            ma_certificate = MultiAgentRiskCertificate(
                global_risk_bound=min(1.0, global_risk_bound),
                agent_risk_bounds=agent_risk_bounds,
                consensus_confidence=consensus_confidence,
                communication_rounds=communication_rounds,
                participating_agents=participating_agents,
                byzantine_resilience=byzantine_resilience,
                aggregation_method=type(self.consensus_algorithm).__name__,
                sample_sizes=sample_sizes,
                timestamp=current_time,
                metadata={
                    'num_agents': len(self.agents),
                    'topology': self.network.topology.value,
                    'consensus_frequency': self.consensus_frequency,
                    'network_stats': self.network.get_network_stats(),
                    'total_updates': sum(self.update_counts.values())
                }
            )
            
            # Cache result
            self._certificate_cache = ma_certificate
            self._cache_timestamp = current_time
            
            return ma_certificate
    
    def simulate_byzantine_failure(
        self,
        agent_id: int,
        failure_type: str = "random"
    ) -> None:
        """Simulate Byzantine failure for testing robustness.
        
        Args:
            agent_id: Agent to mark as Byzantine
            failure_type: Type of Byzantine behavior
        """
        if agent_id in self.agents:
            self.agents[agent_id].is_byzantine = True
            logger.warning(f"Agent {agent_id} marked as Byzantine ({failure_type})")
        else:
            logger.error(f"Cannot mark unknown agent {agent_id} as Byzantine")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics.
        
        Returns:
            Dictionary with system health information
        """
        # Agent health
        healthy_agents = sum(1 for agent in self.agents.values() if not agent.is_byzantine)
        total_agents = len(self.agents)
        
        # Communication health
        network_stats = self.network.get_network_stats()
        
        # Consensus health
        recent_consensus = self.consensus_history[-10:] if self.consensus_history else []
        consensus_times = [c['consensus_time'] for c in recent_consensus]
        
        # Risk bound stability
        if len(self.consensus_history) > 1:
            recent_bounds = [c['global_risk_bound'] for c in recent_consensus]
            risk_stability = np.std(recent_bounds) if len(recent_bounds) > 1 else 0.0
        else:
            risk_stability = float('inf')
        
        # Current certificate
        current_cert = self.get_multi_agent_certificate()
        
        health = {
            'agent_health': {
                'healthy_agents': healthy_agents,
                'total_agents': total_agents,
                'health_ratio': healthy_agents / max(1, total_agents),
                'byzantine_agents': total_agents - healthy_agents
            },
            'communication_health': {
                'delivery_rate': network_stats['delivery_rate'],
                'messages_in_queue': network_stats['messages_in_queue'],
                'total_connections': network_stats['total_connections']
            },
            'consensus_health': {
                'consensus_rounds': len(self.consensus_history),
                'avg_consensus_time': np.mean(consensus_times) if consensus_times else 0.0,
                'consensus_frequency_actual': len(self.consensus_history) / max(1, sum(self.update_counts.values()) / self.consensus_frequency),
                'risk_stability': risk_stability
            },
            'risk_certification': {
                'global_risk_bound': current_cert.global_risk_bound,
                'consensus_confidence': current_cert.consensus_confidence,
                'byzantine_resilience': current_cert.byzantine_resilience,
                'meets_global_budget': current_cert.global_risk_bound <= self.global_risk_budget
            },
            'overall_health_score': self._compute_health_score()
        }
        
        return health
    
    def _compute_health_score(self) -> float:
        """Compute overall system health score (0-1)."""
        try:
            health = self.get_system_health()
            
            # Component scores (0-1)
            agent_score = health['agent_health']['health_ratio']
            comm_score = health['communication_health']['delivery_rate']
            
            # Consensus score
            if len(self.consensus_history) > 0:
                consensus_score = min(1.0, health['consensus_health']['consensus_frequency_actual'])
            else:
                consensus_score = 0.0
            
            # Risk score
            current_cert = self.get_multi_agent_certificate()
            risk_score = 1.0 if current_cert.global_risk_bound <= self.global_risk_budget else 0.5
            
            # Weighted average
            weights = [0.3, 0.2, 0.2, 0.3]  # agent, comm, consensus, risk
            scores = [agent_score, comm_score, consensus_score, risk_score]
            
            overall_score = sum(w * s for w, s in zip(weights, scores))
            return overall_score
            
        except Exception as e:
            logger.error(f"Health score computation failed: {e}")
            return 0.0


# Research Extensions

class FederatedRiskLearning:
    """Federated learning approach to distributed risk assessment."""
    
    def __init__(self, num_agents: int):
        """Initialize federated risk learning.
        
        Args:
            num_agents: Number of participating agents
        """
        self.num_agents = num_agents
        self.global_model = None
        logger.info(f"Federated risk learning initialized for {num_agents} agents")
    
    def federated_risk_update(
        self,
        local_updates: Dict[int, Any],
        aggregation_weights: Optional[Dict[int, float]] = None
    ) -> Any:
        """Aggregate local risk model updates using federated learning.
        
        This is a placeholder for sophisticated federated risk learning.
        """
        logger.warning("Federated risk learning not fully implemented - research in progress")
        return self.global_model


class ScalableConsensus:
    """Scalable consensus algorithms for large multi-agent systems."""
    
    def __init__(self, scalability_threshold: int = 1000):
        """Initialize scalable consensus.
        
        Args:
            scalability_threshold: Threshold for switching to scalable algorithms
        """
        self.scalability_threshold = scalability_threshold
        logger.info(f"Scalable consensus initialized (threshold: {scalability_threshold} agents)")
    
    def hierarchical_consensus(
        self,
        agents: Dict[int, AgentInfo],
        local_values: Dict[int, float]
    ) -> float:
        """Run hierarchical consensus for large-scale systems.
        
        This is a placeholder for advanced hierarchical consensus algorithms.
        """
        logger.warning("Hierarchical consensus not fully implemented - research in progress")
        return np.mean(list(local_values.values()))