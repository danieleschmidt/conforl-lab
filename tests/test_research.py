"""Tests for ConfoRL research extensions.

Comprehensive test suite for cutting-edge research implementations including
causal conformal RL, adversarial robustness, and multi-agent risk control.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch

import conforl
from conforl.research import (
    # Compositional
    HierarchicalPolicy, CompositionalRiskController, HierarchicalPolicyBuilder,
    
    # Causal
    CausalGraph, CausalIntervention, CausalRiskController, CausalShiftDetector,
    CausalConformPredictor, CounterfactualRiskAssessment, CausalGraphLearner,
    
    # Adversarial  
    AttackType, AdversarialAttackGenerator, CertifiedDefense, AdversarialRiskController,
    AdaptiveAttackGeneration, MultiStepAdversarialRisk,
    
    # Multi-Agent
    CommunicationTopology, AgentInfo, CommunicationNetwork, AverageConsensus,
    ByzantineRobustConsensus, MultiAgentRiskController, FederatedRiskLearning,
    ScalableConsensus
)
from conforl.core.types import TrajectoryData, RiskCertificate
from conforl.risk.controllers import AdaptiveRiskController
from conforl.risk.measures import SafetyViolationRisk


class TestCausalConformalRL:
    """Test causal conformal risk control functionality."""
    
    def test_causal_graph_creation(self):
        """Test causal graph creation and validation."""
        nodes = ['state', 'action', 'reward', 'next_state']
        edges = {
            'state': ['action'],
            'action': ['reward', 'next_state'],
            'reward': [],
            'next_state': []
        }
        
        graph = CausalGraph(nodes=nodes, edges=edges)
        
        assert graph.nodes == nodes
        assert graph.edges == edges
        assert len(graph.node_types) == 0  # Default empty
    
    def test_causal_graph_validation(self):
        """Test causal graph validation for cycles."""
        nodes = ['a', 'b']
        edges = {'a': ['a']}  # Self-loop
        
        with pytest.raises(Exception):  # Should detect cycle
            CausalGraph(nodes=nodes, edges=edges)
    
    def test_causal_intervention(self):
        """Test causal intervention representation."""
        intervention = CausalIntervention(
            target_node='action',
            intervention_value=0.5,
            intervention_type='do',
            strength=1.0
        )
        
        assert intervention.target_node == 'action'
        assert intervention.intervention_value == 0.5
        assert intervention.intervention_type == 'do'
        assert intervention.strength == 1.0
    
    def test_causal_shift_detector(self):
        """Test causal shift detection."""
        nodes = ['x', 'y', 'z']
        edges = {'x': ['y'], 'y': ['z'], 'z': []}
        graph = CausalGraph(nodes=nodes, edges=edges)
        
        detector = CausalShiftDetector(graph, detection_threshold=0.05)
        
        # Update baseline with normal data
        for _ in range(100):
            observations = {
                'x': np.random.normal(0, 1),
                'y': np.random.normal(0, 1), 
                'z': np.random.normal(0, 1)
            }
            detector.update_baseline(observations)
        
        # Test shift detection with shifted data
        shifted_observations = {
            'x': 5.0,  # Large shift
            'y': 0.1,
            'z': -0.1
        }
        
        shifts = detector.detect_shift(shifted_observations)
        
        # Should detect shift in x
        assert 'x' in shifts
        assert shifts['x'] < 0.05  # Significant shift detected
    
    def test_causal_conform_predictor(self):
        """Test causal conformal predictor."""
        nodes = ['state', 'action']
        edges = {'state': ['action'], 'action': []}
        graph = CausalGraph(nodes=nodes, edges=edges)
        
        predictor = CausalConformPredictor(graph)
        
        # Update with baseline scores
        baseline_scores = np.random.beta(2, 8, 100).tolist()  # Low risk scores
        predictor.update_baseline_scores(baseline_scores)
        
        # Update with intervention scores
        intervention = CausalIntervention(
            target_node='state',
            intervention_value=1.0,
            intervention_type='do'
        )
        intervention_scores = np.random.beta(1, 4, 50).tolist()  # Higher risk
        predictor.update_intervention_scores(intervention, intervention_scores)
        
        # Get causal quantile
        quantile = predictor.get_causal_quantile(confidence=0.95, max_intervention_strength=1.0)
        
        assert 0.0 <= quantile <= 1.0
        assert quantile > np.quantile(baseline_scores, 0.95)  # Should be higher than baseline
    
    def test_causal_risk_controller(self):
        """Test causal risk controller."""
        # Create simple causal graph
        nodes = ['state', 'action', 'reward']
        edges = {'state': ['action'], 'action': ['reward'], 'reward': []}
        graph = CausalGraph(nodes=nodes, edges=edges)
        
        # Base controller
        base_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        
        # Causal controller
        causal_controller = CausalRiskController(
            causal_graph=graph,
            base_controller=base_controller
        )
        
        # Create mock trajectory
        trajectory = Mock(spec=TrajectoryData)
        trajectory.states = [np.array([1, 2, 3]) for _ in range(10)]
        trajectory.actions = [np.array([0.5]) for _ in range(10)]
        trajectory.rewards = [0.1 * i for i in range(10)]
        
        risk_measure = SafetyViolationRisk(safety_threshold=0.5)
        
        # Update controller
        causal_controller.update(trajectory, risk_measure)
        
        # Get certificate
        cert = causal_controller.get_causal_certificate()
        
        assert hasattr(cert, 'baseline_risk_bound')
        assert hasattr(cert, 'intervention_robust_bound')
        assert 0.0 <= cert.baseline_risk_bound <= 1.0
        assert 0.0 <= cert.intervention_robust_bound <= 1.0
    
    def test_counterfactual_risk_assessment(self):
        """Test counterfactual risk assessment (placeholder)."""
        nodes = ['state', 'action']
        edges = {'state': ['action'], 'action': []}
        graph = CausalGraph(nodes=nodes, edges=edges)
        
        counterfactual = CounterfactualRiskAssessment(graph)
        
        trajectory = Mock(spec=TrajectoryData)
        intervention = CausalIntervention(
            target_node='state',
            intervention_value=1.0,
            intervention_type='do'
        )
        
        # This should return a placeholder value
        risk = counterfactual.compute_counterfactual_risk(trajectory, intervention)
        assert isinstance(risk, float)
        assert 0.0 <= risk <= 1.0


class TestAdversarialConformalRL:
    """Test adversarial robust conformal risk control."""
    
    def test_attack_generation(self):
        """Test adversarial attack generation."""
        generator = AdversarialAttackGenerator(attack_budget=0.1)
        
        # Mock trajectory
        trajectory = Mock(spec=TrajectoryData)
        trajectory.states = [np.array([1.0, 2.0, 3.0]) for _ in range(5)]
        
        # Generate L-inf attack
        attack = generator.generate_attack(trajectory, AttackType.L_INF, epsilon=0.1)
        
        assert attack.attack_type == AttackType.L_INF
        assert attack.epsilon == 0.1
        assert attack.target_component == 'state'
        assert attack.perturbation_vector is not None
        assert np.all(np.abs(attack.perturbation_vector) <= 0.1)
        
        # Generate L-2 attack
        l2_attack = generator.generate_attack(trajectory, AttackType.L_2, epsilon=0.2)
        
        assert l2_attack.attack_type == AttackType.L_2
        assert l2_attack.epsilon == 0.2
        assert np.linalg.norm(l2_attack.perturbation_vector) <= 0.2 + 1e-6  # Small tolerance
    
    def test_certified_defense(self):
        """Test certified defense mechanism."""
        defense = CertifiedDefense(
            defense_type="randomized_smoothing",
            noise_scale=0.1,
            num_samples=100
        )
        
        # Mock trajectory
        trajectory = Mock(spec=TrajectoryData)
        trajectory.states = [np.array([1.0, 2.0]) for _ in range(3)]
        
        # Apply defense
        defended = defense.defend(trajectory)
        assert defended is not None  # Should return something
        
        # Test certification (simplified)
        radius, prob = defense.certify_robustness(trajectory, AttackType.L_2, confidence=0.95)
        
        assert radius >= 0.0
        assert 0.0 <= prob <= 1.0
    
    def test_adversarial_risk_controller(self):
        """Test adversarial risk controller."""
        # Base components
        base_controller = AdaptiveRiskController(target_risk=0.05)
        attack_generator = AdversarialAttackGenerator(attack_budget=0.1)
        certified_defense = CertifiedDefense()
        
        # Adversarial controller
        adv_controller = AdversarialRiskController(
            base_controller=base_controller,
            attack_generator=attack_generator,
            certified_defense=certified_defense,
            robustness_testing_freq=5  # Test every 5 updates
        )
        
        # Mock trajectory and risk measure
        trajectory = Mock(spec=TrajectoryData)
        trajectory.states = [np.array([0.5, 1.0, -0.5]) for _ in range(10)]
        risk_measure = SafetyViolationRisk(safety_threshold=0.8)
        
        # Update multiple times to trigger robustness testing
        for i in range(6):
            adv_controller.update(trajectory, risk_measure, run_robustness_test=(i == 5))
        
        # Get adversarial certificate
        cert = adv_controller.get_adversarial_certificate()
        
        assert hasattr(cert, 'clean_risk_bound')
        assert hasattr(cert, 'adversarial_risk_bound') 
        assert hasattr(cert, 'certified_radius')
        assert 0.0 <= cert.clean_risk_bound <= 1.0
        assert 0.0 <= cert.adversarial_risk_bound <= 1.0
        assert cert.certified_radius >= 0.0
    
    def test_adaptive_attack_generation(self):
        """Test adaptive attack generation (placeholder)."""
        base_generator = AdversarialAttackGenerator()
        adaptive_gen = AdaptiveAttackGeneration(base_generator)
        
        trajectory = Mock(spec=TrajectoryData)
        defense_history = [{'defense_type': 'randomized_smoothing'}]
        
        # This should fallback to base generator
        attack = adaptive_gen.generate_adaptive_attack(trajectory, defense_history)
        assert attack is not None
        assert hasattr(attack, 'attack_type')


class TestMultiAgentConformalRL:
    """Test multi-agent conformal risk control."""
    
    def test_agent_info_creation(self):
        """Test agent info creation."""
        agent = AgentInfo(
            agent_id=1,
            agent_type="learner",
            risk_budget=0.05,
            neighbors={2, 3}
        )
        
        assert agent.agent_id == 1
        assert agent.agent_type == "learner"
        assert agent.risk_budget == 0.05
        assert agent.neighbors == {2, 3}
        assert not agent.is_byzantine
    
    def test_communication_network(self):
        """Test communication network."""
        agents = {
            1: AgentInfo(1, "learner", 0.05),
            2: AgentInfo(2, "learner", 0.05),
            3: AgentInfo(3, "coordinator", 0.01)
        }
        
        network = CommunicationNetwork(
            agents=agents,
            topology=CommunicationTopology.FULLY_CONNECTED,
            communication_delay=0.01,
            message_drop_rate=0.1
        )
        
        # Test message sending
        success = network.send_message(
            sender_id=1,
            receiver_id=2,
            message_type="risk_update",
            content={"risk_bound": 0.03}
        )
        
        # Should succeed in fully connected topology
        assert success
        
        # Test message receiving (after delay)
        time.sleep(0.02)  # Wait for message delivery
        messages = network.receive_messages(agent_id=2)
        
        # May or may not receive due to drop rate, but should be list
        assert isinstance(messages, list)
    
    def test_average_consensus(self):
        """Test average consensus algorithm."""
        consensus = AverageConsensus(convergence_threshold=1e-6)
        
        # Create simple agents
        agents = {
            1: AgentInfo(1, "agent", 0.05, neighbors={2, 3}),
            2: AgentInfo(2, "agent", 0.05, neighbors={1, 3}), 
            3: AgentInfo(3, "agent", 0.05, neighbors={1, 2})
        }
        
        network = CommunicationNetwork(agents, CommunicationTopology.FULLY_CONNECTED)
        
        # Initial values
        local_values = {1: 0.1, 2: 0.05, 3: 0.02}
        
        # Run consensus
        consensus_value, final_values = consensus.run_consensus(
            agents=agents,
            local_values=local_values,
            network=network,
            max_rounds=20
        )
        
        # Should converge to average
        expected_average = np.mean(list(local_values.values()))
        assert abs(consensus_value - expected_average) < 0.01
        
        # All agents should have similar final values
        final_vals = list(final_values.values())
        assert max(final_vals) - min(final_vals) < 0.01
    
    def test_byzantine_robust_consensus(self):
        """Test Byzantine-robust consensus."""
        consensus = ByzantineRobustConsensus(byzantine_fraction=0.3)
        
        # Create agents with some Byzantine
        agents = {
            1: AgentInfo(1, "honest", 0.05, neighbors={2, 3, 4}),
            2: AgentInfo(2, "honest", 0.05, neighbors={1, 3, 4}),
            3: AgentInfo(3, "byzantine", 0.05, neighbors={1, 2, 4}, is_byzantine=True),
            4: AgentInfo(4, "honest", 0.05, neighbors={1, 2, 3})
        }
        
        network = CommunicationNetwork(agents, CommunicationTopology.FULLY_CONNECTED)
        
        # Honest agents have similar values, Byzantine has outlier
        local_values = {1: 0.05, 2: 0.06, 3: 0.5, 4: 0.04}  # Agent 3 is outlier
        
        # Run consensus
        consensus_value, final_values = consensus.run_consensus(
            agents=agents,
            local_values=local_values,
            network=network,
            max_rounds=10
        )
        
        # Should be robust to Byzantine agent
        honest_average = np.mean([0.05, 0.06, 0.04])
        assert abs(consensus_value - honest_average) < 0.1  # Should be close to honest average
    
    def test_multi_agent_risk_controller(self):
        """Test multi-agent risk controller."""
        # Create agents
        agents = {
            1: AgentInfo(1, "learner", 0.05),
            2: AgentInfo(2, "learner", 0.06)
        }
        
        # Communication network
        network = CommunicationNetwork(agents, CommunicationTopology.FULLY_CONNECTED)
        
        # Consensus algorithm
        consensus = AverageConsensus()
        
        # Local controllers
        local_controllers = {
            1: AdaptiveRiskController(target_risk=0.05),
            2: AdaptiveRiskController(target_risk=0.06)
        }
        
        # Multi-agent controller
        ma_controller = MultiAgentRiskController(
            agents=agents,
            communication_network=network,
            consensus_algorithm=consensus,
            local_controllers=local_controllers,
            consensus_frequency=5
        )
        
        # Mock trajectory and risk measure
        trajectory = Mock(spec=TrajectoryData)
        trajectory.states = [np.array([1.0]) for _ in range(5)]
        risk_measure = SafetyViolationRisk(safety_threshold=0.5)
        
        # Update agents
        for i in range(6):  # Trigger consensus at update 5
            ma_controller.update_agent(1, trajectory, risk_measure)
            if i % 2 == 0:  # Update agent 2 less frequently
                ma_controller.update_agent(2, trajectory, risk_measure)
        
        # Get multi-agent certificate
        cert = ma_controller.get_multi_agent_certificate()
        
        assert hasattr(cert, 'global_risk_bound')
        assert hasattr(cert, 'agent_risk_bounds')
        assert hasattr(cert, 'consensus_confidence')
        assert 0.0 <= cert.global_risk_bound <= 1.0
        assert len(cert.agent_risk_bounds) == 2
    
    def test_system_health_monitoring(self):
        """Test system health monitoring."""
        # Simple setup
        agents = {1: AgentInfo(1, "agent", 0.05)}
        network = CommunicationNetwork(agents, CommunicationTopology.FULLY_CONNECTED)
        consensus = AverageConsensus()
        local_controllers = {1: AdaptiveRiskController(target_risk=0.05)}
        
        ma_controller = MultiAgentRiskController(
            agents=agents,
            communication_network=network,
            consensus_algorithm=consensus,
            local_controllers=local_controllers
        )
        
        # Get system health
        health = ma_controller.get_system_health()
        
        assert 'agent_health' in health
        assert 'communication_health' in health
        assert 'consensus_health' in health
        assert 'risk_certification' in health
        assert 'overall_health_score' in health
        
        # Health score should be between 0 and 1
        assert 0.0 <= health['overall_health_score'] <= 1.0
    
    def test_federated_learning_placeholder(self):
        """Test federated learning placeholder."""
        fed_learning = FederatedRiskLearning(num_agents=5)
        
        local_updates = {1: "update1", 2: "update2"}
        result = fed_learning.federated_risk_update(local_updates)
        
        # Should return the global model (None initially)
        assert result is None
    
    def test_scalable_consensus_placeholder(self):
        """Test scalable consensus placeholder."""
        scalable = ScalableConsensus(scalability_threshold=1000)
        
        agents = {1: AgentInfo(1, "agent", 0.05)}
        local_values = {1: 0.05}
        
        result = scalable.hierarchical_consensus(agents, local_values)
        
        # Should return mean of local values
        assert result == 0.05


class TestBenchmarkingFramework:
    """Test enhanced benchmarking framework."""
    
    def test_research_benchmark_integration(self):
        """Test that research extensions integrate with benchmarking."""
        # This would test integration between research modules and benchmarking
        # For now, just verify imports work
        from conforl.benchmarks.framework import BenchmarkRunner
        
        # Should be able to import research extensions
        from conforl.research import (
            CausalRiskController, 
            AdversarialRiskController,
            MultiAgentRiskController
        )
        
        assert CausalRiskController is not None
        assert AdversarialRiskController is not None
        assert MultiAgentRiskController is not None


@pytest.fixture
def mock_trajectory():
    """Create mock trajectory for testing."""
    trajectory = Mock(spec=TrajectoryData)
    trajectory.states = [np.random.randn(3) for _ in range(10)]
    trajectory.actions = [np.random.randn(2) for _ in range(10)]
    trajectory.rewards = np.random.randn(10).tolist()
    trajectory.next_states = [np.random.randn(3) for _ in range(10)]
    trajectory.dones = [False] * 9 + [True]
    return trajectory


@pytest.fixture 
def mock_risk_measure():
    """Create mock risk measure for testing."""
    return SafetyViolationRisk(safety_threshold=0.5)


class TestResearchIntegration:
    """Test integration between different research components."""
    
    def test_compositional_causal_integration(self):
        """Test integration of compositional and causal risk control."""
        # This could test how compositional risk control works with causal interventions
        # For now, verify they can coexist
        from conforl.research.compositional import CompositionalRiskController
        from conforl.research.causal import CausalRiskController
        
        # Both should be importable and instantiable with proper mocking
        assert CompositionalRiskController is not None
        assert CausalRiskController is not None
    
    def test_adversarial_multi_agent_integration(self):
        """Test integration of adversarial robustness with multi-agent systems."""
        from conforl.research.adversarial import AdversarialRiskController
        from conforl.research.multi_agent import MultiAgentRiskController
        
        # Should be able to import both
        assert AdversarialRiskController is not None
        assert MultiAgentRiskController is not None


# Performance benchmarks for research extensions
class TestResearchPerformance:
    """Performance tests for research extensions."""
    
    @pytest.mark.slow
    def test_causal_controller_performance(self, mock_trajectory, mock_risk_measure):
        """Test performance of causal risk controller."""
        # Setup
        nodes = ['state', 'action', 'reward']
        edges = {'state': ['action'], 'action': ['reward'], 'reward': []}
        graph = CausalGraph(nodes=nodes, edges=edges)
        
        base_controller = AdaptiveRiskController(target_risk=0.05)
        causal_controller = CausalRiskController(graph, base_controller)
        
        # Performance test
        start_time = time.time()
        
        for _ in range(100):
            causal_controller.update(mock_trajectory, mock_risk_measure)
        
        elapsed = time.time() - start_time
        
        # Should complete 100 updates in reasonable time
        assert elapsed < 5.0  # 5 seconds for 100 updates
    
    @pytest.mark.slow
    def test_multi_agent_scalability(self):
        """Test scalability of multi-agent system."""
        # Test with varying numbers of agents
        for n_agents in [5, 10, 20]:
            agents = {
                i: AgentInfo(i, "agent", 0.05) 
                for i in range(n_agents)
            }
            
            network = CommunicationNetwork(agents, CommunicationTopology.FULLY_CONNECTED)
            consensus = AverageConsensus()
            
            local_controllers = {
                i: AdaptiveRiskController(target_risk=0.05)
                for i in range(n_agents)
            }
            
            # Should be able to create controller without issues
            ma_controller = MultiAgentRiskController(
                agents=agents,
                communication_network=network,
                consensus_algorithm=consensus,
                local_controllers=local_controllers
            )
            
            # Should be able to get certificate
            cert = ma_controller.get_multi_agent_certificate()
            assert cert is not None


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_research.py -v
    pytest.main([__file__, "-v"])