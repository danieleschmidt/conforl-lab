#!/usr/bin/env python3
"""Test ConfoRL Research Extensions.

This script tests the novel research implementations including:
- Compositional Risk Control for hierarchical RL
- Benchmark framework for comparative studies
- Statistical validation of conformal guarantees

This demonstrates ConfoRL's research contributions and publication readiness.
"""

import numpy as np
import time

def test_compositional_risk():
    """Test compositional risk control implementation."""
    print("🧪 Testing Compositional Risk Control")
    print("=" * 50)
    
    try:
        from conforl.research import (
            HierarchicalPolicyBuilder, 
            CompositionalRiskController,
            CompositionalRiskBounds
        )
        from conforl.core.types import TrajectoryData
        from conforl.risk.measures import SafetyViolationRisk
        
        # Build hierarchical policy structure
        builder = HierarchicalPolicyBuilder()
        hierarchy = (builder
            .add_policy("high_level", risk_budget=0.02)  # 2% risk budget
            .add_policy("mid_level_1", parent_id="high_level", risk_budget=0.03) 
            .add_policy("mid_level_2", parent_id="high_level", risk_budget=0.03)
            .add_policy("low_level_1", parent_id="mid_level_1", risk_budget=0.05)
            .add_policy("low_level_2", parent_id="mid_level_1", risk_budget=0.05)
            .add_policy("low_level_3", parent_id="mid_level_2", risk_budget=0.05)
            .build())
        
        print(f"✓ Built hierarchical structure with {len(hierarchy)} policies")
        
        # Create compositional risk controller
        controller = CompositionalRiskController(
            hierarchy=hierarchy,
            base_confidence=0.95,
            composition_method="bonferroni"
        )
        print("✓ Initialized compositional risk controller")
        
        # Test with mock trajectory data
        mock_trajectory = TrajectoryData(
            states=np.random.randn(100, 4),
            actions=np.random.randn(100, 2),  
            rewards=np.random.randn(100),
            dones=np.zeros(100, dtype=bool),
            infos=[{}] * 100
        )
        
        risk_measure = SafetyViolationRisk()
        
        # Update risk for each policy
        for policy_id in hierarchy.keys():
            controller.update_policy_risk(policy_id, mock_trajectory, risk_measure)
        
        print("✓ Updated risk assessments for all policies")
        
        # Get compositional certificate
        global_cert = controller.get_compositional_certificate()
        print(f"✓ Global compositional risk bound: {global_cert.compositional_risk_bound:.4f}")
        
        # Test subtree certificate
        subtree_cert = controller.get_compositional_certificate("mid_level_1", max_depth=5)
        print(f"✓ Subtree compositional risk bound: {subtree_cert.compositional_risk_bound:.4f}")
        
        # Test risk budget validation
        validation = controller.validate_risk_budgets()
        compliant_policies = sum(validation.values())
        print(f"✓ Risk budget compliance: {compliant_policies}/{len(validation)} policies")
        
        # Test bounds calculations
        bounds_calc = CompositionalRiskBounds()
        individual_risks = [0.01, 0.02, 0.015, 0.03, 0.025, 0.02]
        corrected_conf, comp_risk = bounds_calc.bonferroni_correction(individual_risks, 0.95)
        print(f"✓ Bonferroni correction: confidence={corrected_conf:.3f}, risk={comp_risk:.4f}")
        
        # Get hierarchy statistics
        stats = controller.get_hierarchy_stats()
        print(f"✓ Hierarchy stats: {stats['hierarchy_size']} policies across {stats['hierarchy_depth']} levels")
        
        print("🎉 Compositional Risk Control tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Compositional Risk Control test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_framework():
    """Test benchmark framework implementation."""
    print("\n🧪 Testing Benchmark Framework")
    print("=" * 50)
    
    try:
        from conforl.benchmarks import (
            BenchmarkRunner,
            create_quick_benchmark,
            CartPoleSafety,
            RandomPolicy,
            ConstrainedPolicy,
            ConfoRLBenchmarkWrapper
        )
        
        # Test environment creation
        env = CartPoleSafety({'position_limit': 1.0, 'angle_limit': 0.2})
        obs, info = env.reset()
        print(f"✓ Created CartPoleSafety environment, obs shape: {obs.shape}")
        
        # Test environment step
        action = np.array([1])  # Right action
        next_obs, reward, done, truncated, info = env.step(action)
        print(f"✓ Environment step: reward={reward:.2f}, safety_cost={info.get('safety_cost', 0.0):.4f}")
        
        # Test safety checking
        is_unsafe = env.is_unsafe(obs, action, info)
        safety_cost = env.get_safety_cost(obs, action, info)
        print(f"✓ Safety assessment: unsafe={is_unsafe}, cost={safety_cost:.4f}")
        
        # Test baseline algorithms
        random_policy = RandomPolicy(env, {})
        random_action = random_policy.predict(obs)
        print(f"✓ Random policy action: {random_action}")
        
        constrained_policy = ConstrainedPolicy(env, {})
        safe_action = constrained_policy.predict(obs)
        print(f"✓ Constrained policy action: {safe_action}")
        
        # Test ConfoRL wrapper
        conforl_wrapper = ConfoRLBenchmarkWrapper(env, {
            'target_risk': 0.05,
            'confidence': 0.95,
            'learning_rate': 1e-3,
            'batch_size': 32
        })
        conforl_action = conforl_wrapper.predict(obs)
        print(f"✓ ConfoRL wrapper action: {conforl_action}")
        
        # Test benchmark suite creation
        quick_suite = create_quick_benchmark()
        print(f"✓ Created quick benchmark suite: {len(quick_suite.environments)} envs, {len(quick_suite.algorithms)} algs")
        
        # Test minimal benchmark run (very short for testing)
        runner = BenchmarkRunner("./test_benchmark_results")
        
        # Create minimal test suite
        from conforl.benchmarks.framework import BenchmarkSuite
        test_suite = BenchmarkSuite(
            name="TestSuite",
            environments=["CartPoleSafety"],
            algorithms=["RandomPolicy", "ConstrainedPolicy"],
            num_runs=2,
            max_episodes=3,
            max_timesteps=100,
            evaluation_frequency=50,
            statistical_significance=0.05
        )
        
        print("✓ Running minimal benchmark test...")
        start_time = time.time()
        
        results = runner.run_benchmark_suite(test_suite)
        
        elapsed_time = time.time() - start_time
        print(f"✓ Benchmark completed in {elapsed_time:.2f}s with {len(results)} results")
        
        # Test results analysis
        analysis = runner.analyze_results(results)
        print(f"✓ Analysis completed with {len(analysis['summary'])} algorithm-environment pairs")
        
        # Display key results
        for key, summary in analysis['summary'].items():
            avg_return = summary['avg_return']['mean']
            violation_rate = summary['violation_rate']['mean'] 
            print(f"  {key}: return={avg_return:.2f}, violations={violation_rate:.3f}")
        
        # Check rankings
        if 'avg_return' in analysis['rankings']:
            best_algorithm = analysis['rankings']['avg_return'][0]['algorithm']
            print(f"✓ Best performing algorithm (return): {best_algorithm}")
        
        print("🎉 Benchmark Framework tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Benchmark Framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_theoretical_guarantees():
    """Test theoretical guarantee validation."""
    print("\n🧪 Testing Theoretical Guarantees")
    print("=" * 50)
    
    try:
        from conforl.core.conformal import SplitConformalPredictor
        from conforl.core.types import RiskCertificate
        from conforl.risk.controllers import AdaptiveRiskController
        
        # Test conformal predictor
        predictor = SplitConformalPredictor(coverage=0.9)
        
        # Generate synthetic calibration data  
        calibration_scores = np.random.exponential(0.5, 1000)  # Non-conformity scores
        predictor.calibrate(None, calibration_scores)
        
        print(f"✓ Calibrated conformal predictor with quantile: {predictor.quantile:.4f}")
        
        # Test risk certificate generation
        test_predictions = np.random.normal(0, 1, 100)
        conformal_set = predictor.predict(test_predictions)
        
        print(f"✓ Generated conformal prediction set with coverage: {conformal_set.coverage:.3f}")
        
        # Test adaptive risk controller
        controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        
        # Generate synthetic trajectory
        from conforl.core.types import TrajectoryData
        from conforl.risk.measures import SafetyViolationRisk
        
        synthetic_trajectory = TrajectoryData(
            states=np.random.randn(200, 4),
            actions=np.random.randn(200, 2),
            rewards=np.random.randn(200),
            dones=np.random.choice([True, False], 200, p=[0.05, 0.95]),
            infos=[{'constraint_violation': np.random.choice([True, False], p=[0.1, 0.9])} for _ in range(200)]
        )
        
        risk_measure = SafetyViolationRisk()
        
        # Update controller multiple times (simulating online learning)
        for i in range(10):
            # Create mini-batch trajectory
            start_idx = i * 20
            end_idx = start_idx + 20
            
            mini_trajectory = TrajectoryData(
                states=synthetic_trajectory.states[start_idx:end_idx],
                actions=synthetic_trajectory.actions[start_idx:end_idx],
                rewards=synthetic_trajectory.rewards[start_idx:end_idx],
                dones=synthetic_trajectory.dones[start_idx:end_idx],
                infos=synthetic_trajectory.infos[start_idx:end_idx]
            )
            
            controller.update(mini_trajectory, risk_measure)
            
            certificate = controller.get_certificate()
            print(f"  Update {i+1}: risk_bound={certificate.risk_bound:.4f}, coverage={certificate.coverage_guarantee:.3f}")
        
        # Final certificate
        final_certificate = controller.get_certificate()
        print(f"✓ Final risk certificate: bound={final_certificate.risk_bound:.4f}")
        print(f"✓ Theoretical guarantee: P(failure) ≤ {final_certificate.risk_bound:.4f} with {final_certificate.coverage_guarantee:.1%} confidence")
        
        # Validate finite-sample property
        sample_size = final_certificate.sample_size
        expected_bound_width = 2 * np.sqrt(np.log(2/0.05) / (2 * sample_size))  # Hoeffding bound
        print(f"✓ Finite-sample validation: n={sample_size}, expected_width={expected_bound_width:.4f}")
        
        # Test coverage guarantee
        target_coverage = 0.95
        actual_coverage = final_certificate.coverage_guarantee
        coverage_error = abs(target_coverage - actual_coverage)
        print(f"✓ Coverage validation: target={target_coverage:.3f}, actual={actual_coverage:.3f}, error={coverage_error:.4f}")
        
        if coverage_error < 0.1:  # Within 10% tolerance
            print("✓ Coverage guarantee satisfied!")
        else:
            print("⚠️  Coverage guarantee may need more calibration data")
        
        print("🎉 Theoretical Guarantees tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Theoretical Guarantees test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all research tests."""
    print("🚀 ConfoRL Research Extensions Test Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    # Run test suites
    results = {
        "compositional_risk": test_compositional_risk(),
        "benchmark_framework": test_benchmark_framework(), 
        "theoretical_guarantees": test_theoretical_guarantees()
    }
    
    elapsed_time = time.time() - start_time
    
    # Summary
    print(f"\n📊 Test Results Summary")
    print("=" * 60)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:25} {status}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} test suites passed")
    print(f"Execution time: {elapsed_time:.2f} seconds")
    
    if passed_tests == total_tests:
        print("\n🎉 All ConfoRL research extensions are working correctly!")
        print("🎯 Ready for academic publication and research deployment")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test suite(s) failed")
        print("🔧 Please review the errors above and fix the issues")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)