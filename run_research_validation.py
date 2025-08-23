#!/usr/bin/env python3
"""Research Validation and Benchmarking Suite for ConfoRL"""

import sys
import os
import time
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath('.'))

def research_conformal_coverage_test():
    """Test conformal prediction coverage guarantees"""
    print("🔬 Testing Conformal Coverage Guarantees...")
    
    try:
        from conforl.core.conformal import SplitConformalPredictor
        
        # Create conformal predictor with 95% coverage
        predictor = SplitConformalPredictor(coverage=0.95)
        
        # Generate synthetic calibration data
        np.random.seed(42)
        cal_scores = np.random.exponential(2.0, 1000)
        predictor.fit(cal_scores)
        
        # Test coverage on new data
        test_scores = np.random.exponential(2.0, 500)
        prediction_sets = []
        
        for score in test_scores:
            pred_set = predictor.predict(score, return_prediction_set=True)
            prediction_sets.append(pred_set)
        
        # Calculate empirical coverage
        true_positives = sum(1 for ps in prediction_sets if ps.contains_truth)
        empirical_coverage = true_positives / len(prediction_sets)
        
        print(f"✅ Target coverage: 0.95")
        print(f"✅ Empirical coverage: {empirical_coverage:.3f}")
        
        # Coverage should be close to target (within 5%)
        coverage_valid = abs(empirical_coverage - 0.95) < 0.05
        print(f"✅ Coverage validity: {coverage_valid}")
        
        return {
            'test': 'conformal_coverage',
            'target_coverage': 0.95,
            'empirical_coverage': empirical_coverage,
            'coverage_valid': coverage_valid,
            'passed': coverage_valid
        }
        
    except Exception as e:
        print(f"❌ Conformal coverage test failed: {e}")
        return {'test': 'conformal_coverage', 'passed': False, 'error': str(e)}

def research_algorithm_comparison():
    """Compare different algorithms on benchmark tasks"""
    print("\n🔬 Running Algorithm Comparison Study...")
    
    try:
        import gymnasium as gym
        from conforl.algorithms.sac import ConformaSAC
        from conforl.algorithms.ppo import ConformaPPO
        from conforl.risk.controllers import AdaptiveRiskController
        
        # Test environments
        environments = ['CartPole-v1', 'Pendulum-v1']
        algorithms = [
            ('ConformaSAC', ConformaSAC),
            ('ConformaPPO', ConformaPPO)
        ]
        
        results = []
        
        for env_name in environments:
            env = gym.make(env_name)
            
            for alg_name, alg_class in algorithms:
                try:
                    print(f"  Testing {alg_name} on {env_name}...")
                    
                    risk_controller = AdaptiveRiskController(
                        target_risk=0.05, 
                        confidence=0.95
                    )
                    
                    agent = alg_class(env=env, risk_controller=risk_controller)
                    
                    # Test prediction performance
                    state, _ = env.reset()
                    start_time = time.time()
                    
                    total_reward = 0
                    violations = 0
                    
                    for step in range(10):
                        action, certificate = agent.predict(
                            state, 
                            return_risk_certificate=True
                        )
                        
                        next_state, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        
                        # Check for safety violations
                        if certificate.risk_bound > 0.06:  # Slightly above target
                            violations += 1
                        
                        if terminated or truncated:
                            state, _ = env.reset()
                        else:
                            state = next_state
                    
                    prediction_time = time.time() - start_time
                    avg_prediction_time = prediction_time / 10
                    
                    result = {
                        'algorithm': alg_name,
                        'environment': env_name,
                        'total_reward': total_reward,
                        'safety_violations': violations,
                        'avg_prediction_time': avg_prediction_time,
                        'passed': violations <= 2  # Allow some tolerance
                    }
                    
                    results.append(result)
                    print(f"    ✅ Reward: {total_reward:.2f}, Violations: {violations}, Time: {avg_prediction_time:.4f}s")
                    
                except Exception as e:
                    print(f"    ❌ Failed: {e}")
                    results.append({
                        'algorithm': alg_name,
                        'environment': env_name,
                        'passed': False,
                        'error': str(e)
                    })
            
            env.close()
        
        # Calculate summary statistics
        passed_tests = sum(1 for r in results if r.get('passed', False))
        total_tests = len(results)
        
        return {
            'test': 'algorithm_comparison',
            'results': results,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'passed': passed_tests >= total_tests * 0.75  # 75% pass rate
        }
        
    except Exception as e:
        print(f"❌ Algorithm comparison failed: {e}")
        return {'test': 'algorithm_comparison', 'passed': False, 'error': str(e)}

def research_safety_violation_analysis():
    """Analyze safety violation rates across different scenarios"""
    print("\n🔬 Running Safety Violation Analysis...")
    
    try:
        from conforl.risk.measures import SafetyViolationRisk
        from conforl.core.types import TrajectoryData
        
        # Create safety violation risk measure
        risk_measure = SafetyViolationRisk()
        
        # Test different violation scenarios
        scenarios = []
        
        # Scenario 1: Safe trajectory (no violations)
        safe_trajectory = TrajectoryData(
            states=[[1, 2, 3, 4]] * 10,
            actions=[[0.1]] * 10,
            rewards=[1.0] * 10,
            infos=[{}] * 10  # No constraint violations
        )
        
        safe_risk = risk_measure.compute(safe_trajectory)
        scenarios.append({
            'scenario': 'safe_trajectory',
            'risk_score': safe_risk,
            'expected_risk': 0.0,
            'passed': abs(safe_risk - 0.0) < 0.01
        })
        
        # Scenario 2: Unsafe trajectory (with violations)
        unsafe_trajectory = TrajectoryData(
            states=[[1, 2, 3, 4]] * 10,
            actions=[[0.1]] * 10,
            rewards=[1.0] * 10,
            infos=[{'constraint_violation': 1.0} if i % 3 == 0 else {} for i in range(10)]
        )
        
        unsafe_risk = risk_measure.compute(unsafe_trajectory)
        expected_violation_rate = 4 / 10  # 4 out of 10 steps have violations
        scenarios.append({
            'scenario': 'unsafe_trajectory',
            'risk_score': unsafe_risk,
            'expected_risk': expected_violation_rate,
            'passed': abs(unsafe_risk - expected_violation_rate) < 0.1
        })
        
        print(f"✅ Safe trajectory risk: {safe_risk:.3f}")
        print(f"✅ Unsafe trajectory risk: {unsafe_risk:.3f}")
        
        # Summary
        passed_scenarios = sum(1 for s in scenarios if s['passed'])
        
        return {
            'test': 'safety_violation_analysis',
            'scenarios': scenarios,
            'passed_scenarios': passed_scenarios,
            'total_scenarios': len(scenarios),
            'passed': passed_scenarios == len(scenarios)
        }
        
    except Exception as e:
        print(f"❌ Safety violation analysis failed: {e}")
        return {'test': 'safety_violation_analysis', 'passed': False, 'error': str(e)}

def research_performance_benchmark():
    """Benchmark performance across different scales"""
    print("\n🔬 Running Performance Benchmark...")
    
    try:
        import gymnasium as gym
        from conforl.algorithms.sac import ConformaSAC
        from conforl.risk.controllers import AdaptiveRiskController
        
        # Test different batch sizes
        batch_sizes = [1, 5, 10, 20]
        performance_results = []
        
        env = gym.make('CartPole-v1')
        risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        agent = ConformaSAC(env=env, risk_controller=risk_controller)
        
        for batch_size in batch_sizes:
            print(f"  Testing batch size: {batch_size}")
            
            # Generate batch of states
            states = []
            for _ in range(batch_size):
                state, _ = env.reset()
                states.append(state)
            
            # Time batch predictions
            start_time = time.time()
            for state in states:
                action, cert = agent.predict(state, return_risk_certificate=True)
            total_time = time.time() - start_time
            
            avg_time_per_prediction = total_time / batch_size
            throughput = batch_size / total_time
            
            performance_results.append({
                'batch_size': batch_size,
                'total_time': total_time,
                'avg_time_per_prediction': avg_time_per_prediction,
                'throughput': throughput
            })
            
            print(f"    ✅ Avg time: {avg_time_per_prediction:.4f}s, Throughput: {throughput:.1f} pred/s")
        
        env.close()
        
        # Check performance scaling
        baseline_throughput = performance_results[0]['throughput']
        scaling_efficient = all(
            r['avg_time_per_prediction'] < 0.1  # Less than 100ms per prediction
            for r in performance_results
        )
        
        return {
            'test': 'performance_benchmark',
            'performance_results': performance_results,
            'baseline_throughput': baseline_throughput,
            'scaling_efficient': scaling_efficient,
            'passed': scaling_efficient
        }
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return {'test': 'performance_benchmark', 'passed': False, 'error': str(e)}

def generate_research_plots(results: List[Dict]):
    """Generate research plots and visualizations"""
    print("\n📊 Generating Research Plots...")
    
    try:
        # Create plots directory
        plots_dir = Path('research_plots')
        plots_dir.mkdir(exist_ok=True)
        
        # Plot 1: Algorithm Comparison
        alg_comparison = next((r for r in results if r['test'] == 'algorithm_comparison'), None)
        if alg_comparison and 'results' in alg_comparison:
            alg_results = alg_comparison['results']
            
            # Group by algorithm
            algorithms = {}
            for result in alg_results:
                if 'algorithm' in result and 'total_reward' in result:
                    alg = result['algorithm']
                    if alg not in algorithms:
                        algorithms[alg] = {'rewards': [], 'violations': []}
                    algorithms[alg]['rewards'].append(result['total_reward'])
                    algorithms[alg]['violations'].append(result['safety_violations'])
            
            if algorithms:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Rewards comparison
                alg_names = list(algorithms.keys())
                rewards = [np.mean(algorithms[alg]['rewards']) for alg in alg_names]
                ax1.bar(alg_names, rewards)
                ax1.set_title('Algorithm Performance Comparison')
                ax1.set_ylabel('Average Total Reward')
                ax1.tick_params(axis='x', rotation=45)
                
                # Safety violations comparison
                violations = [np.mean(algorithms[alg]['violations']) for alg in alg_names]
                ax2.bar(alg_names, violations, color='red', alpha=0.7)
                ax2.set_title('Safety Violations Comparison')
                ax2.set_ylabel('Average Safety Violations')
                ax2.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(plots_dir / 'algorithm_comparison.png', dpi=150, bbox_inches='tight')
                plt.close()
                print("✅ Algorithm comparison plot saved")
        
        # Plot 2: Performance Scaling
        perf_benchmark = next((r for r in results if r['test'] == 'performance_benchmark'), None)
        if perf_benchmark and 'performance_results' in perf_benchmark:
            perf_results = perf_benchmark['performance_results']
            
            batch_sizes = [r['batch_size'] for r in perf_results]
            throughputs = [r['throughput'] for r in perf_results]
            avg_times = [r['avg_time_per_prediction'] * 1000 for r in perf_results]  # Convert to ms
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Throughput scaling
            ax1.plot(batch_sizes, throughputs, 'bo-')
            ax1.set_title('Throughput Scaling')
            ax1.set_xlabel('Batch Size')
            ax1.set_ylabel('Throughput (predictions/sec)')
            ax1.grid(True)
            
            # Average prediction time
            ax2.plot(batch_sizes, avg_times, 'ro-')
            ax2.set_title('Prediction Latency')
            ax2.set_xlabel('Batch Size')
            ax2.set_ylabel('Avg Prediction Time (ms)')
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'performance_scaling.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("✅ Performance scaling plot saved")
        
        return True
        
    except Exception as e:
        print(f"❌ Plot generation failed: {e}")
        return False

def main():
    """Run comprehensive research validation"""
    print("=" * 60)
    print("🔬 RESEARCH VALIDATION AND BENCHMARKING SUITE")
    print("=" * 60)
    
    # Run all research tests
    research_tests = [
        research_conformal_coverage_test,
        research_algorithm_comparison,
        research_safety_violation_analysis,
        research_performance_benchmark
    ]
    
    results = []
    for test_func in research_tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")
            results.append({
                'test': test_func.__name__, 
                'passed': False, 
                'error': str(e)
            })
    
    # Generate research plots
    plot_success = generate_research_plots(results)
    
    # Calculate summary statistics
    passed_tests = sum(1 for r in results if r.get('passed', False))
    total_tests = len(results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    # Save research report
    research_report = {
        'timestamp': time.time(),
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'success_rate': success_rate,
        'results': results,
        'plots_generated': plot_success
    }
    
    with open('research_validation_report.json', 'w') as f:
        json.dump(research_report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 RESEARCH VALIDATION SUMMARY")
    print("=" * 60)
    
    for result in results:
        status = "✅" if result.get('passed', False) else "❌"
        test_name = result.get('test', 'unknown')
        print(f"{status} {test_name}")
        
        if not result.get('passed', False) and 'error' in result:
            print(f"    Error: {result['error'][:100]}...")
    
    print(f"\n📈 SUCCESS RATE: {success_rate:.1%} ({passed_tests}/{total_tests})")
    
    if success_rate >= 0.75:
        print("✅ Research validation PASSED")
        overall_success = True
    else:
        print("❌ Research validation FAILED")
        overall_success = False
    
    if plot_success:
        print("✅ Research plots generated in research_plots/")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)