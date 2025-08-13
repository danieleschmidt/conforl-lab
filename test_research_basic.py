#!/usr/bin/env python3
"""Basic Test Suite for Research Extensions (No NumPy Required).

Tests core functionality of research extensions without numerical dependencies.

Author: ConfoRL Research Team
License: Apache 2.0
"""

import sys
import time
import unittest
import warnings
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Import research modules (core functionality only)
from conforl.research.error_recovery import (
    ErrorRecoveryManager, RecoveryStrategy, with_retry, with_circuit_breaker
)
from conforl.research.validation_framework import (
    AlgorithmValidator, ValidationLevel, StatisticalTest, ValidationResult
)
from conforl.research.distributed_training import (
    DistributedTrainingManager, DistributionStrategy, ScalingPolicy,
    ResourceMonitor
)
from conforl.research.performance_optimization import (
    PerformanceOptimizer, OptimizationLevel, ComputeBackend,
    ComputationCache
)

# Import core components for integration testing
from conforl.utils.logging import get_logger
from conforl.utils.errors import ConfoRLError, ValidationError

logger = get_logger(__name__)


class TestErrorRecovery(unittest.TestCase):
    """Test error recovery and fault tolerance."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.recovery_manager = ErrorRecoveryManager(
            max_retries=2,
            retry_delay=0.1,  # Fast retries for testing
            circuit_breaker_threshold=3
        )
    
    def test_retry_mechanism(self):
        """Test retry mechanism."""
        call_count = 0
        
        @self.recovery_manager.with_recovery(RecoveryStrategy.RETRY)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return "success"
        
        result = flaky_function()
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # Failed twice, succeeded third time
    
    def test_circuit_breaker(self):
        """Test circuit breaker pattern."""
        call_count = 0
        
        @self.recovery_manager.with_recovery(RecoveryStrategy.CIRCUIT_BREAKER)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")
        
        # Should fail and eventually open circuit breaker
        for i in range(5):
            try:
                always_failing_function()
            except (RuntimeError, ConfoRLError):
                pass  # Expected failures
        
        # Check that circuit breaker opened
        stats = self.recovery_manager.get_error_statistics()
        self.assertGreater(len(stats['circuit_breakers']), 0)
    
    def test_fallback_mechanism(self):
        """Test fallback mechanism."""
        def fallback_function():
            return "fallback_result"
        
        self.recovery_manager.register_fallback("test_function", fallback_function)
        
        @self.recovery_manager.with_recovery(RecoveryStrategy.FALLBACK, "test_function")
        def failing_function():
            raise RuntimeError("Primary function failed")
        
        result = failing_function()
        self.assertEqual(result, "fallback_result")
    
    def test_graceful_degradation(self):
        """Test graceful degradation."""
        @self.recovery_manager.with_recovery(RecoveryStrategy.GRACEFUL_DEGRADATION)
        def predict_function():
            raise RuntimeError("Prediction failed")
        
        result = predict_function()
        # Should return conservative default
        self.assertEqual(result, 0.0)
    
    def test_error_statistics(self):
        """Test error statistics collection."""
        @self.recovery_manager.with_recovery(RecoveryStrategy.RETRY)
        def test_function():
            raise ValueError("Test error")
        
        try:
            test_function()
        except:
            pass
        
        stats = self.recovery_manager.get_error_statistics()
        self.assertIn('total_errors', stats)
        self.assertIn('error_types', stats)
        self.assertGreater(stats['total_errors'], 0)


class TestValidationFramework(unittest.TestCase):
    """Test validation framework."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = AlgorithmValidator(
            validation_level=ValidationLevel.COMPREHENSIVE,
            min_sample_size=10,  # Lower for testing
            enable_statistical_tests=False  # Disable scipy-dependent tests
        )
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        self.assertEqual(self.validator.validation_level, ValidationLevel.COMPREHENSIVE)
        self.assertEqual(self.validator.min_sample_size, 10)
        self.assertFalse(self.validator.enable_statistical_tests)
    
    def test_input_validation(self):
        """Test input data validation."""
        # Valid data
        algorithm_results = {
            'return': [100.0, 105.0, 95.0, 110.0],
            'risk': [0.05, 0.04, 0.06, 0.03]
        }
        
        baseline_results = {
            'return': [95.0, 100.0, 90.0, 105.0],
            'risk': [0.08, 0.07, 0.09, 0.06]
        }
        
        # Should not raise exception
        try:
            self.validator._validate_input_data(algorithm_results, baseline_results)
        except Exception as e:
            self.fail(f"Valid data validation failed: {e}")
        
        # Invalid data (empty)
        empty_results = {'return': []}
        
        with self.assertRaises(ValidationError):
            self.validator._validate_input_data(empty_results, baseline_results)
    
    def test_validation_result_creation(self):
        """Test validation result creation."""
        result = ValidationResult(
            test_name="test_validation",
            passed=True,
            score=0.95,
            threshold=0.90,
            p_value=0.02
        )
        
        self.assertEqual(result.test_name, "test_validation")
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 0.95)
        self.assertEqual(result.threshold, 0.90)
        self.assertEqual(result.p_value, 0.02)
    
    def test_validation_summary(self):
        """Test validation summary generation."""
        summary = self.validator.get_validation_summary()
        
        self.assertIn('total_tests_run', summary)
        self.assertIn('tests_passed', summary)
        self.assertIn('pass_rate', summary)
        self.assertIn('validation_level', summary)


class TestDistributedTraining(unittest.TestCase):
    """Test distributed training components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.distributed_manager = DistributedTrainingManager(
            initial_workers=2,  # Small for testing
            enable_auto_scaling=False  # Disable for deterministic testing
        )
    
    def test_distributed_manager_initialization(self):
        """Test distributed manager initialization."""
        self.assertEqual(self.distributed_manager.initial_workers, 2)
        self.assertEqual(self.distributed_manager.strategy, DistributionStrategy.DATA_PARALLEL)
        self.assertFalse(self.distributed_manager.enable_auto_scaling)
    
    def test_distributed_manager_lifecycle(self):
        """Test distributed manager start/shutdown."""
        # Start manager
        self.distributed_manager.start()
        
        # Check that workers are initialized
        self.assertGreater(len(self.distributed_manager.workers), 0)
        
        # Shutdown manager
        self.distributed_manager.shutdown(wait=True, timeout=5.0)
        
        # Check that workers are cleaned up
        self.assertEqual(len(self.distributed_manager.workers), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        self.distributed_manager.start()
        
        try:
            metrics = self.distributed_manager.get_performance_metrics()
            
            self.assertIn('total_tasks_submitted', metrics)
            self.assertIn('current_workers', metrics)
            self.assertIn('distribution_strategy', metrics)
            self.assertIn('resource_metrics', metrics)
            
        finally:
            self.distributed_manager.shutdown(wait=True, timeout=5.0)
    
    def test_resource_monitor(self):
        """Test resource monitoring."""
        monitor = ResourceMonitor(monitoring_interval=0.1)  # Fast for testing
        
        monitor.start_monitoring()
        time.sleep(0.5)  # Let it collect some data
        monitor.stop_monitoring()
        
        # Check that monitoring was active
        self.assertFalse(monitor.monitoring_active)
        
        # Test scaling recommendation
        recommendation = monitor.get_scaling_recommendation(
            current_workers=2, queue_size=10
        )
        self.assertIsInstance(recommendation, int)
        self.assertGreater(recommendation, 0)


class TestPerformanceOptimization(unittest.TestCase):
    """Test performance optimization components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.optimizer = PerformanceOptimizer(
            optimization_level=OptimizationLevel.BASIC,
            enable_jit=False,  # Disable JIT for testing
            enable_caching=True,
            enable_memory_pooling=False  # Disable numpy-dependent pooling
        )
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        self.assertEqual(self.optimizer.optimization_level, OptimizationLevel.BASIC)
        self.assertFalse(self.optimizer.enable_jit)
        self.assertTrue(self.optimizer.enable_caching)
        self.assertFalse(self.optimizer.enable_memory_pooling)
    
    def test_function_optimization(self):
        """Test function optimization."""
        def test_function(x, y):
            return x * y + x - y
        
        optimized_func = self.optimizer.optimize_function(test_function)
        
        # Test that optimized function works
        result = optimized_func(5, 3)
        expected = 5 * 3 + 5 - 3  # 17
        self.assertEqual(result, expected)
    
    def test_computation_cache(self):
        """Test computation cache."""
        cache = ComputationCache(max_cache_size=10, ttl_seconds=1.0)
        
        # Test cache operations
        cache.put("key1", "value1")
        result = cache.get("key1")
        self.assertEqual(result, "value1")
        
        # Test cache miss
        result = cache.get("nonexistent_key")
        self.assertIsNone(result)
        
        # Test TTL
        time.sleep(1.1)  # Wait for TTL to expire
        result = cache.get("key1")
        self.assertIsNone(result)  # Should be expired
        
        # Get stats
        stats = cache.get_stats()
        self.assertIn('hit_count', stats)
        self.assertIn('miss_count', stats)
    
    def test_performance_report(self):
        """Test performance reporting."""
        def test_function(x):
            time.sleep(0.01)  # Small delay to measure
            return x * 2
        
        optimized_func = self.optimizer.optimize_function(test_function, "test_func")
        
        # Call function a few times
        for i in range(3):
            optimized_func(i)
        
        # Get performance report
        report = self.optimizer.get_performance_report()
        
        self.assertIn('function_profiles', report)
        self.assertIn('optimization_level', report)
        self.assertIn('enabled_optimizations', report)
        
        if 'test_func' in report['function_profiles']:
            profile = report['function_profiles']['test_func']
            self.assertGreater(profile['call_count'], 0)
            self.assertGreater(profile['total_time'], 0)


class TestIntegration(unittest.TestCase):
    """Integration tests for research components."""
    
    def test_end_to_end_research_pipeline(self):
        """Test end-to-end research pipeline."""
        try:
            # Initialize components
            optimizer = PerformanceOptimizer(
                optimization_level=OptimizationLevel.BASIC,
                enable_jit=False,  # Disable for testing
                enable_memory_pooling=False  # Disable numpy dependency
            )
            
            error_recovery = ErrorRecoveryManager(max_retries=2)
            validator = AlgorithmValidator(
                min_sample_size=5,
                enable_statistical_tests=False  # Disable scipy dependency
            )
            
            # Define research function with optimizations
            @optimizer.optimize_function
            @error_recovery.with_recovery(RecoveryStrategy.RETRY)
            def research_algorithm(data_size: int):
                # Simulate research computation without numpy
                results = [1.0 + (i % 10) * 0.1 for i in range(data_size)]
                metrics = {
                    'accuracy': sum(1 for r in results if r > 0.5) / len(results),
                    'variance': sum((r - 1.0)**2 for r in results) / len(results)
                }
                return {'results': results, 'metrics': metrics}
            
            # Run research algorithm
            output = research_algorithm(10)
            
            self.assertIn('results', output)
            self.assertIn('metrics', output)
            self.assertEqual(len(output['results']), 10)
            
            # Get performance metrics
            perf_report = optimizer.get_performance_report()
            self.assertIn('function_profiles', perf_report)
            
            # Get error statistics
            error_stats = error_recovery.get_error_statistics()
            self.assertIn('total_errors', error_stats)
            
            logger.info("End-to-end research pipeline test completed successfully")
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            raise
    
    def test_component_health_checks(self):
        """Test health checks for all components."""
        # Error recovery health check
        error_recovery = ErrorRecoveryManager()
        health = error_recovery.health_check()
        
        self.assertIn('status', health)
        self.assertIn('recent_error_count', health)
        
        # Optimizer performance report
        optimizer = PerformanceOptimizer(enable_memory_pooling=False)
        report = optimizer.get_performance_report()
        
        self.assertIn('optimization_level', report)
        self.assertIn('enabled_optimizations', report)
        
        # Distributed manager metrics
        manager = DistributedTrainingManager(initial_workers=1, enable_auto_scaling=False)
        manager.start()
        
        try:
            metrics = manager.get_performance_metrics()
            self.assertIn('current_workers', metrics)
            self.assertIn('distribution_strategy', metrics)
        finally:
            manager.shutdown(wait=True, timeout=2.0)


def run_basic_tests():
    """Run all basic tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestErrorRecovery,
        TestValidationFramework,
        TestDistributedTraining,
        TestPerformanceOptimization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == '__main__':
    print("=" * 80)
    print("ConfoRL Research Extensions - Basic Test Suite (No NumPy)")
    print("=" * 80)
    
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)
    
    # Run tests
    success = run_basic_tests()
    
    if success:
        print("\n" + "=" * 80)
        print("ALL BASIC TESTS PASSED! ✅")
        print("Research extensions core functionality is working correctly.")
        print("Note: Advanced numerical tests require NumPy/SciPy installation.")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("SOME TESTS FAILED! ❌")
        print("Please check the output above for details.")
        print("=" * 80)
        sys.exit(1)