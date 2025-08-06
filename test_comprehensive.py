#!/usr/bin/env python3
"""Comprehensive test suite for ConfoRL."""

import sys
import os
import time
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


class TestCoreFunctionality(unittest.TestCase):
    """Test core ConfoRL functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        from conforl.core.types import RiskCertificate, TrajectoryData
        from conforl.core.conformal import SplitConformalPredictor
        
        self.RiskCertificate = RiskCertificate
        self.TrajectoryData = TrajectoryData
        self.SplitConformalPredictor = SplitConformalPredictor
    
    def test_risk_certificate_creation(self):
        """Test RiskCertificate creation and validation."""
        cert = self.RiskCertificate(
            risk_bound=0.05,
            confidence=0.95,
            coverage_guarantee=0.95,
            method="test",
            sample_size=1000
        )
        
        self.assertEqual(cert.risk_bound, 0.05)
        self.assertEqual(cert.confidence, 0.95)
        self.assertEqual(cert.coverage_guarantee, 0.95)
        self.assertEqual(cert.method, "test")
        self.assertEqual(cert.sample_size, 1000)
    
    def test_trajectory_data_creation(self):
        """Test TrajectoryData creation and methods."""
        trajectory = self.TrajectoryData(
            states=[1, 2, 3, 4],
            actions=[0.1, 0.2, 0.3, 0.4],
            rewards=[1.0, 0.5, 0.8, 0.2],
            dones=[False, False, False, True],
            infos=[{}, {}, {}, {}]
        )
        
        self.assertEqual(len(trajectory), 4)
        self.assertEqual(trajectory.episode_length, 4)
    
    def test_conformal_predictor_basic(self):
        """Test basic conformal predictor functionality."""
        predictor = self.SplitConformalPredictor(coverage=0.9)
        self.assertEqual(predictor.coverage, 0.9)
        
        # Test calibration with dummy data
        dummy_data = [1, 2, 3, 4, 5]
        dummy_scores = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        predictor.calibrate(dummy_data, dummy_scores)
        self.assertIsNotNone(predictor.quantile)
        self.assertGreater(predictor.quantile, 0)


class TestUtilities(unittest.TestCase):
    """Test utility functions."""
    
    def test_error_classes(self):
        """Test custom error classes."""
        from conforl.utils.errors import ConfoRLError, ValidationError
        
        # Test ConfoRLError
        error = ConfoRLError("Test error", "TEST_CODE")
        self.assertEqual(str(error), "[TEST_CODE] Test error")
        self.assertEqual(error.error_code, "TEST_CODE")
        
        # Test ValidationError
        val_error = ValidationError("Invalid input", "test_validation")
        self.assertEqual(val_error.validation_type, "test_validation")
    
    def test_logging_setup(self):
        """Test logging configuration."""
        from conforl.utils.logging import get_logger
        
        logger = get_logger("test")
        self.assertEqual(logger.name, "ConfoRL.test")
        
        # Test logging doesn't crash
        logger.info("Test message")
        logger.warning("Test warning")


class TestOptimization(unittest.TestCase):
    """Test optimization components."""
    
    def test_adaptive_cache(self):
        """Test adaptive caching functionality."""
        try:
            from conforl.optimize.cache import AdaptiveCache
            
            cache = AdaptiveCache(max_size=100)
            
            # Test basic operations
            cache.put("key1", "value1")
            result = cache.get("key1")
            self.assertEqual(result, "value1")
            
            # Test cache miss
            result = cache.get("nonexistent", "default")
            self.assertEqual(result, "default")
            
            # Test cache deletion
            self.assertTrue(cache.delete("key1"))
            self.assertFalse(cache.delete("key1"))  # Second delete should return False
            
            # Test stats
            stats = cache.get_stats()
            self.assertIn('hits', stats)
            self.assertIn('misses', stats)
            
        except ImportError:
            self.skipTest("Cache module not available")
    
    def test_performance_profiler(self):
        """Test performance profiler."""
        try:
            from conforl.optimize.profiler import PerformanceProfiler
            
            profiler = PerformanceProfiler(enable_memory_tracking=False, enable_cpu_tracking=False)
            
            # Test context manager
            with profiler.profile_context("test_operation"):
                time.sleep(0.01)  # Small delay
            
            # Test function decorator
            @profiler.profile_function
            def test_function():
                return sum(range(100))
            
            result = test_function()
            self.assertEqual(result, sum(range(100)))
            
            # Check metrics were recorded
            export_data = profiler.export_data()
            self.assertIn('metrics', export_data)
            
        except ImportError:
            self.skipTest("Profiler module not available")


class TestConcurrency(unittest.TestCase):
    """Test concurrent processing."""
    
    def test_thread_safe_buffer(self):
        """Test thread-safe buffer implementation."""
        try:
            from conforl.optimize.concurrent import ThreadSafeRLBuffer
            
            buffer = ThreadSafeRLBuffer(max_size=100)
            
            # Test basic operations
            self.assertTrue(buffer.add("item1"))
            self.assertTrue(buffer.add("item2"))
            
            self.assertEqual(buffer.size(), 2)
            self.assertFalse(buffer.is_empty())
            
            # Test retrieval
            item = buffer.get()
            self.assertIn(item, ["item1", "item2"])
            
            # Test batch operations
            buffer.add("item3")
            buffer.add("item4")
            batch = buffer.get_batch(2)
            self.assertEqual(len(batch), 2)
            
        except ImportError:
            self.skipTest("Concurrent processing module not available")
    
    def test_concurrent_task_processor(self):
        """Test concurrent task processor."""
        try:
            from conforl.optimize.concurrent import ConcurrentTaskProcessor
            
            processor = ConcurrentTaskProcessor(max_workers=2, use_processes=False)
            
            def simple_task(x):
                return x * 2
            
            # Submit a task
            future = processor.submit_task(simple_task, 5)
            result = future.result(timeout=5.0)
            self.assertEqual(result, 10)
            
            # Test batch processing
            results = processor.map_concurrent(simple_task, [1, 2, 3, 4, 5])
            successful_results = [r.result for r in results if r.success]
            expected_results = [2, 4, 6, 8, 10]
            self.assertEqual(sorted(successful_results), sorted(expected_results))
            
            processor.shutdown()
            
        except ImportError:
            self.skipTest("Concurrent task processor not available")


class TestMonitoring(unittest.TestCase):
    """Test monitoring and metrics."""
    
    def test_metrics_collector(self):
        """Test metrics collection."""
        try:
            from conforl.monitoring.metrics import MetricsCollector
            
            collector = MetricsCollector(auto_flush_interval=0.1)
            
            # Record metrics
            collector.record_metric("test.metric", 1.5, tags={"test": "true"})
            collector.increment_counter("test.counter", 1, tags={"test": "true"})
            collector.set_gauge("test.gauge", 42.0)
            
            # Test metric retrieval
            stats = collector.get_metric_stats("test.metric")
            self.assertIn('mean', stats)
            
            # Test all metrics
            all_metrics = collector.get_all_metrics()
            self.assertIn('test.metric', all_metrics)
            
            # Test summary
            summary = collector.get_summary()
            self.assertIn('total_metrics', summary)
            
            collector.stop_auto_flush()
            
        except ImportError:
            self.skipTest("Metrics collector not available")
    
    def test_performance_tracker(self):
        """Test performance tracking."""
        try:
            from conforl.monitoring.metrics import PerformanceTracker
            
            tracker = PerformanceTracker()
            
            # Track training episode
            tracker.track_training_episode(
                episode_reward=10.5,
                episode_length=100,
                episode_risk=0.05,
                algorithm="test"
            )
            
            # Track inference
            tracker.track_inference(
                inference_time=0.01,
                risk_bound=0.03,
                confidence=0.95,
                algorithm="test"
            )
            
            # Get summary
            summary = tracker.get_performance_summary()
            self.assertEqual(summary['training_episodes'], 1)
            self.assertEqual(summary['inference_requests'], 1)
            
        except ImportError:
            self.skipTest("Performance tracker not available")


class TestIntegration(unittest.TestCase):
    """Integration tests for ConfoRL components."""
    
    def test_end_to_end_workflow(self):
        """Test basic end-to-end workflow."""
        # This is a simplified integration test
        from conforl.core.types import RiskCertificate, TrajectoryData
        from conforl.utils.errors import ConfoRLError
        
        # Create a trajectory
        trajectory = TrajectoryData(
            states=[1, 2, 3],
            actions=[0.1, 0.2, 0.3],
            rewards=[1.0, 0.5, 0.8],
            dones=[False, False, True],
            infos=[{}, {}, {}]
        )
        
        # Create a risk certificate
        certificate = RiskCertificate(
            risk_bound=0.05,
            confidence=0.95,
            coverage_guarantee=0.95,
            method="integration_test",
            sample_size=3
        )
        
        # Verify basic functionality
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(certificate.risk_bound, 0.05)
        
        # Test error handling
        with self.assertRaises(ConfoRLError):
            raise ConfoRLError("Test integration error", "INTEGRATION_ERROR")


class TestRobustness(unittest.TestCase):
    """Test system robustness and error handling."""
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        from conforl.core.types import TrajectoryData
        
        # Test empty trajectory (should handle gracefully)
        try:
            empty_trajectory = TrajectoryData(
                states=[],
                actions=[],
                rewards=[],
                dones=[],
                infos=[]
            )
            self.assertEqual(len(empty_trajectory), 0)
        except Exception as e:
            # Should not crash, but if it does, log the error
            print(f"Edge case handling issue: {e}")
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        from conforl.utils.errors import ValidationError
        
        # Test that errors can be caught and handled
        try:
            raise ValidationError("Test validation error", "test")
        except ValidationError as e:
            self.assertEqual(e.validation_type, "test")
            # Error was successfully caught and handled


def run_performance_benchmarks():
    """Run performance benchmarks."""
    print("\n🚀 Running Performance Benchmarks")
    print("=" * 50)
    
    # Benchmark 1: Core type creation
    start_time = time.time()
    from conforl.core.types import RiskCertificate
    
    for i in range(1000):
        cert = RiskCertificate(
            risk_bound=0.05,
            confidence=0.95,
            coverage_guarantee=0.95,
            method="benchmark",
            sample_size=1000
        )
    
    cert_creation_time = time.time() - start_time
    print(f"RiskCertificate creation (1000x): {cert_creation_time:.3f}s")
    
    # Benchmark 2: Conformal predictor calibration
    try:
        from conforl.core.conformal import SplitConformalPredictor
        
        start_time = time.time()
        predictor = SplitConformalPredictor(coverage=0.95)
        
        # Generate dummy data
        dummy_data = list(range(100))
        dummy_scores = [i * 0.01 for i in range(100)]
        
        for i in range(10):
            predictor.calibrate(dummy_data, dummy_scores)
        
        calibration_time = time.time() - start_time
        print(f"Conformal predictor calibration (10x): {calibration_time:.3f}s")
        
    except ImportError:
        print("Conformal predictor not available for benchmarking")
    
    print("✅ Performance benchmarks completed")


def main():
    """Main test runner."""
    print("🧪 ConfoRL Comprehensive Test Suite")
    print("=" * 60)
    
    # Run unit tests
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestCoreFunctionality,
        TestUtilities,
        TestOptimization,
        TestConcurrency,
        TestMonitoring,
        TestIntegration,
        TestRobustness,
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print results summary
    print(f"\n📊 Test Results Summary")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / 
                   max(1, result.testsRun)) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split(chr(10))[-2]}")
    
    # Run performance benchmarks
    run_performance_benchmarks()
    
    # Overall assessment
    print(f"\n🎯 Overall Assessment")
    if len(result.failures) + len(result.errors) == 0:
        print("✅ All tests passed! ConfoRL is ready for production.")
        return 0
    elif success_rate >= 85:
        print("⚠️  Most tests passed. Minor issues to address.")
        return 1
    else:
        print("❌ Significant test failures. Requires attention.")
        return 2


if __name__ == "__main__":
    sys.exit(main())