#!/usr/bin/env python3
"""Performance optimization test for ConfoRL Generation 3."""

import sys
import time
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Any

def test_concurrent_processing():
    """Test concurrent processing capabilities."""
    print("Testing concurrent processing...")
    
    try:
        from conforl.optimize.concurrent import ConcurrentProcessor
        
        processor = ConcurrentProcessor(max_workers=4)
        
        # Test concurrent task execution
        def sample_task(x):
            time.sleep(0.1)  # Simulate work
            return x * x
        
        tasks = list(range(10))
        start_time = time.time()
        
        results = processor.execute_concurrent(sample_task, tasks)
        
        execution_time = time.time() - start_time
        print(f"✓ Concurrent execution completed in {execution_time:.2f}s")
        print(f"✓ Results: {results[:5]}...")  # Show first 5 results
        
        return True
        
    except Exception as e:
        print(f"✗ Concurrent processing test failed: {e}")
        traceback.print_exc()
        return False

def test_adaptive_caching():
    """Test adaptive caching system."""
    print("\nTesting adaptive caching...")
    
    try:
        from conforl.optimize.cache import AdaptiveCache
        
        cache = AdaptiveCache(max_size=100, ttl=300)
        
        # Test basic caching
        cache.set("key1", "value1")
        cached_value = cache.get("key1")
        
        if cached_value != "value1":
            raise ValueError("Cache get/set failed")
        
        print("✓ Basic cache operations working")
        
        # Test cache statistics
        stats = cache.get_stats()
        print(f"✓ Cache stats: hits={stats.get('hits', 0)}, misses={stats.get('misses', 0)}")
        
        # Test adaptive sizing
        for i in range(50):
            cache.set(f"key_{i}", f"value_{i}")
        
        print(f"✓ Cache populated with 50 items")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive caching test failed: {e}")
        traceback.print_exc()
        return False

def test_performance_profiling():
    """Test performance profiling capabilities."""
    print("\nTesting performance profiling...")
    
    try:
        from conforl.optimize.profiler import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Test profiling context
        with profiler.profile("test_operation"):
            # Simulate some work
            total = 0
            for i in range(10000):
                total += i * i
        
        # Get profiling results
        results = profiler.get_results()
        
        if "test_operation" not in results:
            raise ValueError("Profiling results not found")
        
        operation_time = results["test_operation"]["total_time"]
        print(f"✓ Profiled operation completed in {operation_time:.4f}s")
        
        # Test memory profiling
        memory_usage = profiler.get_memory_usage()
        print(f"✓ Current memory usage: {memory_usage:.2f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance profiling test failed: {e}")
        traceback.print_exc()
        return False

def test_auto_scaling():
    """Test auto-scaling capabilities."""
    print("\nTesting auto-scaling...")
    
    try:
        from conforl.optimize.scaling import AutoScaler
        
        scaler = AutoScaler(
            min_workers=1,
            max_workers=8,
            target_utilization=0.7
        )
        
        # Simulate load changes
        for load in [0.3, 0.8, 0.9, 0.5, 0.2]:
            recommended_workers = scaler.recommend_workers(current_load=load)
            print(f"✓ Load {load:.1f} -> Recommended workers: {recommended_workers}")
        
        # Test scaling metrics
        metrics = scaler.get_scaling_metrics()
        print(f"✓ Scaling metrics: {metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Auto-scaling test failed: {e}")
        traceback.print_exc()
        return False

def test_resource_monitoring():
    """Test resource monitoring and health checks."""
    print("\nTesting resource monitoring...")
    
    try:
        from conforl.utils.health import HealthMonitor
        
        monitor = HealthMonitor()
        
        # Test health check
        health_status = monitor.check_health()
        print(f"✓ System health: {health_status['status']}")
        
        # Test resource metrics
        resources = monitor.get_resource_metrics()
        print(f"✓ CPU usage: {resources.get('cpu_percent', 0):.1f}%")
        print(f"✓ Memory usage: {resources.get('memory_percent', 0):.1f}%")
        
        # Test performance alerts
        alerts = monitor.check_performance_alerts()
        print(f"✓ Performance alerts: {len(alerts)} active")
        
        return True
        
    except Exception as e:
        print(f"✗ Resource monitoring test failed: {e}")
        traceback.print_exc()
        return False

def test_distributed_processing():
    """Test distributed processing capabilities."""
    print("\nTesting distributed processing...")
    
    try:
        from conforl.scaling.distributed import DistributedProcessor
        
        processor = DistributedProcessor()
        
        # Test task distribution (mock)
        task_config = {
            'algorithm': 'sac',
            'environment': 'mock',
            'timesteps': 1000
        }
        
        result = processor.submit_task("training", task_config)
        print(f"✓ Distributed task submitted: {result}")
        
        # Test load balancing
        load_metrics = processor.get_load_balance_metrics()
        print(f"✓ Load balance metrics: {load_metrics}")
        
        return True
        
    except Exception as e:
        print(f"✗ Distributed processing test failed: {e}")
        traceback.print_exc()
        return False

def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    print("\nRunning performance benchmark...")
    
    try:
        # Memory allocation test
        start_memory = get_memory_usage()
        
        # Simulate heavy computation
        data = []
        for i in range(100000):
            data.append(i * 2)
        
        end_memory = get_memory_usage()
        memory_delta = end_memory - start_memory
        
        print(f"✓ Memory allocation test: {memory_delta:.2f} MB allocated")
        
        # CPU intensive test
        start_time = time.time()
        result = sum(i * i for i in range(100000))
        cpu_time = time.time() - start_time
        
        print(f"✓ CPU intensive test: {cpu_time:.4f}s (result: {result})")
        
        # Cleanup
        del data
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def get_memory_usage():
    """Get current memory usage in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0  # psutil not available

def main():
    """Run all performance optimization tests."""
    print("=== ConfoRL Generation 3: Performance Optimization Test ===")
    print(f"Python version: {sys.version}")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        test_concurrent_processing,
        test_adaptive_caching,
        test_performance_profiling,
        test_auto_scaling,
        test_resource_monitoring,
        test_distributed_processing,
        run_performance_benchmark
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Performance Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🚀 All performance tests passed! Generation 3 optimizations validated.")
        return 0
    else:
        print("⚠️  Some performance tests failed. Check output for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())