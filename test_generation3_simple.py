#!/usr/bin/env python3
"""
Generation 3 Simple Tests
Quick validation of performance and scaling features.
"""

import sys
import os
import time

# Add the repo to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_performance_cache():
    """Test performance optimization cache."""
    print("🧪 Testing Performance Cache...")
    
    try:
        from conforl.optimize.performance_engine import AdaptiveCache
        
        # Test basic cache operations
        cache = AdaptiveCache(max_size=10)
        
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"
        
        cache.put("key2", "value2")
        stats = cache.get_stats()
        assert stats['size'] == 2
        assert stats['hit_rate'] > 0
        
        print("   ✅ Adaptive cache basic operations work")
        
        # Test cache eviction
        for i in range(15):  # More than max_size
            cache.put(f"key_{i}", f"value_{i}")
        
        final_stats = cache.get_stats()
        assert final_stats['size'] <= 10
        assert final_stats['evictions'] > 0
        
        print("   ✅ Cache eviction working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_performance_profiler():
    """Test performance profiling."""
    print("🧪 Testing Performance Profiler...")
    
    try:
        from conforl.optimize.performance_engine import PerformanceProfiler
        
        profiler = PerformanceProfiler()
        
        # Start operation
        op_id = profiler.start_operation("test_op")
        time.sleep(0.01)  # Simulate work
        
        # Stop operation
        metrics = profiler.stop_operation(op_id)
        
        assert metrics.operation_name == "test_op"
        assert metrics.execution_time > 0
        
        print("   ✅ Performance profiling working")
        
        # Test operation stats
        stats = profiler.get_operation_stats("test_op")
        assert stats is not None
        assert stats['total_calls'] == 1
        
        print("   ✅ Performance stats working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_concurrent_processor():
    """Test concurrent processing."""
    print("🧪 Testing Concurrent Processor...")
    
    try:
        from conforl.optimize.performance_engine import ConcurrentProcessor
        
        processor = ConcurrentProcessor(max_workers=2)
        
        # Test concurrent map
        def square(x):
            return x * x
        
        items = [1, 2, 3, 4, 5]
        results = processor.map_concurrent(square, items)
        
        assert results == [1, 4, 9, 16, 25]
        print("   ✅ Concurrent mapping working")
        
        # Test batch processing
        batch_results = processor.batch_process(square, items, batch_size=2)
        assert len(batch_results) == 5
        print("   ✅ Batch processing working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_auto_scaler_basic():
    """Test basic auto-scaling functionality."""
    print("🧪 Testing Auto-Scaler Basic...")
    
    try:
        from conforl.scaling.auto_scaler import (
            LoadBalancer, WorkerNode, LoadBalancingStrategy
        )
        
        # Test load balancer
        lb = LoadBalancer(LoadBalancingStrategy.ROUND_ROBIN)
        
        # Add workers
        worker1 = WorkerNode("worker1", capacity=100)
        worker2 = WorkerNode("worker2", capacity=100)
        
        lb.add_worker(worker1)
        lb.add_worker(worker2)
        
        # Test worker selection
        selected = lb.get_next_worker()
        assert selected is not None
        
        print("   ✅ Load balancer working")
        
        # Test stats
        stats = lb.get_load_balancer_stats()
        assert stats['total_workers'] == 2
        
        print("   ✅ Load balancer stats working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_optimization_decorators():
    """Test optimization decorators."""
    print("🧪 Testing Optimization Decorators...")
    
    try:
        from conforl.optimize.performance_engine import cached, profile_performance
        
        # Test caching decorator
        call_count = 0
        
        @cached(ttl=60)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        result1 = expensive_function(5)
        result2 = expensive_function(5)  # Should be cached
        
        assert result1 == result2 == 10
        assert call_count == 1  # Only called once due to caching
        
        print("   ✅ Caching decorator working")
        
        # Test profiling decorator
        @profile_performance("decorated_function")
        def profiled_function(x):
            time.sleep(0.001)
            return x + 1
        
        result = profiled_function(10)
        assert result == 11
        
        print("   ✅ Profiling decorator working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_integrated_scaling():
    """Test integrated scaling with ConfoRL."""
    print("🧪 Testing Integrated Scaling...")
    
    try:
        import conforl
        from conforl.optimize.performance_engine import optimize_function
        
        # Create optimized ConfoRL function
        @optimize_function
        def optimized_risk_computation(target_risk, confidence):
            controller = conforl.AdaptiveRiskController(
                target_risk=target_risk,
                confidence=confidence
            )
            return controller.get_risk_bound()
        
        # Test optimized function
        risk1 = optimized_risk_computation(0.05, 0.95)
        risk2 = optimized_risk_computation(0.05, 0.95)  # Should be cached
        
        assert isinstance(risk1, (int, float))
        assert risk1 == risk2
        
        print("   ✅ Integrated optimization working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    """Run simple Generation 3 tests."""
    print("=" * 50)
    print("⚡ ConfoRL Generation 3 - Simple Tests")
    print("=" * 50)
    
    tests = [
        test_performance_cache,
        test_performance_profiler,
        test_concurrent_processor,
        test_auto_scaler_basic,
        test_optimization_decorators,
        test_integrated_scaling
    ]
    
    passed = 0
    total = len(tests)
    
    for i, test in enumerate(tests, 1):
        print(f"\n[{i}/{total}] Running {test.__name__}...")
        try:
            if test():
                passed += 1
                print(f"   🟢 PASSED")
            else:
                print(f"   🔴 FAILED")
        except Exception as e:
            print(f"   🔴 FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 3 TESTS PASSED!")
        print("⚡ Performance optimization working")
        print("📈 Auto-scaling mechanisms functional")
        print("🚀 Concurrent processing operational")
        print("💾 Caching and profiling active")
        print("🌟 ConfoRL is PRODUCTION READY!")
    else:
        print(f"⚠️  {total - passed} tests failed.")
        return False
    
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)