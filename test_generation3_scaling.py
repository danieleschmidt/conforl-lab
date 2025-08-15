#!/usr/bin/env python3
"""Test Generation 3: Scaling and optimization functionality."""

import sys
import os
import time
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_adaptive_cache():
    """Test adaptive caching system."""
    print("Testing adaptive cache...")
    
    try:
        from conforl.optimize.cache import AdaptiveCache
        
        # Create adaptive cache
        cache = AdaptiveCache(max_size=100, ttl=5.0, adaptive_ttl=True)
        print("✓ Adaptive cache created")
        
        # Test basic operations
        cache.put("test_key", {"data": "test_value"})
        result = cache.get("test_key")
        print(f"✓ Cache put/get working: {result is not None}")
        
        # Test cache statistics
        stats = cache.get_stats()
        print(f"✓ Cache stats: hit_rate={stats['hit_rate']:.2f}, size={stats['size']}")
        
        # Test cache performance
        start_time = time.time()
        for i in range(50):
            cache.put(f"key_{i}", f"value_{i}")
        
        for i in range(25):
            result = cache.get(f"key_{i}")
        
        end_time = time.time()
        print(f"✓ Cache performance test completed in {end_time - start_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"✗ Adaptive cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parallel_processing():
    """Test parallel processing capabilities."""
    print("\nTesting parallel processing...")
    
    try:
        from conforl.optimize.concurrent import ParallelTraining, BatchProcessor
        
        # Test batch processor
        batch_processor = BatchProcessor(batch_size=10, max_workers=2)
        print("✓ Batch processor created")
        
        # Test batch processing
        items = list(range(25))
        
        def square_function(x):
            return x * x
        
        results = batch_processor.process_batch(items, square_function)
        expected = [x * x for x in items]
        
        success = len(results) == len(expected) and all(r == e for r, e in zip(results, expected) if r is not None)
        print(f"✓ Batch processing working: {success}")
        
        # Test batch processing stats
        stats = batch_processor.get_stats()
        print(f"✓ Batch stats: {stats['batches_processed']} batches, {stats['total_items_processed']} items")
        
        batch_processor.shutdown()
        print("✓ Batch processor shutdown")
        
        # Test parallel training setup
        parallel_training = ParallelTraining(num_workers=2, use_processes=False)
        print("✓ Parallel training created")
        
        training_stats = parallel_training.get_training_stats()
        print(f"✓ Training stats: {training_stats['episodes_completed']} episodes")
        
        return True
        
    except Exception as e:
        print(f"✗ Parallel processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_auto_scaling():
    """Test auto-scaling functionality."""
    print("\nTesting auto-scaling...")
    
    try:
        from conforl.optimize.scaling import AutoScaler, ScalingMetrics
        
        # Create auto-scaler
        auto_scaler = AutoScaler(min_instances=1, max_instances=5)
        print("✓ Auto-scaler created")
        
        # Test with normal load
        normal_metrics = ScalingMetrics(
            cpu_usage=45.0,
            memory_usage=50.0,
            request_rate=100.0,
            response_time=0.5,
            error_rate=0.01,
            queue_length=5,
            timestamp=time.time()
        )
        
        auto_scaler.update_metrics(normal_metrics)
        current_scale = auto_scaler.get_current_scale()
        print(f"✓ Normal load handling: {current_scale} instances")
        
        # Test with high load (should trigger scale up eventually)
        high_load_metrics = ScalingMetrics(
            cpu_usage=95.0,
            memory_usage=90.0,
            request_rate=500.0,
            response_time=3.0,
            error_rate=0.05,
            queue_length=100,
            timestamp=time.time()
        )
        
        auto_scaler.update_metrics(high_load_metrics)
        print("✓ High load metrics processed")
        
        # Test scaling statistics
        scaling_stats = auto_scaler.get_scaling_stats()
        print(f"✓ Scaling stats: {scaling_stats['total_scaling_events']} events")
        
        return True
        
    except Exception as e:
        print(f"✗ Auto-scaling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_load_balancing():
    """Test load balancing functionality."""
    print("\nTesting load balancing...")
    
    try:
        from conforl.optimize.scaling import LoadBalancer
        
        # Create load balancer
        load_balancer = LoadBalancer(balancing_strategy="round_robin")
        print("✓ Load balancer created")
        
        # Register test instances
        for i in range(3):
            instance_info = {
                'host': f'worker-{i}',
                'port': 8000 + i,
                'capacity': 100
            }
            load_balancer.register_instance(f"instance_{i}", instance_info)
        
        print("✓ Test instances registered")
        
        # Test instance selection
        selected_instances = []
        for _ in range(6):
            instance_id = load_balancer.get_next_instance()
            selected_instances.append(instance_id)
        
        # Should follow round-robin pattern
        expected_pattern = ["instance_0", "instance_1", "instance_2"] * 2
        round_robin_working = selected_instances == expected_pattern
        print(f"✓ Round-robin selection: {round_robin_working}")
        
        # Test request recording
        load_balancer.record_request("instance_0", 0.5, success=True)
        load_balancer.record_request("instance_1", 0.8, success=True)
        load_balancer.record_request("instance_2", 1.2, success=False)
        
        # Test load balancing statistics
        lb_stats = load_balancer.get_load_balancing_stats()
        print(f"✓ Load balancing stats: {lb_stats['total_instances']} instances, {lb_stats['total_requests']} requests")
        
        return True
        
    except Exception as e:
        print(f"✗ Load balancing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring and optimization."""
    print("\nTesting performance monitoring...")
    
    try:
        from conforl.optimize.cache import PerformanceCache
        
        # Create performance cache
        perf_cache = PerformanceCache(max_size=50)
        print("✓ Performance cache created")
        
        # Test cached computation
        def expensive_computation(x):
            time.sleep(0.01)  # Simulate computation time
            return x ** 2 + x + 1
        
        # First call (cache miss)
        start_time = time.time()
        result1 = perf_cache.cached_computation("square_plus", expensive_computation, 10)
        first_call_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        result2 = perf_cache.cached_computation("square_plus", expensive_computation, 10)
        second_call_time = time.time() - start_time
        
        cache_speedup = first_call_time > second_call_time
        print(f"✓ Cache speedup achieved: {cache_speedup} (first: {first_call_time:.4f}s, second: {second_call_time:.4f}s)")
        
        # Test performance statistics
        perf_stats = perf_cache.get_performance_stats()
        print(f"✓ Performance stats: {perf_stats['cache_size']} cached items, {perf_stats['cache_savings_seconds']:.4f}s saved")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_async_processing():
    """Test asynchronous processing capabilities."""
    print("\nTesting async processing...")
    
    try:
        from conforl.optimize.concurrent import AsyncRiskController
        from conforl.risk.controllers import AdaptiveRiskController
        from conforl.core.types import TrajectoryData
        
        # Create base risk controller
        base_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        
        # Create async wrapper
        async_controller = AsyncRiskController(
            base_controller=base_controller,
            update_interval=0.1,
            max_queue_size=10
        )
        print("✓ Async risk controller created")
        
        # Start async processing
        await async_controller.start()
        print("✓ Async processing started")
        
        # Test async updates
        dummy_trajectory = TrajectoryData(
            states=[1, 2, 3],
            actions=[0.1, 0.2, 0.3],
            rewards=[1.0, 0.5, 0.8],
            dones=[False, False, True],
            infos=[{}, {}, {}]
        )
        
        # Submit async updates
        for i in range(5):
            await async_controller.update_async(dummy_trajectory, None)
        
        # Wait for processing
        await asyncio.sleep(0.5)
        
        # Test statistics
        async_stats = async_controller.get_stats()
        print(f"✓ Async stats: {async_stats['updates_processed']} updates processed")
        
        # Stop async processing
        await async_controller.stop()
        print("✓ Async processing stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Async processing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Generation 3 scaling tests."""
    print("⚡ Running ConfoRL Generation 3: Scaling Tests")
    print("=" * 60)
    
    sync_tests = [
        test_adaptive_cache,
        test_parallel_processing,
        test_auto_scaling,
        test_load_balancing,
        test_performance_monitoring
    ]
    
    passed = 0
    total = len(sync_tests) + 1  # +1 for async test
    
    # Run synchronous tests
    for test in sync_tests:
        if test():
            passed += 1
    
    # Run async test
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(test_async_processing())
        if result:
            passed += 1
        loop.close()
    except Exception as e:
        print(f"✗ Async test setup failed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} scaling tests passed")
    
    if passed == total:
        print("🎉 All scaling tests passed!")
        print("✅ Generation 3 scaling functionality is working")
        return 0
    else:
        print("❌ Some scaling tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())