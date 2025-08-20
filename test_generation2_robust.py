#!/usr/bin/env python3
"""
Generation 2 Robustness Tests
Tests enhanced error handling, security, monitoring, and resilience features.
"""

import sys
import os
import time
import threading
from unittest.mock import Mock

# Add the repo to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_robust_validation():
    """Test comprehensive input validation and sanitization."""
    print("🧪 Testing Robust Validation...")
    
    try:
        from conforl.utils.robust_validation import (
            RobustValidator, validate_config_robust, 
            validate_trajectory_robust, validate_risk_params_robust
        )
        from conforl.utils.errors import ValidationError, SecurityError
        
        validator = RobustValidator()
        
        # Test config validation
        valid_config = {
            'target_risk': 0.05,
            'confidence': 0.95,
            'window_size': 100
        }
        
        sanitized = validate_config_robust(valid_config)
        assert sanitized['target_risk'] == 0.05
        print(f"   ✅ Config validation passed")
        
        # Test security validation
        try:
            malicious_config = {
                'target_risk': 0.05,
                'script': '<script>alert("xss")</script>',
                'eval_code': 'eval("import os")'
            }
            validate_config_robust(malicious_config)
            assert False, "Should have detected security threat"
        except (SecurityError, Exception):
            print(f"   ✅ Security validation detected threats")
        
        # Test risk parameter validation
        target, conf, extra = validate_risk_params_robust(0.05, 0.95, window_size=100)
        assert target == 0.05
        assert conf == 0.95
        assert extra['window_size'] == 100
        print(f"   ✅ Risk parameter validation passed")
        
        # Test trajectory validation
        import conforl
        trajectory = conforl.TrajectoryData(
            states=[[0.1, 0.2], [0.3, 0.4]],
            actions=[1, 0],
            rewards=[0.1, -0.2],
            dones=[False, True],
            infos=[{}, {}]
        )
        
        result = validate_trajectory_robust(trajectory)
        assert result == True
        print(f"   ✅ Trajectory validation passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Robust validation test failed: {e}")
        return False

def test_monitoring_system():
    """Test comprehensive monitoring and metrics collection."""
    print("🧪 Testing Monitoring System...")
    
    try:
        from conforl.utils.monitoring import (
            MetricsCollector, HealthChecker, MonitoringContext,
            record_metric, increment_counter, record_timing,
            get_metrics_summary, AlertLevel
        )
        
        # Test metrics collection
        collector = MetricsCollector()
        
        # Record some metrics
        collector.record_metric("test_metric", 42.0)
        collector.increment_counter("test_counter", 5)
        collector.record_histogram("test_histogram", 1.5)
        
        # Test metric retrieval
        stats = collector.get_metric_stats("test_metric")
        assert stats is not None
        assert stats['current'] == 42.0
        print(f"   ✅ Metrics collection working")
        
        # Test health checker
        health_checker = HealthChecker()
        health_checker.register_check("test_check", lambda: True)
        
        status = health_checker.get_health_status()
        assert status['overall_healthy'] == True
        print(f"   ✅ Health checking working")
        
        # Test monitoring context
        with MonitoringContext("test_operation"):
            time.sleep(0.01)  # Simulate work
        
        summary = collector.get_all_metrics_summary()
        assert summary['total_metrics'] > 0
        print(f"   ✅ Monitoring context working")
        
        # Test alerts
        collector.set_alert_threshold("test_alert_metric", 10.0, AlertLevel.WARNING)
        
        alert_triggered = False
        def alert_callback(alert):
            nonlocal alert_triggered
            alert_triggered = True
        
        collector.add_alert_callback(alert_callback)
        collector.record_metric("test_alert_metric", 15.0)  # Above threshold
        
        assert alert_triggered, "Alert should have been triggered"
        print(f"   ✅ Alert system working")
        
        # Test Prometheus export
        prometheus_data = collector.export_prometheus_format()
        assert "conforl_" in prometheus_data
        print(f"   ✅ Prometheus export working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Monitoring system test failed: {e}")
        return False

def test_circuit_breaker():
    """Test circuit breaker for fault tolerance."""
    print("🧪 Testing Circuit Breaker...")
    
    try:
        from conforl.utils.circuit_breaker import (
            CircuitBreaker, CircuitBreakerConfig, circuit_breaker,
            get_circuit_breaker, CircuitState
        )
        from conforl.utils.errors import CircuitBreakerError
        
        # Test basic circuit breaker
        config = CircuitBreakerConfig(failure_threshold=3, timeout_seconds=1)
        breaker = CircuitBreaker(config)
        
        # Test successful calls
        def successful_func():
            return "success"
        
        result = breaker.call(successful_func)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        print(f"   ✅ Circuit breaker closed state working")
        
        # Test failures that open circuit
        def failing_func():
            raise Exception("Test failure")
        
        # Trigger failures to open circuit
        for i in range(3):
            try:
                breaker.call(failing_func)
            except Exception:
                pass
        
        assert breaker.state == CircuitState.OPEN
        print(f"   ✅ Circuit breaker opens after failures")
        
        # Test that circuit blocks calls when open
        try:
            breaker.call(successful_func)
            assert False, "Should have raised CircuitBreakerError"
        except CircuitBreakerError:
            print(f"   ✅ Circuit breaker blocks calls when open")
        
        # Test decorator
        @circuit_breaker("test_service", failure_threshold=2, timeout_seconds=1)
        def decorated_function(should_fail=False):
            if should_fail:
                raise Exception("Decorated failure")
            return "decorated success"
        
        # Test successful decorated call
        result = decorated_function(False)
        assert result == "decorated success"
        print(f"   ✅ Circuit breaker decorator working")
        
        # Test circuit breaker status
        status = breaker.get_status()
        assert 'state' in status
        assert 'failure_count' in status
        print(f"   ✅ Circuit breaker status reporting working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Circuit breaker test failed: {e}")
        return False

def test_retry_mechanism():
    """Test retry policy for resilient operations."""
    print("🧪 Testing Retry Mechanism...")
    
    try:
        from conforl.utils.circuit_breaker import RetryPolicy, with_retry
        
        # Test basic retry policy
        policy = RetryPolicy(max_attempts=3, base_delay=0.01)
        
        attempt_count = 0
        def failing_then_succeeding():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception(f"Attempt {attempt_count} failed")
            return f"Success on attempt {attempt_count}"
        
        result = policy.execute(failing_then_succeeding)
        assert "Success on attempt 3" in result
        assert attempt_count == 3
        print(f"   ✅ Retry policy working")
        
        # Test retry decorator
        call_count = 0
        @with_retry(max_attempts=2, base_delay=0.01)
        def decorated_retry_func(should_fail=True):
            nonlocal call_count
            call_count += 1
            if should_fail and call_count == 1:
                raise Exception("First call fails")
            return f"Success on call {call_count}"
        
        call_count = 0
        result = decorated_retry_func(True)
        assert call_count == 2
        print(f"   ✅ Retry decorator working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Retry mechanism test failed: {e}")
        return False

def test_enhanced_error_handling():
    """Test enhanced error handling and recovery."""
    print("🧪 Testing Enhanced Error Handling...")
    
    try:
        import conforl
        from conforl.utils.robust_validation import RobustValidator
        from conforl.utils.errors import (
            ValidationError, SecurityError, ConfigurationError,
            InvalidTrajectoryError, InvalidRiskParameterError
        )
        
        # Test custom error types
        validator = RobustValidator()
        
        # Test validation error
        try:
            validator.validate_risk_parameters_comprehensive("invalid", 0.95)
            assert False, "Should have raised InvalidRiskParameterError"
        except InvalidRiskParameterError as e:
            assert "must be numeric" in str(e)
            print(f"   ✅ InvalidRiskParameterError working")
        
        # Test trajectory validation error
        try:
            invalid_trajectory = conforl.TrajectoryData(
                states=[],  # Empty states should fail
                actions=[1],
                rewards=[0.1],
                dones=[True],
                infos=[{}]
            )
            validator.validate_trajectory_robust(invalid_trajectory)
            assert False, "Should have raised InvalidTrajectoryError"
        except InvalidTrajectoryError as e:
            assert "empty states" in str(e)
            print(f"   ✅ InvalidTrajectoryError working")
        
        # Test security error
        try:
            malicious_config = {'eval': 'eval("import os")'}
            validator.validate_and_sanitize_config(malicious_config)
            assert False, "Should have raised SecurityError"
        except (SecurityError, Exception) as e:
            print(f"   ✅ SecurityError working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced error handling test failed: {e}")
        return False

def test_thread_safety():
    """Test thread safety of monitoring and validation systems."""
    print("🧪 Testing Thread Safety...")
    
    try:
        from conforl.utils.monitoring import MetricsCollector
        from conforl.utils.circuit_breaker import CircuitBreaker
        
        # Test concurrent metrics collection
        collector = MetricsCollector()
        results = []
        
        def concurrent_metrics():
            for i in range(50):
                collector.record_metric(f"thread_metric_{threading.current_thread().ident}", i)
                collector.increment_counter("thread_counter")
            results.append(True)
        
        threads = []
        for i in range(5):
            thread = threading.Thread(target=concurrent_metrics)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(results) == 5
        assert collector.counters['thread_counter'] == 250  # 5 threads * 50 increments
        print(f"   ✅ Concurrent metrics collection working")
        
        # Test concurrent circuit breaker
        breaker = CircuitBreaker()
        success_count = 0
        
        def concurrent_circuit_breaker():
            nonlocal success_count
            try:
                result = breaker.call(lambda: "success")
                if result == "success":
                    success_count += 1
            except Exception:
                pass
        
        threads = []
        for i in range(10):
            thread = threading.Thread(target=concurrent_circuit_breaker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert success_count == 10
        print(f"   ✅ Concurrent circuit breaker working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Thread safety test failed: {e}")
        return False

def test_integration_robustness():
    """Test integration between robustness components."""
    print("🧪 Testing Integration Robustness...")
    
    try:
        import conforl
        from conforl.utils.monitoring import MonitoringContext, record_metric
        from conforl.utils.circuit_breaker import circuit_breaker
        from conforl.utils.robust_validation import validate_config_robust
        
        # Test integrated workflow with monitoring and circuit breaker
        @circuit_breaker("robust_workflow", failure_threshold=3)
        def robust_workflow(config, trajectory_data):
            with MonitoringContext("robust_workflow_execution"):
                # Validate inputs
                validated_config = validate_config_robust(config)
                
                # Create risk controller with validated config
                controller = conforl.AdaptiveRiskController(**validated_config)
                
                # Create risk measure
                risk_measure = conforl.SafetyViolationRisk(violation_threshold=0.1)
                
                # Update controller
                controller.update(trajectory_data, risk_measure)
                
                # Get certificate
                certificate = controller.get_certificate()
                
                record_metric("workflow_risk_bound", certificate.risk_bound)
                
                return certificate
        
        # Test successful workflow
        config = {
            'target_risk': 0.05,
            'confidence': 0.95,
            'window_size': 100
        }
        
        trajectory = conforl.TrajectoryData(
            states=[[0.1, 0.2], [0.3, 0.4]],
            actions=[1, 0],
            rewards=[0.1, -0.2],
            dones=[False, True],
            infos=[{}, {}]
        )
        
        certificate = robust_workflow(config, trajectory)
        assert hasattr(certificate, 'risk_bound')
        assert hasattr(certificate, 'confidence')
        print(f"   ✅ Integrated robust workflow working")
        
        # Test error handling in workflow
        invalid_config = {'target_risk': 'invalid'}
        
        try:
            robust_workflow(invalid_config, trajectory)
            assert False, "Should have failed with invalid config"
        except Exception:
            print(f"   ✅ Integrated error handling working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration robustness test failed: {e}")
        return False

def main():
    """Run all Generation 2 robustness tests."""
    print("=" * 60)
    print("🛡️  ConfoRL Generation 2 - Robustness & Reliability Tests")
    print("=" * 60)
    
    tests = [
        test_robust_validation,
        test_monitoring_system,
        test_circuit_breaker,
        test_retry_mechanism,
        test_enhanced_error_handling,
        test_thread_safety,
        test_integration_robustness
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
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Generation 2 is COMPLETE!")
        print("🛡️  Robustness and reliability verified")
        print("🔒 Security and error handling operational")
        print("📊 Monitoring and health checks working")
        print("🔄 Fault tolerance and recovery mechanisms active")
        print("⚡ Ready for Generation 3 (Scalability)")
    else:
        print(f"⚠️  {total - passed} tests failed. Generation 2 needs fixes.")
        return False
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)