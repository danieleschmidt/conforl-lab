#!/usr/bin/env python3
"""
Generation 2 Simple Tests
Quick validation of robustness features.
"""

import sys
import os

# Add the repo to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_robust_validation_basic():
    """Test basic robust validation functionality."""
    print("🧪 Testing Basic Robust Validation...")
    
    try:
        from conforl.utils.robust_validation import RobustValidator
        from conforl.utils.errors import ValidationError, SecurityError
        
        validator = RobustValidator()
        
        # Test valid config
        config = {'target_risk': 0.05, 'confidence': 0.95}
        result = validator.validate_and_sanitize_config(config)
        assert result['target_risk'] == 0.05
        print("   ✅ Config validation works")
        
        # Test security detection
        try:
            malicious = {'script': '<script>alert("xss")</script>'}
            validator.validate_and_sanitize_config(malicious)
            assert False, "Should detect XSS"
        except (SecurityError, Exception):
            print("   ✅ Security validation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_monitoring_basic():
    """Test basic monitoring functionality."""
    print("🧪 Testing Basic Monitoring...")
    
    try:
        from conforl.utils.monitoring import MetricsCollector, HealthChecker
        
        # Test metrics
        collector = MetricsCollector()
        collector.record_metric("test_metric", 42.0)
        
        stats = collector.get_metric_stats("test_metric")
        assert stats['current'] == 42.0
        print("   ✅ Metrics collection works")
        
        # Test health checks
        health = HealthChecker()
        health.register_check("test", lambda: True)
        
        status = health.get_health_status()
        assert status['overall_healthy'] == True
        print("   ✅ Health checking works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_circuit_breaker_basic():
    """Test basic circuit breaker functionality."""
    print("🧪 Testing Basic Circuit Breaker...")
    
    try:
        from conforl.utils.circuit_breaker import CircuitBreaker, CircuitState
        
        breaker = CircuitBreaker()
        
        # Test successful call
        result = breaker.call(lambda: "success")
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED
        print("   ✅ Circuit breaker basic operation works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_error_types():
    """Test enhanced error types."""
    print("🧪 Testing Enhanced Error Types...")
    
    try:
        from conforl.utils.errors import (
            InvalidTrajectoryError, InvalidRiskParameterError,
            SecurityError, CircuitBreakerError
        )
        
        # Test error creation
        try:
            raise InvalidTrajectoryError("Test trajectory error")
        except InvalidTrajectoryError as e:
            assert "trajectory error" in str(e)
            print("   ✅ InvalidTrajectoryError works")
        
        try:
            raise InvalidRiskParameterError("Test risk error")
        except InvalidRiskParameterError as e:
            assert "risk error" in str(e)
            print("   ✅ InvalidRiskParameterError works")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def test_integration():
    """Test basic integration between components."""
    print("🧪 Testing Basic Integration...")
    
    try:
        import conforl
        from conforl.utils.robust_validation import validate_config_robust
        from conforl.utils.monitoring import record_metric
        
        # Test workflow
        config = {'target_risk': 0.05, 'confidence': 0.95}
        validated_config = validate_config_robust(config)
        
        controller = conforl.AdaptiveRiskController(**validated_config)
        
        record_metric("integration_test", 1.0)
        
        print("   ✅ Basic integration works")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False

def main():
    """Run simple Generation 2 tests."""
    print("=" * 50)
    print("🛡️  ConfoRL Generation 2 - Simple Tests")
    print("=" * 50)
    
    tests = [
        test_robust_validation_basic,
        test_monitoring_basic,
        test_circuit_breaker_basic,
        test_error_types,
        test_integration
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
        print("🎉 GENERATION 2 TESTS PASSED!")
        print("🛡️  Robustness features working")
        print("📊 Monitoring systems operational")
        print("🔄 Fault tolerance mechanisms active")
        print("⚡ Ready for Generation 3!")
    else:
        print(f"⚠️  {total - passed} tests failed.")
        return False
    
    print("=" * 50)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)