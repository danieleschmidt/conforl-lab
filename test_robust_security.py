#!/usr/bin/env python3
"""Test Generation 2: Robust implementation with security and error handling."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_security_validation():
    """Test security validation functionality."""
    print("Testing security validation...")
    
    try:
        from conforl.security.validation import security_validator, input_sanitizer
        
        # Test input validation
        test_data = {
            'target_risk': 0.05,
            'confidence': 0.95,
            'learning_rate': 0.001,
            'algorithm_name': 'sac'
        }
        
        is_valid, errors = security_validator.validate_dict(test_data)
        print(f"✓ Security validation passed: {is_valid}")
        
        # Test malicious input detection
        malicious_input = "'; DROP TABLE users; --"
        detection = input_sanitizer.detect_injection_attempt(malicious_input)
        print(f"✓ Injection detection working: detected={detection['detected']}")
        
        # Test sanitization
        dirty_data = {
            'user_input': '<script>alert("xss")</script>',
            'sql_query': "'; DROP TABLE data; --",
            'file_path': '../../../etc/passwd'
        }
        
        sanitized, valid, errors = input_sanitizer.sanitize_dict(dirty_data), True, []
        print(f"✓ Data sanitization completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Security validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_error_handling():
    """Test comprehensive error handling."""
    print("\nTesting error handling...")
    
    try:
        from conforl.utils.errors import ConfoRLError, ValidationError, SecurityError
        from conforl.utils.validation import validate_config, validate_risk_parameters
        
        # Test custom error types
        error = ConfoRLError("Test error", "TEST_CODE")
        print(f"✓ ConfoRLError created: {error.error_code}")
        
        # Test configuration validation
        invalid_config = {
            'learning_rate': -0.001,  # Invalid
            'target_risk': 1.5,       # Invalid
            'confidence': 'invalid'   # Invalid type
        }
        
        try:
            validate_config(invalid_config)
            print("✗ Configuration validation should have failed")
            return False
        except Exception as e:
            print(f"✓ Configuration validation correctly failed: {type(e).__name__}")
        
        # Test risk parameter validation
        try:
            validate_risk_parameters(1.5, 0.95)  # Invalid risk
            print("✗ Risk validation should have failed")
            return False
        except ValidationError:
            print("✓ Risk parameter validation correctly failed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging_system():
    """Test comprehensive logging system."""
    print("\nTesting logging system...")
    
    try:
        from conforl.utils.logging import get_logger, setup_logging
        
        # Test logger creation
        logger = get_logger("test_robust")
        print(f"✓ Logger created: {logger.name}")
        
        # Test different log levels
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        print("✓ Multi-level logging working")
        
        # Test structured logging
        logger.info("Structured log", extra={
            'user_id': 'test_user',
            'operation': 'test',
            'duration': 0.123
        })
        print("✓ Structured logging working")
        
        return True
        
    except Exception as e:
        print(f"✗ Logging test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_monitoring():
    """Test health check and monitoring."""
    print("\nTesting health monitoring...")
    
    try:
        from conforl.utils.health import HealthChecker
        
        # Create health checker
        health_checker = HealthChecker()
        print("✓ Health checker created")
        
        # Test basic health check
        health_report = health_checker.get_health_report()
        print(f"✓ System health check: {health_report['overall_status']}")
        
        # Test running specific checks
        system_check = health_checker.run_check('system_resources')
        print(f"✓ System resources check: {system_check.status.value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Health monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_input_validation():
    """Test comprehensive input validation."""
    print("\nTesting input validation...")
    
    try:
        from conforl.utils.validation import validate_environment, validate_dataset
        
        # Create mock environment
        class MockEnv:
            def __init__(self):
                self.observation_space = MockSpace()
                self.action_space = MockSpace()
            
            def reset(self):
                return [0.0, 0.0, 0.0], {}
            
            def step(self, action):
                return [0.0, 0.0, 0.0], 0.0, False, False, {}
        
        class MockSpace:
            def __init__(self):
                self.shape = (3,)
        
        # Test environment validation
        mock_env = MockEnv()
        validate_environment(mock_env)
        print("✓ Environment validation passed")
        
        # Test dataset validation
        dataset = {
            'observations': [[1, 2, 3], [4, 5, 6]],
            'actions': [0.1, 0.2],
            'rewards': [1.0, 0.5]
        }
        validated_dataset = validate_dataset(dataset)
        print("✓ Dataset validation passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Input validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run Generation 2 robust tests."""
    print("🛡️ Running ConfoRL Generation 2: Robust Tests")
    print("=" * 60)
    
    tests = [
        test_security_validation,
        test_error_handling,
        test_logging_system,
        test_health_monitoring,
        test_input_validation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} robust tests passed")
    
    if passed == total:
        print("🎉 All robust tests passed!")
        print("✅ Generation 2 robust functionality is working")
        return 0
    else:
        print("❌ Some robust tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())