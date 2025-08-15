#!/usr/bin/env python3
"""Autonomous execution test for ConfoRL - basic functionality validation."""

import sys
import time

def test_core_imports():
    """Test core module imports."""
    print("Testing core imports...")
    
    try:
        import conforl
        print(f"✓ ConfoRL v{conforl.__version__} imported successfully")
    except Exception as e:
        print(f"✗ ConfoRL import failed: {e}")
        return False
    
    try:
        from conforl.core.conformal import SplitConformalPredictor
        print("✓ Core conformal predictor imported")
    except Exception as e:
        print(f"✗ Core conformal import failed: {e}")
        return False
    
    try:
        from conforl.core.types import RiskCertificate, TrajectoryData
        print("✓ Core types imported")
    except Exception as e:
        print(f"✗ Core types import failed: {e}")
        return False
    
    return True

def test_conformal_prediction():
    """Test basic conformal prediction functionality."""
    print("\nTesting conformal prediction...")
    
    try:
        from conforl.core.conformal import SplitConformalPredictor
        from conforl.core.types import ConformalSet
        
        # Create predictor
        predictor = SplitConformalPredictor(coverage=0.95)
        
        # Mock calibration scores
        calibration_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        predictor.calibrate(calibration_scores)
        
        # Test prediction intervals
        test_predictions = [0.5, 0.6, 0.7]
        lower, upper = predictor.get_prediction_interval(test_predictions)
        
        print(f"✓ Conformal prediction intervals: {list(zip(lower, upper))}")
        
        # Test conformal set
        conformal_set = predictor.predict(test_predictions)
        print(f"✓ Conformal set generated with coverage: {conformal_set.coverage}")
        
        return True
        
    except Exception as e:
        print(f"✗ Conformal prediction test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_risk_components():
    """Test risk control components."""
    print("\nTesting risk control components...")
    
    try:
        from conforl.risk.controllers import AdaptiveRiskController
        from conforl.risk.measures import SafetyViolationRisk
        from conforl.core.types import TrajectoryData
        
        # Create risk controller
        controller = AdaptiveRiskController(
            target_risk=0.05,
            confidence=0.95
        )
        print("✓ Risk controller created")
        
        # Create risk measure
        risk_measure = SafetyViolationRisk()
        print("✓ Risk measure created")
        
        # Mock trajectory data
        trajectory = TrajectoryData(
            states=[[0.1, 0.2], [0.3, 0.4]],
            actions=[0, 1],
            rewards=[1.0, 1.0],
            dones=[False, True],
            infos=[{}, {}]
        )
        
        # Update risk controller
        controller.update(trajectory, risk_measure)
        print("✓ Risk controller updated")
        
        # Get risk certificate
        certificate = controller.get_certificate()
        print(f"✓ Risk certificate: bound={certificate.risk_bound:.4f}, confidence={certificate.confidence}")
        
        return True
        
    except Exception as e:
        print(f"✗ Risk components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_cli_functionality():
    """Test CLI functionality."""
    print("\nTesting CLI functionality...")
    
    try:
        from conforl.cli import main
        print("✓ CLI module imported successfully")
        return True
    except Exception as e:
        print(f"✗ CLI test failed: {e}")
        return False

def test_security_validation():
    """Test security and validation components."""
    print("\nTesting security and validation...")
    
    try:
        from conforl.utils.validation import validate_config, validate_risk_parameters
        from conforl.utils.security import SecurityContext, sanitize_config_dict
        
        # Test config validation
        config = {'learning_rate': 0.001, 'buffer_size': 10000}
        validated = validate_config(config)
        print("✓ Config validation passed")
        
        # Test risk parameter validation
        validate_risk_parameters(0.05, 0.95)
        print("✓ Risk parameter validation passed")
        
        # Test security context
        with SecurityContext("test_operation", "test_user"):
            sanitized = sanitize_config_dict({'test': 'value'})
        print("✓ Security context and sanitization passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Security validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_components():
    """Test utility components."""
    print("\nTesting utility components...")
    
    try:
        from conforl.utils.logging import get_logger
        from conforl.utils.errors import ConfoRLError, ValidationError
        
        # Test logging
        logger = get_logger(__name__)
        logger.info("Test log message")
        print("✓ Logging system functional")
        
        # Test custom errors
        try:
            raise ConfoRLError("Test error", error_code="TEST_ERROR")
        except ConfoRLError as e:
            if e.error_code == "TEST_ERROR":
                print("✓ Custom error system functional")
            else:
                print("✗ Custom error code not preserved")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ Utils components test failed: {e}")
        return False

def main():
    """Run autonomous execution tests."""
    print("=== ConfoRL Autonomous Execution Test ===")
    print(f"Python version: {sys.version}")
    print(f"Test started at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        test_core_imports,
        test_conformal_prediction,
        test_risk_components,
        test_cli_functionality,
        test_security_validation,
        test_utils_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed! ConfoRL is ready for autonomous execution.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())