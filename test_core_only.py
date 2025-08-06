#!/usr/bin/env python3
"""Test only ConfoRL core functionality without full package imports."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_core_types_direct():
    """Test core types import directly."""
    print("Testing core types direct import...")
    
    try:
        # Import the types module directly
        import conforl.core.types as types
        
        # Test RiskCertificate creation
        cert = types.RiskCertificate(
            risk_bound=0.05,
            confidence=0.95,
            coverage_guarantee=0.95,
            method="test",
            sample_size=1000
        )
        
        print(f"✓ RiskCertificate created: risk_bound={cert.risk_bound}")
        
        # Test TrajectoryData creation
        trajectory = types.TrajectoryData(
            states=[1, 2, 3],
            actions=[0.1, 0.2, 0.3],
            rewards=[1.0, 0.5, 0.8],
            dones=[False, False, True],
            infos=[{}, {}, {}]
        )
        
        print(f"✓ TrajectoryData created: length={len(trajectory)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed core types test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_direct():
    """Test utilities import directly."""
    print("\nTesting utilities direct import...")
    
    try:
        # Test error classes
        import conforl.utils.errors as errors
        
        error = errors.ConfoRLError("Test error", "TEST_CODE")
        print(f"✓ ConfoRLError: {error}")
        
        val_error = errors.ValidationError("Invalid input", "test_validation")
        print(f"✓ ValidationError: {val_error.validation_type}")
        
        # Test logging
        import conforl.utils.logging as logging_utils
        
        logger = logging_utils.get_logger("test")
        print(f"✓ Logger created: {logger.name}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed utilities test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_conformal_prediction():
    """Test basic conformal prediction functionality."""
    print("\nTesting conformal prediction...")
    
    try:
        import conforl.core.conformal as conformal
        
        # Test predictor creation
        predictor = conformal.SplitConformalPredictor(coverage=0.95)
        print(f"✓ SplitConformalPredictor created: coverage={predictor.coverage}")
        
        # Test with dummy data
        dummy_data = [1, 2, 3, 4, 5]
        dummy_scores = [0.1, 0.2, 0.15, 0.3, 0.25]
        
        predictor.calibrate(dummy_data, dummy_scores)
        print(f"✓ Calibration completed: quantile={predictor.quantile}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed conformal prediction test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run core-only tests."""
    print("🧪 Running ConfoRL Core-Only Tests")
    print("=" * 50)
    
    tests = [
        test_core_types_direct,
        test_utils_direct,
        test_conformal_prediction
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Results: {passed}/{total} core tests passed")
    
    if passed == total:
        print("🎉 All core tests passed!")
        print("✅ Generation 1 core functionality is working")
        return 0
    else:
        print("❌ Some core tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())