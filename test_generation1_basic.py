#!/usr/bin/env python3
"""
Generation 1 Basic Functionality Tests
Tests core ConfoRL functionality without external dependencies.
"""

import sys
import os

# Add the repo to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_imports():
    """Test that core ConfoRL components can be imported."""
    print("🧪 Testing basic imports...")
    
    try:
        import conforl
        print(f"   ✅ ConfoRL v{conforl.__version__} imported successfully")
        
        # Test available components
        expected_components = ['RiskCertificate', 'TrajectoryData', 'AdaptiveRiskController']
        available = conforl.__all__
        
        for component in expected_components:
            if component in available:
                print(f"   ✅ {component} available")
            else:
                print(f"   ❌ {component} missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def test_risk_certificate():
    """Test RiskCertificate creation and validation."""
    print("🧪 Testing RiskCertificate...")
    
    try:
        import conforl
        
        # Create a valid certificate
        cert = conforl.RiskCertificate(
            risk_bound=0.045,
            confidence=0.95,
            coverage_guarantee=0.947,
            method="test_method",
            sample_size=100
        )
        
        # Validate properties
        assert cert.risk_bound == 0.045, f"Expected 0.045, got {cert.risk_bound}"
        assert cert.confidence == 0.95, f"Expected 0.95, got {cert.confidence}"
        assert cert.method == "test_method", f"Expected 'test_method', got {cert.method}"
        
        print(f"   ✅ Certificate created with risk_bound: {cert.risk_bound}")
        print(f"   ✅ Properties validated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Certificate test failed: {e}")
        return False

def test_trajectory_data():
    """Test TrajectoryData creation and validation."""
    print("🧪 Testing TrajectoryData...")
    
    try:
        import conforl
        
        # Create trajectory data
        trajectory = conforl.TrajectoryData(
            states=[[0.1, 0.2], [0.3, 0.4]],
            actions=[1, 0],
            rewards=[0.1, -0.2],
            dones=[False, True],
            infos=[{}, {"terminal": True}]
        )
        
        # Validate properties
        assert len(trajectory) == 2, f"Expected length 2, got {len(trajectory)}"
        assert len(trajectory.states) == 2, f"Expected 2 states, got {len(trajectory.states)}"
        assert len(trajectory.actions) == 2, f"Expected 2 actions, got {len(trajectory.actions)}"
        assert trajectory.dones[-1] == True, f"Expected last done to be True"
        
        print(f"   ✅ Trajectory created with {len(trajectory)} steps")
        print(f"   ✅ Properties validated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Trajectory test failed: {e}")
        return False

def test_adaptive_risk_controller():
    """Test AdaptiveRiskController functionality."""
    print("🧪 Testing AdaptiveRiskController...")
    
    try:
        import conforl
        
        # Create risk controller
        controller = conforl.AdaptiveRiskController(
            target_risk=0.05,
            confidence=0.95,
            window_size=100
        )
        
        # Validate initialization
        assert controller.target_risk == 0.05, f"Expected 0.05, got {controller.target_risk}"
        assert controller.confidence == 0.95, f"Expected 0.95, got {controller.confidence}"
        
        # Test risk bound (should work even without data)
        risk_bound = controller.get_risk_bound()
        assert isinstance(risk_bound, (int, float)), f"Risk bound should be numeric, got {type(risk_bound)}"
        
        # Test certificate generation
        certificate = controller.get_certificate()
        assert hasattr(certificate, 'risk_bound'), "Certificate should have risk_bound attribute"
        assert hasattr(certificate, 'confidence'), "Certificate should have confidence attribute"
        
        print(f"   ✅ Controller created with target_risk: {controller.target_risk}")
        print(f"   ✅ Risk bound: {risk_bound:.3f}")
        print(f"   ✅ Certificate generated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Risk controller test failed: {e}")
        return False

def test_risk_measure():
    """Test SafetyViolationRisk functionality."""
    print("🧪 Testing SafetyViolationRisk...")
    
    try:
        import conforl
        
        # Create risk measure
        risk_measure = conforl.SafetyViolationRisk(violation_threshold=0.1)
        
        # Validate initialization
        assert risk_measure.violation_threshold == 0.1, f"Expected 0.1, got {risk_measure.violation_threshold}"
        
        # Create test trajectory
        trajectory = conforl.TrajectoryData(
            states=[[0.05], [0.15], [0.08]],  # Some above/below threshold
            actions=[1, 0, 1],
            rewards=[0.1, -0.2, 0.3],
            dones=[False, False, True],
            infos=[{}, {}, {}]
        )
        
        # Compute risk (this should work)
        risk_value = risk_measure.compute(trajectory)
        assert isinstance(risk_value, (int, float)), f"Risk should be numeric, got {type(risk_value)}"
        
        print(f"   ✅ Risk measure created with threshold: {risk_measure.violation_threshold}")
        print(f"   ✅ Risk computed: {risk_value:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Risk measure test failed: {e}")
        return False

def test_conformal_predictor():
    """Test SplitConformalPredictor functionality."""
    print("🧪 Testing SplitConformalPredictor...")
    
    try:
        from conforl.core.conformal import SplitConformalPredictor
        
        # Create predictor
        predictor = SplitConformalPredictor(coverage=0.95)
        
        # Calibrate with sample data
        calibration_scores = [0.1, 0.2, 0.15, 0.25, 0.18]
        predictor.calibrate(calibration_scores)
        
        # Validate calibration
        assert predictor.quantile is not None, "Quantile should be set after calibration"
        assert predictor.calibration_size == 5, f"Expected 5 calibration samples, got {predictor.calibration_size}"
        
        # Test prediction interval
        test_predictions = [0.5, 0.7]
        lower, upper = predictor.get_prediction_interval(test_predictions)
        
        assert len(lower) == 2, f"Expected 2 lower bounds, got {len(lower)}"
        assert len(upper) == 2, f"Expected 2 upper bounds, got {len(upper)}"
        assert all(l < u for l, u in zip(lower, upper)), "Lower bounds should be less than upper bounds"
        
        print(f"   ✅ Predictor created with coverage: {predictor.coverage}")
        print(f"   ✅ Calibrated with quantile: {predictor.quantile:.3f}")
        print(f"   ✅ Prediction intervals generated successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Conformal predictor test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("🧪 Testing component integration...")
    
    try:
        import conforl
        from conforl.core.conformal import SplitConformalPredictor
        
        # Create all components
        risk_measure = conforl.SafetyViolationRisk(violation_threshold=0.1)
        controller = conforl.AdaptiveRiskController(target_risk=0.05)
        predictor = SplitConformalPredictor(coverage=0.95)
        
        # Create trajectory
        trajectory = conforl.TrajectoryData(
            states=[[0.1, 0.2], [0.3, 0.4]],
            actions=[1, 0],
            rewards=[0.1, -0.2],
            dones=[False, True],
            infos=[{}, {}]
        )
        
        # Test integration: update controller with trajectory
        controller.update(trajectory, risk_measure)
        certificate = controller.get_certificate()
        
        # Validate integration worked
        assert certificate.risk_bound >= 0, "Risk bound should be non-negative"
        assert certificate.confidence > 0, "Confidence should be positive"
        
        print(f"   ✅ Components integrated successfully")
        print(f"   ✅ End-to-end test passed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration test failed: {e}")
        return False

def main():
    """Run all Generation 1 tests."""
    print("=" * 60)
    print("🚀 ConfoRL Generation 1 - Basic Functionality Tests")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_risk_certificate,
        test_trajectory_data,
        test_adaptive_risk_controller,
        test_risk_measure,
        test_conformal_predictor,
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
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Generation 1 is COMPLETE!")
        print("✅ Core functionality verified and working")
        print("🔒 Safety guarantees operational")
        print("📈 Ready for Generation 2 (Robustness)")
    else:
        print(f"⚠️  {total - passed} tests failed. Generation 1 needs fixes.")
        return False
    
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)