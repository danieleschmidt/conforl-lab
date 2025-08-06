#!/usr/bin/env python3
"""Basic test of ConfoRL core functionality without external dependencies."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

def test_imports():
    """Test basic imports work."""
    print("Testing imports...")
    
    try:
        # Test core types
        from conforl.core.types import RiskCertificate, TrajectoryData, ConformalSet
        print("✓ Core types imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core types: {e}")
        # Try to import step by step to find the issue
        try:
            from conforl.core import types
            print("✓ Core types module imported")
        except ImportError as e2:
            print(f"✗ Failed to import core types module: {e2}")
        return False
    
    try:
        # Test utilities
        from conforl.utils.errors import ConfoRLError, ValidationError
        from conforl.utils.logging import get_logger
        print("✓ Utilities imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import utilities: {e}")
        return False
    
    return True

def test_core_types():
    """Test core type instantiation."""
    print("\nTesting core types...")
    
    try:
        from conforl.core.types import RiskCertificate
        
        # Test RiskCertificate creation
        cert = RiskCertificate(
            risk_bound=0.05,
            confidence=0.95,
            coverage_guarantee=0.95,
            method="test",
            sample_size=1000
        )
        
        assert cert.risk_bound == 0.05
        assert cert.confidence == 0.95
        print("✓ RiskCertificate created successfully")
        
    except Exception as e:
        print(f"✗ Failed to create RiskCertificate: {e}")
        return False
    
    return True

def test_error_handling():
    """Test custom error classes."""
    print("\nTesting error handling...")
    
    try:
        from conforl.utils.errors import ConfoRLError, ValidationError
        
        # Test basic error
        error = ConfoRLError("Test error", "TEST_CODE")
        assert str(error) == "[TEST_CODE] Test error"
        print("✓ ConfoRLError works correctly")
        
        # Test validation error
        val_error = ValidationError("Invalid input", "test_validation")
        assert val_error.validation_type == "test_validation"
        print("✓ ValidationError works correctly")
        
    except Exception as e:
        print(f"✗ Failed error handling test: {e}")
        return False
    
    return True

def test_logging():
    """Test logging setup without dependencies."""
    print("\nTesting logging...")
    
    try:
        from conforl.utils.logging import get_logger
        
        logger = get_logger("test")
        assert logger.name == "ConfoRL.test"
        print("✓ Logger created successfully")
        
    except Exception as e:
        print(f"✗ Failed logging test: {e}")
        return False
    
    return True

def main():
    """Run basic tests."""
    print("🧪 Running ConfoRL Basic Tests")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_core_types,
        test_error_handling,
        test_logging
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        else:
            break  # Stop on first failure
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All basic tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())