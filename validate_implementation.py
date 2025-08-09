#!/usr/bin/env python3
"""
Comprehensive validation script for ConfoRL implementation.
Tests core functionality, research extensions, and production readiness.
"""

import sys
import os
import time
import traceback
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic ConfoRL imports."""
    print("🧪 Testing basic imports...")
    
    try:
        import conforl
        from conforl.core import conformal, types
        from conforl.algorithms import base, sac
        from conforl.risk import measures, controllers
        from conforl.utils import logging, errors, validation
        print("✅ Basic imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_core_functionality():
    """Test core ConfoRL functionality."""
    print("🧪 Testing core functionality...")
    
    try:
        from conforl.core.conformal import SplitConformalPredictor
        from conforl.risk.measures import SafetyViolationRisk
        from conforl.core.types import TrajectoryData
        
        # Test conformal predictor
        predictor = SplitConformalPredictor(risk_level=0.05)
        
        # Test risk measure
        risk_measure = SafetyViolationRisk(safety_threshold=0.5)
        
        # Test trajectory (mock)
        trajectory = type('MockTrajectory', (), {
            'states': [[1, 2, 3], [4, 5, 6]],
            'actions': [[0.1], [0.2]],
            'rewards': [1.0, 2.0],
            'next_states': [[4, 5, 6], [7, 8, 9]],
            'dones': [False, True]
        })()
        
        # Test risk computation
        risk_score = risk_measure.compute(trajectory)
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 1.0
        
        print("✅ Core functionality tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_research_extensions():
    """Test research extension imports."""
    print("🧪 Testing research extensions...")
    
    try:
        # Test causal research
        from conforl.research.causal import CausalGraph, CausalRiskController
        
        # Test adversarial research  
        from conforl.research.adversarial import AdversarialAttackGenerator, AttackType
        
        # Test multi-agent research
        from conforl.research.multi_agent import MultiAgentRiskController, AgentInfo
        
        # Test compositional research
        from conforl.research.compositional import CompositionalRiskController
        
        print("✅ Research extensions import successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Research extensions import failed: {e}")
        return False

def test_security_features():
    """Test security framework."""
    print("🧪 Testing security features...")
    
    try:
        from conforl.security.validation import SecurityValidator, InputSanitizer
        from conforl.security.encryption import EncryptionManager
        from conforl.security.audit import SecurityAuditor
        from conforl.security.access_control import AccessController
        
        # Test input validation
        validator = SecurityValidator()
        is_valid, error = validator.validate_input('target_risk', 0.05)
        assert is_valid == True
        
        is_valid, error = validator.validate_input('target_risk', 1.5)
        assert is_valid == False
        
        # Test input sanitization
        sanitizer = InputSanitizer()
        clean_input = sanitizer.sanitize_string("normal input")
        assert clean_input == "normal input"
        
        malicious_input = sanitizer.sanitize_string("'; DROP TABLE users; --")
        assert "DROP TABLE" not in malicious_input
        
        print("✅ Security features tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Security tests failed: {e}")
        traceback.print_exc()
        return False

def test_scaling_features():
    """Test scaling and performance features."""
    print("🧪 Testing scaling features...")
    
    try:
        from conforl.scaling.distributed import DistributedAgent, ClusterManager
        from conforl.scaling.performance import MemoryManager, ComputationCache, PerformanceOptimizer
        
        # Test performance optimizer
        memory_manager = MemoryManager(max_cache_size_mb=100.0)
        cache = ComputationCache(max_size=100)
        optimizer = PerformanceOptimizer(memory_manager, cache)
        
        # Test function optimization
        @optimizer.optimize_function
        def test_function(x):
            return x * 2
        
        result = test_function(5)
        assert result == 10
        
        print("✅ Scaling features tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Scaling tests failed: {e}")
        traceback.print_exc()
        return False

def test_benchmarking():
    """Test benchmarking framework."""
    print("🧪 Testing benchmarking framework...")
    
    try:
        from conforl.benchmarks.framework import BenchmarkRunner
        from conforl.benchmarks.research_benchmarks import ResearchBenchmarkRunner
        
        # Test basic benchmark runner creation
        runner = BenchmarkRunner()
        assert runner is not None
        
        # Test research benchmark runner
        research_runner = ResearchBenchmarkRunner()
        assert research_runner is not None
        
        print("✅ Benchmarking framework tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Benchmarking tests failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between components."""
    print("🧪 Testing component integration...")
    
    try:
        from conforl.algorithms.sac import ConformaSAC
        from conforl.risk.controllers import AdaptiveRiskController
        from conforl.risk.measures import SafetyViolationRisk
        
        # Create mock environment
        mock_env = type('MockEnv', (), {
            'observation_space': type('Space', (), {'shape': (4,)})(),
            'action_space': type('Space', (), {'shape': (2,)})()
        })()
        
        # Test SAC integration
        risk_controller = AdaptiveRiskController(target_risk=0.05)
        agent = ConformaSAC(env=mock_env, risk_controller=risk_controller)
        
        assert agent is not None
        assert agent.risk_controller == risk_controller
        
        print("✅ Integration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        traceback.print_exc()
        return False

def run_performance_check():
    """Run basic performance checks."""
    print("⚡ Running performance checks...")
    
    try:
        import numpy as np
        from conforl.scaling.performance import get_performance_optimizer
        
        # Test computation caching performance
        optimizer = get_performance_optimizer()
        
        @optimizer.computation_cache.memoize()
        def expensive_computation(n):
            return np.sum(np.random.randn(n, n))
        
        # Time cached vs uncached
        start_time = time.time()
        result1 = expensive_computation(100)
        first_call_time = time.time() - start_time
        
        start_time = time.time()
        result2 = expensive_computation(100)  # Should be cached
        second_call_time = time.time() - start_time
        
        assert result1 == result2  # Results should be identical
        
        speedup = first_call_time / max(second_call_time, 1e-6)
        print(f"✅ Caching speedup: {speedup:.1f}x")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance check failed: {e}")
        return False

def validate_file_structure():
    """Validate project file structure."""
    print("📁 Validating file structure...")
    
    required_files = [
        'setup.py',
        'requirements.txt',
        'README.md',
        'CLAUDE.md',
        'conforl/__init__.py',
        'conforl/core/__init__.py',
        'conforl/algorithms/__init__.py',
        'conforl/risk/__init__.py',
        'conforl/utils/__init__.py',
        'conforl/research/__init__.py',
        'conforl/security/__init__.py',
        'conforl/scaling/__init__.py',
        'conforl/benchmarks/__init__.py',
        'tests/__init__.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ File structure validation passed")
        return True

def main():
    """Run comprehensive validation."""
    print("🚀 ConfoRL Implementation Validation")
    print("=" * 50)
    
    test_results = []
    
    # Run all validation tests
    tests = [
        ("File Structure", validate_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Core Functionality", test_core_functionality),
        ("Research Extensions", test_research_extensions),
        ("Security Features", test_security_features),
        ("Scaling Features", test_scaling_features),
        ("Benchmarking", test_benchmarking),
        ("Integration", test_integration),
        ("Performance", run_performance_check)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} validation failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed_tests += 1
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("ConfoRL implementation is ready for production deployment.")
        return 0
    else:
        print("⚠️  Some validation tests failed.")
        print("Please review the failed tests before deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())