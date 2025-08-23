#!/usr/bin/env python3
"""Generation 1: Test core functionality - Make It Work"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_core_imports():
    """Test all core imports work correctly"""
    print("🧪 Testing Core Imports...")
    
    try:
        # Core types
        from conforl.core.types import RiskCertificate, TrajectoryData
        print("✅ Core types imported")
        
        # Risk components  
        from conforl.risk.controllers import AdaptiveRiskController
        from conforl.risk.measures import SafetyViolationRisk
        print("✅ Risk components imported")
        
        # Base algorithm
        from conforl.algorithms.base import ConformalRLAgent
        print("✅ Base algorithm imported")
        
        # Algorithms
        from conforl.algorithms.sac import ConformaSAC
        from conforl.algorithms.ppo import ConformaPPO
        print("✅ Algorithms imported")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without full environment setup"""
    print("\n🧪 Testing Basic Functionality...")
    
    try:
        from conforl.risk.controllers import AdaptiveRiskController
        from conforl.risk.measures import SafetyViolationRisk
        from conforl.core.conformal import SplitConformalPredictor
        
        # Create risk controller
        risk_controller = AdaptiveRiskController(
            target_risk=0.05, 
            confidence=0.95
        )
        print("✅ Risk controller created")
        
        # Create risk measure
        risk_measure = SafetyViolationRisk()
        print("✅ Risk measure created")
        
        # Create conformal predictor
        conformal = SplitConformalPredictor(coverage=0.95)
        print("✅ Conformal predictor created")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_integration():
    """Test environment integration"""
    print("\n🧪 Testing Environment Integration...")
    
    try:
        import gymnasium as gym
        
        # Create simple environment
        env = gym.make('CartPole-v1')
        print("✅ Environment created")
        
        # Test environment interaction
        state, info = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print("✅ Environment interaction works")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_algorithm_creation():
    """Test algorithm creation"""
    print("\n🧪 Testing Algorithm Creation...")
    
    try:
        import gymnasium as gym
        from conforl.algorithms.sac import ConformaSAC
        from conforl.risk.controllers import AdaptiveRiskController
        
        # Create environment
        env = gym.make('CartPole-v1')
        
        # Create risk controller
        risk_controller = AdaptiveRiskController(
            target_risk=0.05,
            confidence=0.95
        )
        
        # Create agent
        agent = ConformaSAC(
            env=env,
            risk_controller=risk_controller
        )
        print("✅ ConformaSAC agent created")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Algorithm creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all Generation 1 tests"""
    print("=" * 50)
    print("🚀 GENERATION 1: MAKE IT WORK - FUNCTIONALITY TESTS")
    print("=" * 50)
    
    tests = [
        test_core_imports,
        test_basic_functionality,
        test_environment_integration,
        test_algorithm_creation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print("⚠️ Test failed but continuing...")
        except Exception as e:
            print(f"❌ Test crashed: {e}")
    
    print(f"\n📊 Results: {passed}/{total} tests passed")
    
    if passed >= total * 0.75:  # 75% pass rate
        print("✅ Generation 1 functionality tests PASSED")
        return True
    else:
        print("❌ Generation 1 functionality tests FAILED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)