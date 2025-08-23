#!/usr/bin/env python3
"""Generation 2: Simple robust testing"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_basic_robustness():
    """Test basic robustness features"""
    print("🧪 Testing Basic Robustness...")
    
    try:
        # Test import robustness
        from conforl.utils.validation import validate_config
        from conforl.utils.errors import ValidationError, ConfoRLError
        from conforl.utils.logging import get_logger
        
        # Test logger
        logger = get_logger("test")
        logger.info("Robustness test")
        print("✅ Logging works")
        
        # Test basic validation
        try:
            validate_config({})  # Empty config should fail
        except Exception:
            print("✅ Validation catches empty config")
        
        return True
    except Exception as e:
        print(f"❌ Basic robustness error: {e}")
        return False

def test_algorithm_robustness():
    """Test algorithm error handling"""
    print("\n🧪 Testing Algorithm Robustness...")
    
    try:
        import gymnasium as gym
        from conforl.algorithms.sac import ConformaSAC
        from conforl.risk.controllers import AdaptiveRiskController
        
        env = gym.make('CartPole-v1')
        risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        agent = ConformaSAC(env=env, risk_controller=risk_controller)
        
        # Test with valid state
        state, _ = env.reset()
        action, cert = agent.predict(state, return_risk_certificate=True)
        print(f"✅ Normal prediction works: action shape = {getattr(action, 'shape', 'scalar')}")
        
        # Test error handling
        try:
            agent.predict([])  # Empty state
        except Exception:
            print("✅ Empty state properly handled")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Algorithm robustness error: {e}")
        return False

def main():
    """Run Generation 2 robustness tests"""
    print("=" * 50)
    print("🛡️ GENERATION 2: MAKE IT ROBUST - SIMPLE TESTS")  
    print("=" * 50)
    
    tests = [test_basic_robustness, test_algorithm_robustness]
    passed = sum(1 for test in tests if test())
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed >= len(tests) * 0.8:
        print("✅ Generation 2 robustness tests PASSED")
        return True
    else:
        print("❌ Generation 2 robustness tests FAILED") 
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)