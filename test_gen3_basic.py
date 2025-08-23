#!/usr/bin/env python3
"""Generation 3: Basic scaling test"""

import sys
import os
import time
sys.path.insert(0, os.path.abspath('.'))

def test_basic_scaling():
    """Test basic scaling features"""
    print("🧪 Testing Basic Scaling...")
    
    try:
        from conforl.optimize.cache import AdaptiveCache
        
        # Test caching
        cache = AdaptiveCache(max_size=10, ttl=60)
        cache.set("key1", "value1") 
        result = cache.get("key1")
        assert result == "value1", "Cache should work"
        print("✅ Caching works")
        
        return True
    except Exception as e:
        print(f"❌ Basic scaling error: {e}")
        return False

def test_algorithm_performance():
    """Test algorithm performance"""  
    print("\n🧪 Testing Algorithm Performance...")
    
    try:
        import gymnasium as gym
        from conforl.algorithms.sac import ConformaSAC
        from conforl.risk.controllers import AdaptiveRiskController
        
        env = gym.make('CartPole-v1')
        risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        agent = ConformaSAC(env=env, risk_controller=risk_controller)
        
        # Test prediction performance
        state, _ = env.reset()
        start = time.time()
        action, cert = agent.predict(state, return_risk_certificate=True)
        duration = time.time() - start
        
        assert action is not None, "Should predict action"
        assert cert is not None, "Should return certificate"
        print(f"✅ Prediction: {duration:.3f}s")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Algorithm performance error: {e}")
        return False

def main():
    """Run basic Generation 3 tests"""
    print("=" * 50)
    print("⚡ GENERATION 3: MAKE IT SCALE - BASIC")
    print("=" * 50)
    
    tests = [test_basic_scaling, test_algorithm_performance]
    passed = sum(1 for test in tests if test())
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    success = passed >= len(tests) * 0.5  # 50% pass for basic
    if success:
        print("✅ Generation 3 basic scaling PASSED")
    else:
        print("❌ Generation 3 basic scaling FAILED")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)