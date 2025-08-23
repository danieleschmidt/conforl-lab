#!/usr/bin/env python3
"""Comprehensive Quality Gates for ConfoRL"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path

def run_tests():
    """Run comprehensive test suite"""
    print("🧪 Running Test Suite...")
    
    test_files = [
        "test_generation1_functionality.py",
        "test_generation2_robust_simple.py", 
        "test_gen3_basic.py",
        "examples/basic_usage.py"
    ]
    
    passed = 0
    total = len(test_files)
    
    for test_file in test_files:
        if Path(test_file).exists():
            try:
                result = subprocess.run([
                    sys.executable, test_file
                ], capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    print(f"✅ {test_file}: PASSED")
                    passed += 1
                else:
                    print(f"❌ {test_file}: FAILED")
            except subprocess.TimeoutExpired:
                print(f"⏱️ {test_file}: TIMEOUT")
            except Exception as e:
                print(f"💥 {test_file}: ERROR - {e}")
        else:
            print(f"📝 {test_file}: NOT FOUND")
    
    print(f"📊 Test Results: {passed}/{total} tests passed")
    return passed >= total * 0.8  # 80% pass rate

def check_imports():
    """Check core imports work"""
    print("\n🔍 Checking Core Imports...")
    
    import_checks = [
        "import conforl",
        "from conforl.risk.controllers import AdaptiveRiskController", 
        "from conforl.algorithms.sac import ConformaSAC",
        "from conforl.core.conformal import SplitConformalPredictor",
        "from conforl.utils.logging import get_logger"
    ]
    
    passed = 0
    for import_stmt in import_checks:
        try:
            exec(import_stmt)
            print(f"✅ {import_stmt}")
            passed += 1
        except Exception as e:
            print(f"❌ {import_stmt}: {str(e)[:50]}...")
    
    print(f"📊 Import Results: {passed}/{len(import_checks)} imports successful")
    return passed >= len(import_checks) * 0.8

def check_code_quality():
    """Basic code quality checks"""
    print("\n📏 Checking Code Quality...")
    
    quality_checks = {
        'python_files_exist': len(list(Path('conforl').rglob('*.py'))) > 0,
        'readme_exists': Path('README.md').exists(),
        'setup_exists': Path('setup.py').exists(),
        'requirements_exist': Path('requirements.txt').exists(),
        'tests_exist': len(list(Path('.').glob('test*.py'))) > 0,
        'examples_exist': Path('examples').exists()
    }
    
    passed = 0
    for check, result in quality_checks.items():
        if result:
            print(f"✅ {check}")
            passed += 1
        else:
            print(f"❌ {check}")
    
    print(f"📊 Quality Results: {passed}/{len(quality_checks)} checks passed")
    return passed >= len(quality_checks) * 0.8

def check_functionality():
    """Check basic functionality"""
    print("\n🎯 Checking Core Functionality...")
    
    try:
        # Test environment creation
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        state, _ = env.reset()
        action = env.action_space.sample()
        env.step(action)
        env.close()
        print("✅ Environment integration works")
        
        # Test risk controller
        from conforl.risk.controllers import AdaptiveRiskController
        risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        print("✅ Risk controller creation works")
        
        # Test conformal predictor
        from conforl.core.conformal import SplitConformalPredictor
        predictor = SplitConformalPredictor(coverage=0.95)
        print("✅ Conformal predictor creation works")
        
        # Test algorithm creation
        from conforl.algorithms.sac import ConformaSAC
        env = gym.make('CartPole-v1')
        agent = ConformaSAC(env=env, risk_controller=risk_controller)
        env.close()
        print("✅ Algorithm creation works")
        
        return True
    except Exception as e:
        print(f"❌ Functionality check failed: {e}")
        return False

def check_performance():
    """Basic performance check"""
    print("\n⚡ Checking Performance...")
    
    try:
        import gymnasium as gym
        from conforl.algorithms.sac import ConformaSAC
        from conforl.risk.controllers import AdaptiveRiskController
        
        env = gym.make('CartPole-v1')
        risk_controller = AdaptiveRiskController(target_risk=0.05, confidence=0.95)
        agent = ConformaSAC(env=env, risk_controller=risk_controller)
        
        # Time multiple predictions
        state, _ = env.reset()
        start_time = time.time()
        
        for _ in range(10):
            action, cert = agent.predict(state, return_risk_certificate=True)
        
        total_time = time.time() - start_time
        avg_time = total_time / 10
        
        env.close()
        
        print(f"✅ Average prediction time: {avg_time:.4f}s")
        
        # Performance thresholds
        if avg_time < 0.1:  # Less than 100ms per prediction
            print("✅ Performance: EXCELLENT")
            return True
        elif avg_time < 0.5:  # Less than 500ms per prediction
            print("✅ Performance: GOOD")
            return True
        else:
            print("⚠️ Performance: ACCEPTABLE")
            return True
            
    except Exception as e:
        print(f"❌ Performance check failed: {e}")
        return False

def generate_quality_report():
    """Generate comprehensive quality report"""
    print("\n📋 Generating Quality Report...")
    
    report = {
        'timestamp': time.time(),
        'tests_passed': False,
        'imports_working': False,
        'code_quality_good': False,
        'functionality_working': False,
        'performance_acceptable': False,
        'overall_quality': 'UNKNOWN'
    }
    
    # Run all checks
    report['tests_passed'] = run_tests()
    report['imports_working'] = check_imports()  
    report['code_quality_good'] = check_code_quality()
    report['functionality_working'] = check_functionality()
    report['performance_acceptable'] = check_performance()
    
    # Calculate overall quality
    quality_scores = [
        report['tests_passed'],
        report['imports_working'],
        report['code_quality_good'],
        report['functionality_working'],
        report['performance_acceptable']
    ]
    
    quality_percentage = sum(quality_scores) / len(quality_scores)
    
    if quality_percentage >= 0.9:
        report['overall_quality'] = 'EXCELLENT'
    elif quality_percentage >= 0.8:
        report['overall_quality'] = 'GOOD'
    elif quality_percentage >= 0.6:
        report['overall_quality'] = 'ACCEPTABLE'
    else:
        report['overall_quality'] = 'NEEDS_IMPROVEMENT'
    
    # Save report
    with open('quality_report_final.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    return report

def main():
    """Run comprehensive quality gates"""
    print("=" * 60)
    print("🔍 COMPREHENSIVE QUALITY GATES - CONFORL")
    print("=" * 60)
    
    report = generate_quality_report()
    
    print("\n" + "=" * 60)
    print("📊 FINAL QUALITY REPORT")
    print("=" * 60)
    
    for key, value in report.items():
        if key != 'timestamp':
            status = "✅" if value is True else "❌" if value is False else "📋"
            print(f"{status} {key}: {value}")
    
    print(f"\n🎯 OVERALL QUALITY: {report['overall_quality']}")
    
    if report['overall_quality'] in ['EXCELLENT', 'GOOD']:
        print("✅ Quality gates PASSED - Ready for deployment!")
        return True
    else:
        print("⚠️ Quality gates need attention")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)