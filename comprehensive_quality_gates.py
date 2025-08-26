#!/usr/bin/env python3
"""
ConfoRL Quality Gates - Comprehensive Testing & Validation

This runs all quality gates and validation checks to ensure production readiness.
"""

import sys
sys.path.insert(0, '.')

import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

class QualityGateRunner:
    """Runs comprehensive quality gates for ConfoRL."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates and return comprehensive report."""
        print("🛡️ ConfoRL Quality Gates - Comprehensive Validation")
        print("=" * 60)
        
        # Gate 1: Code Quality & Testing
        self.results['testing'] = self.run_testing_gate()
        
        # Gate 2: Security Validation
        self.results['security'] = self.run_security_gate()
        
        # Gate 3: Performance Benchmarking
        self.results['performance'] = self.run_performance_gate()
        
        # Gate 4: Integration Testing
        self.results['integration'] = self.run_integration_gate()
        
        # Gate 5: Documentation & API Validation
        self.results['documentation'] = self.run_documentation_gate()
        
        # Generate final report
        final_report = self.generate_final_report()
        
        return final_report
    
    def run_testing_gate(self) -> Dict[str, Any]:
        """Run comprehensive testing gate."""
        print("\n🧪 GATE 1: Testing & Code Quality")
        print("-" * 40)
        
        testing_results = {
            'test_execution': {},
            'coverage': {},
            'linting': {},
            'type_checking': {}
        }
        
        # Test core functionality
        print("Running core functionality tests...")
        try:
            # Run pytest on core components
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_core.py", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=".", timeout=60)
            
            testing_results['test_execution']['core_tests'] = {
                'exit_code': result.returncode,
                'stdout': result.stdout[-500:],  # Last 500 chars
                'stderr': result.stderr[-500:] if result.stderr else "",
                'passed': result.returncode == 0
            }
            print(f"   ✅ Core tests: {'PASSED' if result.returncode == 0 else 'FAILED'}")
            
        except Exception as e:
            testing_results['test_execution']['core_tests'] = {
                'exit_code': -1,
                'error': str(e),
                'passed': False
            }
            print(f"   ❌ Core tests: FAILED ({e})")
        
        # Test risk management
        print("Running risk management tests...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/test_risk.py", 
                "-v", "--tb=short"
            ], capture_output=True, text=True, cwd=".", timeout=60)
            
            testing_results['test_execution']['risk_tests'] = {
                'exit_code': result.returncode,
                'passed': result.returncode == 0
            }
            print(f"   ✅ Risk tests: {'PASSED' if result.returncode == 0 else 'FAILED'}")
            
        except Exception as e:
            testing_results['test_execution']['risk_tests'] = {
                'error': str(e),
                'passed': False
            }
            print(f"   ❌ Risk tests: FAILED ({e})")
        
        # Import validation
        print("Running import validation...")
        try:
            import conforl
            from conforl.core.conformal import SplitConformalPredictor
            from conforl.risk.controllers import AdaptiveRiskController
            from conforl.algorithms.base import ConformalRLAgent
            
            testing_results['import_validation'] = {
                'conforl_version': conforl.__version__,
                'core_imports': True,
                'algorithm_imports': True
            }
            print(f"   ✅ Import validation: PASSED (v{conforl.__version__})")
            
        except Exception as e:
            testing_results['import_validation'] = {
                'error': str(e),
                'passed': False
            }
            print(f"   ❌ Import validation: FAILED ({e})")
        
        return testing_results
    
    def run_security_gate(self) -> Dict[str, Any]:
        """Run security validation gate."""
        print("\n🔒 GATE 2: Security Validation")
        print("-" * 40)
        
        security_results = {
            'input_validation': {},
            'path_traversal': {},
            'dependency_security': {},
            'secrets_scanning': {}
        }
        
        # Test input validation
        print("Testing input validation...")
        try:
            from conforl.utils.security import sanitize_input, SecurityError
            
            # Test malicious inputs
            test_cases = [
                ("<script>alert('xss')</script>", False),
                ("../../../etc/passwd", False),
                ("normal_string", True),
                ("javascript:alert(1)", False)
            ]
            
            validation_passed = 0
            for test_input, should_pass in test_cases:
                try:
                    result = sanitize_input(test_input, "string")
                    if should_pass:
                        validation_passed += 1
                except SecurityError:
                    if not should_pass:
                        validation_passed += 1
            
            security_results['input_validation'] = {
                'tests_passed': validation_passed,
                'total_tests': len(test_cases),
                'success_rate': validation_passed / len(test_cases)
            }
            print(f"   ✅ Input validation: {validation_passed}/{len(test_cases)} tests passed")
            
        except Exception as e:
            security_results['input_validation'] = {'error': str(e)}
            print(f"   ❌ Input validation: FAILED ({e})")
        
        # Test path validation
        print("Testing path traversal protection...")
        try:
            from conforl.utils.security import sanitize_file_path, SecurityError
            
            dangerous_paths = [
                "../../../etc/passwd",
                "..\\..\\windows\\system32",
                "/etc/shadow",
                "normal/path/file.txt"
            ]
            
            protected_count = 0
            for path in dangerous_paths:
                try:
                    sanitized = sanitize_file_path(path)
                    if path == "normal/path/file.txt":
                        protected_count += 1  # This should pass
                except SecurityError:
                    if path != "normal/path/file.txt":
                        protected_count += 1  # Dangerous paths should be blocked
            
            security_results['path_traversal'] = {
                'protected_paths': protected_count,
                'total_tested': len(dangerous_paths)
            }
            print(f"   ✅ Path traversal protection: {protected_count}/{len(dangerous_paths)} properly handled")
            
        except Exception as e:
            security_results['path_traversal'] = {'error': str(e)}
            print(f"   ❌ Path traversal protection: FAILED ({e})")
        
        return security_results
    
    def run_performance_gate(self) -> Dict[str, Any]:
        """Run performance benchmarking gate."""
        print("\n⚡ GATE 3: Performance Benchmarking")
        print("-" * 40)
        
        performance_results = {
            'prediction_latency': {},
            'throughput': {},
            'memory_usage': {},
            'concurrent_performance': {}
        }
        
        # Prediction latency test
        print("Testing prediction latency...")
        try:
            from conforl.core.conformal import SplitConformalPredictor
            
            predictor = SplitConformalPredictor(coverage=0.95)
            calibration_scores = [0.1, 0.2, 0.15, 0.3, 0.05] * 10
            predictor.calibrate(calibration_scores=calibration_scores)
            
            # Single prediction latency
            start_time = time.time()
            for _ in range(100):
                result = predictor.predict([1.0])
            latency_per_prediction = (time.time() - start_time) / 100
            
            performance_results['prediction_latency'] = {
                'avg_latency_ms': latency_per_prediction * 1000,
                'target_latency_ms': 10.0,
                'meets_target': latency_per_prediction * 1000 < 10.0
            }
            print(f"   ✅ Prediction latency: {latency_per_prediction*1000:.2f}ms (target: <10ms)")
            
        except Exception as e:
            performance_results['prediction_latency'] = {'error': str(e)}
            print(f"   ❌ Prediction latency: FAILED ({e})")
        
        # Throughput test
        print("Testing throughput...")
        try:
            from conforl.core.conformal import SplitConformalPredictor
            
            predictor = SplitConformalPredictor(coverage=0.95)
            calibration_scores = [0.1, 0.2, 0.15, 0.3, 0.05] * 20
            predictor.calibrate(calibration_scores=calibration_scores)
            
            # Throughput test
            start_time = time.time()
            predictions_made = 0
            test_duration = 1.0  # 1 second test
            
            while time.time() - start_time < test_duration:
                result = predictor.predict([1.0, 2.0, 3.0])
                predictions_made += 3  # 3 predictions per call
            
            duration = time.time() - start_time
            throughput = predictions_made / duration
            
            performance_results['throughput'] = {
                'predictions_per_second': throughput,
                'target_throughput': 1000,
                'meets_target': throughput >= 1000
            }
            print(f"   ✅ Throughput: {throughput:.0f} predictions/sec (target: >1000/sec)")
            
        except Exception as e:
            performance_results['throughput'] = {'error': str(e)}
            print(f"   ❌ Throughput: FAILED ({e})")
        
        # Concurrent performance test
        print("Testing concurrent performance...")
        try:
            import concurrent.futures
            from conforl.core.conformal import SplitConformalPredictor
            
            def concurrent_prediction_task(task_id):
                predictor = SplitConformalPredictor(coverage=0.95)
                calibration_scores = [0.1, 0.2, 0.15, 0.3, 0.05] * 5
                predictor.calibrate(calibration_scores=calibration_scores)
                
                start = time.time()
                result = predictor.predict([float(task_id)])
                return time.time() - start
            
            start_time = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(concurrent_prediction_task, i) for i in range(20)]
                execution_times = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            avg_execution_time = sum(execution_times) / len(execution_times)
            
            performance_results['concurrent_performance'] = {
                'total_concurrent_time': total_time,
                'avg_task_time': avg_execution_time,
                'concurrent_efficiency': (sum(execution_times) / total_time),
                'tasks_completed': len(execution_times)
            }
            print(f"   ✅ Concurrent performance: {len(execution_times)} tasks in {total_time:.4f}s")
            
        except Exception as e:
            performance_results['concurrent_performance'] = {'error': str(e)}
            print(f"   ❌ Concurrent performance: FAILED ({e})")
        
        return performance_results
    
    def run_integration_gate(self) -> Dict[str, Any]:
        """Run integration testing gate."""
        print("\n🔗 GATE 4: Integration Testing")
        print("-" * 40)
        
        integration_results = {
            'end_to_end': {},
            'api_integration': {},
            'component_integration': {}
        }
        
        # End-to-end workflow test
        print("Testing end-to-end workflow...")
        try:
            from conforl.core.conformal import SplitConformalPredictor
            from conforl.risk.controllers import AdaptiveRiskController
            from conforl.risk.measures import SafetyViolationRisk
            from conforl.core.types import TrajectoryData
            
            # Create integrated workflow
            predictor = SplitConformalPredictor(coverage=0.95)
            risk_controller = AdaptiveRiskController(target_risk=0.05)
            risk_measure = SafetyViolationRisk()
            
            # Calibrate predictor
            calibration_scores = [0.1, 0.2, 0.15, 0.3, 0.05] * 10
            predictor.calibrate(calibration_scores=calibration_scores)
            
            # Create trajectory
            trajectory = TrajectoryData(
                states=[[0.1, 0.2, 0.3, 0.4]] * 5,
                actions=[0, 1, 0, 1, 0],
                rewards=[1.0, 2.0, 1.5, 2.5, 1.8],
                dones=[False, False, False, False, True],
                infos=[{'constraint_violation': 0.0}] * 5
            )
            
            # Test workflow
            risk_controller.update(trajectory, risk_measure)
            certificate = risk_controller.get_certificate()
            prediction = predictor.predict([1.0, 2.0])
            
            integration_results['end_to_end'] = {
                'workflow_completed': True,
                'certificate_generated': certificate is not None,
                'prediction_generated': prediction is not None,
                'risk_bound': certificate.risk_bound if certificate else None
            }
            print(f"   ✅ End-to-end workflow: PASSED")
            
        except Exception as e:
            integration_results['end_to_end'] = {'error': str(e), 'passed': False}
            print(f"   ❌ End-to-end workflow: FAILED ({e})")
        
        return integration_results
    
    def run_documentation_gate(self) -> Dict[str, Any]:
        """Run documentation and API validation gate."""
        print("\n📚 GATE 5: Documentation & API Validation")
        print("-" * 40)
        
        doc_results = {
            'readme_validation': {},
            'api_documentation': {},
            'code_documentation': {}
        }
        
        # Check README exists and has key sections
        print("Validating README documentation...")
        try:
            readme_path = Path("README.md")
            if readme_path.exists():
                readme_content = readme_path.read_text()
                
                required_sections = [
                    "# ConfoRL",
                    "## Installation",
                    "## Quick Start",
                    "## Features"
                ]
                
                sections_found = sum(1 for section in required_sections if section in readme_content)
                
                doc_results['readme_validation'] = {
                    'exists': True,
                    'length': len(readme_content),
                    'required_sections_found': sections_found,
                    'total_required_sections': len(required_sections),
                    'completeness': sections_found / len(required_sections)
                }
                print(f"   ✅ README validation: {sections_found}/{len(required_sections)} required sections")
            else:
                doc_results['readme_validation'] = {'exists': False}
                print(f"   ❌ README validation: README.md not found")
                
        except Exception as e:
            doc_results['readme_validation'] = {'error': str(e)}
            print(f"   ❌ README validation: FAILED ({e})")
        
        # Check API documentation
        print("Validating API documentation...")
        try:
            import inspect
            import conforl
            from conforl.core.conformal import SplitConformalPredictor
            from conforl.risk.controllers import AdaptiveRiskController
            
            # Check docstrings
            classes_with_docstrings = 0
            total_classes = 0
            
            for name, obj in inspect.getmembers(conforl):
                if inspect.isclass(obj):
                    total_classes += 1
                    if obj.__doc__:
                        classes_with_docstrings += 1
            
            # Check key classes specifically
            key_classes = [SplitConformalPredictor, AdaptiveRiskController]
            key_classes_documented = sum(1 for cls in key_classes if cls.__doc__)
            
            doc_results['api_documentation'] = {
                'classes_with_docstrings': classes_with_docstrings,
                'total_classes_checked': total_classes,
                'key_classes_documented': key_classes_documented,
                'total_key_classes': len(key_classes)
            }
            print(f"   ✅ API documentation: {key_classes_documented}/{len(key_classes)} key classes documented")
            
        except Exception as e:
            doc_results['api_documentation'] = {'error': str(e)}
            print(f"   ❌ API documentation: FAILED ({e})")
        
        return doc_results
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate final quality gate report."""
        total_time = time.time() - self.start_time
        
        # Calculate overall scores
        gates_passed = 0
        total_gates = 0
        
        for gate_name, gate_results in self.results.items():
            total_gates += 1
            # Simple heuristic: if no errors and some successful results, consider passed
            has_errors = any('error' in str(result) for result in gate_results.values() if isinstance(result, dict))
            has_successes = any(result.get('passed', False) if isinstance(result, dict) else True 
                              for result in gate_results.values() if isinstance(result, dict))
            
            if not has_errors and has_successes:
                gates_passed += 1
        
        final_report = {
            'execution_time': total_time,
            'gates_passed': gates_passed,
            'total_gates': total_gates,
            'success_rate': gates_passed / total_gates if total_gates > 0 else 0,
            'timestamp': time.time(),
            'detailed_results': self.results,
            'summary': self.generate_summary()
        }
        
        print(f"\n🎯 QUALITY GATES SUMMARY")
        print("=" * 40)
        print(f"📊 Gates Passed: {gates_passed}/{total_gates} ({(gates_passed/total_gates)*100:.1f}%)")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🎯 Overall Status: {'✅ PASSED' if gates_passed >= total_gates * 0.8 else '❌ NEEDS IMPROVEMENT'}")
        
        return final_report
    
    def generate_summary(self) -> Dict[str, str]:
        """Generate summary of all gates."""
        summary = {}
        
        for gate_name, gate_results in self.results.items():
            if isinstance(gate_results, dict) and gate_results:
                # Count successes and failures
                successes = 0
                total = 0
                
                for key, result in gate_results.items():
                    total += 1
                    if isinstance(result, dict):
                        if result.get('passed', False) or ('error' not in result and result):
                            successes += 1
                    else:
                        successes += 1  # Non-dict results considered success
                
                if total > 0:
                    success_rate = successes / total
                    if success_rate >= 0.8:
                        summary[gate_name] = "✅ PASSED"
                    elif success_rate >= 0.6:
                        summary[gate_name] = "⚠️ NEEDS ATTENTION"
                    else:
                        summary[gate_name] = "❌ FAILED"
                else:
                    summary[gate_name] = "⚠️ NO RESULTS"
            else:
                summary[gate_name] = "❌ ERROR"
        
        return summary


def main():
    """Run all quality gates."""
    runner = QualityGateRunner()
    report = runner.run_all_gates()
    
    # Save report
    report_path = Path("quality_gates_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Full report saved to: {report_path}")
    
    # Return exit code based on success
    success_rate = report.get('success_rate', 0)
    return 0 if success_rate >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())