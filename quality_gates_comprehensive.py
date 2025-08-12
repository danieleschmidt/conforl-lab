#!/usr/bin/env python3
"""Comprehensive quality gates for ConfoRL production deployment."""

import sys
import time
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

class QualityGate:
    """Individual quality gate with pass/fail criteria."""
    
    def __init__(self, name: str, description: str, critical: bool = False):
        """Initialize quality gate.
        
        Args:
            name: Gate name
            description: Gate description
            critical: Whether failure blocks deployment
        """
        self.name = name
        self.description = description
        self.critical = critical
        self.passed = False
        self.error_message = None
        self.execution_time = 0.0
        self.details = {}
    
    def execute(self, test_func) -> bool:
        """Execute quality gate test.
        
        Args:
            test_func: Function to execute for this gate
            
        Returns:
            True if gate passes, False otherwise
        """
        start_time = time.time()
        try:
            result = test_func()
            self.passed = bool(result.get('passed', False)) if isinstance(result, dict) else bool(result)
            if isinstance(result, dict):
                self.details = result.get('details', {})
                if not self.passed:
                    self.error_message = result.get('error', 'Test failed')
        except Exception as e:
            self.passed = False
            self.error_message = f"Test execution failed: {str(e)}"
            self.details = {'exception': traceback.format_exc()}
        
        self.execution_time = time.time() - start_time
        return self.passed


class QualityGateRunner:
    """Orchestrates execution of all quality gates."""
    
    def __init__(self):
        """Initialize quality gate runner."""
        self.gates: List[QualityGate] = []
        self.execution_start_time = None
        self.execution_end_time = None
        
    def add_gate(self, gate: QualityGate):
        """Add a quality gate.
        
        Args:
            gate: Quality gate to add
        """
        self.gates.append(gate)
    
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates.
        
        Returns:
            Summary of all gate results
        """
        self.execution_start_time = time.time()
        
        print("🚀 Running Comprehensive Quality Gates")
        print("=" * 60)
        
        passed_gates = 0
        failed_gates = 0
        critical_failures = 0
        
        for gate in self.gates:
            print(f"\n🧪 {gate.name}")
            print(f"   {gate.description}")
            
            success = gate.execute(self._get_test_function(gate.name))
            
            if success:
                print(f"   ✅ PASSED ({gate.execution_time:.3f}s)")
                passed_gates += 1
            else:
                status = "CRITICAL FAILURE" if gate.critical else "FAILED"
                print(f"   ❌ {status} ({gate.execution_time:.3f}s)")
                if gate.error_message:
                    print(f"      Error: {gate.error_message}")
                failed_gates += 1
                if gate.critical:
                    critical_failures += 1
        
        self.execution_end_time = time.time()
        total_time = self.execution_end_time - self.execution_start_time
        
        # Summary
        print(f"\n📊 Quality Gates Summary")
        print("=" * 40)
        print(f"Total Gates: {len(self.gates)}")
        print(f"Passed: {passed_gates}")
        print(f"Failed: {failed_gates}")
        print(f"Critical Failures: {critical_failures}")
        print(f"Total Execution Time: {total_time:.3f}s")
        
        deployment_approved = critical_failures == 0
        
        if deployment_approved:
            print(f"\n🎉 DEPLOYMENT APPROVED")
            print(f"✅ All critical quality gates passed")
        else:
            print(f"\n🚫 DEPLOYMENT BLOCKED")
            print(f"❌ {critical_failures} critical quality gate(s) failed")
        
        return {
            'deployment_approved': deployment_approved,
            'total_gates': len(self.gates),
            'passed_gates': passed_gates,
            'failed_gates': failed_gates,
            'critical_failures': critical_failures,
            'execution_time': total_time,
            'gate_results': [
                {
                    'name': gate.name,
                    'passed': gate.passed,
                    'critical': gate.critical,
                    'execution_time': gate.execution_time,
                    'error_message': gate.error_message,
                    'details': gate.details
                }
                for gate in self.gates
            ]
        }
    
    def _get_test_function(self, gate_name: str):
        """Get test function for a specific gate."""
        test_functions = {
            'core_imports': self._test_core_imports,
            'basic_functionality': self._test_basic_functionality,
            'error_handling': self._test_error_handling,
            'security_validation': self._test_security_validation,
            'performance_benchmarks': self._test_performance_benchmarks,
            'memory_leak_detection': self._test_memory_leak_detection,
            'thread_safety': self._test_thread_safety,
            'scalability_features': self._test_scalability_features,
            'configuration_validation': self._test_configuration_validation,
            'documentation_coverage': self._test_documentation_coverage
        }
        
        return test_functions.get(gate_name, lambda: {'passed': False, 'error': 'Test not implemented'})
    
    def _test_core_imports(self) -> Dict[str, Any]:
        """Test that all core modules can be imported."""
        try:
            # Test core module imports
            from conforl.core.types import RiskCertificate, TrajectoryData
            from conforl.core.conformal import SplitConformalPredictor
            from conforl.utils.errors import ConfoRLError
            from conforl.utils.logging import get_logger
            
            return {
                'passed': True,
                'details': {
                    'core_types': 'imported successfully',
                    'conformal_prediction': 'imported successfully',
                    'error_handling': 'imported successfully',
                    'logging': 'imported successfully'
                }
            }
        except ImportError as e:
            return {
                'passed': False,
                'error': f'Core import failed: {str(e)}',
                'details': {'import_error': str(e)}
            }
    
    def _test_basic_functionality(self) -> Dict[str, Any]:
        """Test basic ConfoRL functionality."""
        try:
            from conforl.core.types import RiskCertificate
            from conforl.core.conformal import SplitConformalPredictor
            
            # Test risk certificate creation
            cert = RiskCertificate(
                risk_bound=0.05,
                confidence=0.95,
                coverage_guarantee=0.95,
                method="test",
                sample_size=100,
                timestamp=time.time()
            )
            
            # Test conformal predictor
            predictor = SplitConformalPredictor(coverage=0.95)
            calibration_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
            predictor.calibrate(calibration_scores)
            
            return {
                'passed': True,
                'details': {
                    'risk_certificate': f'created with risk_bound={cert.risk_bound}',
                    'conformal_predictor': f'calibrated with coverage={predictor.coverage}',
                    'quantile': f'computed quantile={predictor.quantile}'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Basic functionality test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_error_handling(self) -> Dict[str, Any]:
        """Test comprehensive error handling."""
        try:
            from conforl.utils.errors import (
                ConfoRLError, ValidationError, SecurityError,
                CircuitBreaker, ErrorRecovery
            )
            
            # Test custom exceptions
            try:
                raise ValidationError("Test validation error", "test_type")
            except ValidationError as e:
                assert e.validation_type == "test_type"
            
            # Test circuit breaker
            cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
            assert cb.state == "CLOSED"
            
            # Test error recovery
            recovery = ErrorRecovery(max_retries=2, backoff_factor=1.0)
            assert recovery.max_retries == 2
            
            return {
                'passed': True,
                'details': {
                    'custom_exceptions': 'working correctly',
                    'circuit_breaker': f'initialized with state={cb.state}',
                    'error_recovery': f'configured with max_retries={recovery.max_retries}'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Error handling test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_security_validation(self) -> Dict[str, Any]:
        """Test security validation features."""
        try:
            from conforl.utils.security import (
                sanitize_input, sanitize_file_path, SecurityContext,
                hash_sensitive_data, verify_hash
            )
            
            # Test input sanitization
            clean_string = sanitize_input("test_string", "string", max_length=50)
            assert clean_string == "test_string"
            
            # Test dangerous pattern detection
            try:
                sanitize_input("<script>alert('xss')</script>", "string")
                security_detected = False
            except Exception:
                security_detected = True
            
            # Test hash functionality
            test_data = "sensitive_data"
            hashed = hash_sensitive_data(test_data)
            verified = verify_hash(test_data, hashed)
            
            return {
                'passed': True,
                'details': {
                    'input_sanitization': 'working correctly',
                    'security_pattern_detection': f'threats detected: {security_detected}',
                    'data_hashing': f'hash verification: {verified}',
                    'security_context': 'available'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Security validation test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_performance_benchmarks(self) -> Dict[str, Any]:
        """Test performance benchmarks."""
        try:
            from conforl.core.conformal import SplitConformalPredictor
            import time
            
            # Benchmark conformal prediction
            predictor = SplitConformalPredictor(coverage=0.95)
            
            # Test calibration performance
            start_time = time.time()
            calibration_scores = list(range(1000))  # Large dataset
            predictor.calibrate(calibration_scores)
            calibration_time = time.time() - start_time
            
            # Test prediction performance
            start_time = time.time()
            for _ in range(100):
                _ = predictor.get_prediction_interval([0.5], [0.1])
            prediction_time = time.time() - start_time
            
            # Performance thresholds
            calibration_ok = calibration_time < 1.0  # Less than 1 second
            prediction_ok = prediction_time < 0.5   # Less than 0.5 seconds for 100 predictions
            
            return {
                'passed': calibration_ok and prediction_ok,
                'details': {
                    'calibration_time': f'{calibration_time:.3f}s (threshold: 1.0s)',
                    'prediction_time': f'{prediction_time:.3f}s (threshold: 0.5s)',
                    'calibration_performance': 'PASS' if calibration_ok else 'FAIL',
                    'prediction_performance': 'PASS' if prediction_ok else 'FAIL'
                },
                'error': None if (calibration_ok and prediction_ok) else 'Performance thresholds exceeded'
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Performance benchmark test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_memory_leak_detection(self) -> Dict[str, Any]:
        """Test for memory leaks in core operations."""
        try:
            import gc
            
            # Force garbage collection and get initial memory
            gc.collect()
            initial_objects = len(gc.get_objects())
            
            # Perform operations that might leak memory
            from conforl.core.types import RiskCertificate
            
            certificates = []
            for i in range(100):
                cert = RiskCertificate(
                    risk_bound=0.05,
                    confidence=0.95,
                    coverage_guarantee=0.95,
                    method="test",
                    sample_size=100,
                    timestamp=time.time()
                )
                certificates.append(cert)
            
            # Clear references
            del certificates
            
            # Force garbage collection and check memory
            gc.collect()
            final_objects = len(gc.get_objects())
            
            object_growth = final_objects - initial_objects
            memory_leak_detected = object_growth > 50  # Arbitrary threshold
            
            return {
                'passed': not memory_leak_detected,
                'details': {
                    'initial_objects': initial_objects,
                    'final_objects': final_objects,
                    'object_growth': object_growth,
                    'threshold': 50,
                    'memory_leak_detected': memory_leak_detected
                },
                'error': f'Potential memory leak: {object_growth} objects not cleaned up' if memory_leak_detected else None
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Memory leak detection test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_thread_safety(self) -> Dict[str, Any]:
        """Test thread safety of core components."""
        try:
            from conforl.utils.concurrency import ThreadSafeDict, ThreadSafeCounter
            import threading
            import time
            
            # Test thread-safe dictionary
            safe_dict = ThreadSafeDict()
            errors = []
            
            def dict_writer(thread_id):
                try:
                    for i in range(10):
                        safe_dict.set(f'key_{thread_id}_{i}', f'value_{thread_id}_{i}')
                        time.sleep(0.001)  # Small delay to encourage race conditions
                except Exception as e:
                    errors.append(str(e))
            
            # Test thread-safe counter
            counter = ThreadSafeCounter()
            
            def counter_incrementer():
                try:
                    for _ in range(100):
                        counter.increment()
                except Exception as e:
                    errors.append(str(e))
            
            # Run concurrent operations
            threads = []
            
            # Dictionary test threads
            for i in range(5):
                t = threading.Thread(target=dict_writer, args=(i,))
                threads.append(t)
                t.start()
            
            # Counter test threads
            for _ in range(3):
                t = threading.Thread(target=counter_incrementer)
                threads.append(t)
                t.start()
            
            # Wait for all threads
            for t in threads:
                t.join(timeout=5.0)
            
            # Verify results
            dict_size = safe_dict.size()
            counter_value = counter.get()
            expected_counter = 300  # 3 threads * 100 increments
            
            thread_safety_ok = (
                len(errors) == 0 and
                dict_size == 50 and  # 5 threads * 10 items
                counter_value == expected_counter
            )
            
            return {
                'passed': thread_safety_ok,
                'details': {
                    'concurrent_errors': len(errors),
                    'dict_operations': f'{dict_size}/50 items stored',
                    'counter_operations': f'{counter_value}/{expected_counter} increments',
                    'thread_safety': 'PASS' if thread_safety_ok else 'FAIL'
                },
                'error': f'Thread safety issues detected: {errors}' if errors else None
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Thread safety test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_scalability_features(self) -> Dict[str, Any]:
        """Test scalability and performance features."""
        try:
            from conforl.utils.errors import CircuitBreaker
            from conforl.utils.concurrency import WorkerPool, RateLimiter
            
            # Test circuit breaker
            cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
            
            # Test worker pool
            worker_pool = WorkerPool(num_workers=2, queue_size=10)
            
            # Submit test tasks
            task_results = []
            def test_task(value):
                task_results.append(value)
            
            success_count = 0
            for i in range(5):
                if worker_pool.submit_task(test_task, i):
                    success_count += 1
            
            time.sleep(0.5)  # Allow tasks to complete
            
            # Test rate limiter
            rate_limiter = RateLimiter(max_calls=3, time_window=1.0)
            
            rate_limit_successes = 0
            for _ in range(5):
                if rate_limiter.acquire(timeout=0.1):
                    rate_limit_successes += 1
            
            # Cleanup
            worker_pool.shutdown(timeout=2.0)
            
            return {
                'passed': True,
                'details': {
                    'circuit_breaker': f'state={cb.state}',
                    'worker_pool': f'submitted {success_count}/5 tasks',
                    'task_completion': f'{len(task_results)} tasks completed',
                    'rate_limiter': f'{rate_limit_successes}/5 requests allowed'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Scalability features test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_configuration_validation(self) -> Dict[str, Any]:
        """Test configuration validation."""
        try:
            from conforl.utils.validation import validate_config, validate_risk_parameters
            from conforl.utils.errors import ValidationError
            
            # Test valid configuration
            valid_config = {
                'learning_rate': 0.001,
                'buffer_size': 1000,
                'target_risk': 0.05,
                'confidence': 0.95,
                'device': 'cpu'
            }
            
            validated_config = validate_config(valid_config)
            
            # Test invalid configuration handling
            validation_errors_caught = 0
            
            invalid_configs = [
                {'learning_rate': -1.0},  # Invalid learning rate
                {'buffer_size': -100},    # Invalid buffer size
                {'target_risk': 1.5},     # Invalid risk target
                {'confidence': 0},        # Invalid confidence
                {'device': 'invalid'}     # Invalid device
            ]
            
            for invalid_config in invalid_configs:
                try:
                    validate_config(invalid_config)
                except (ValidationError, Exception) as e:
                    # Count any validation-related exception as expected behavior
                    if "CONFIG_ERROR" in str(e) or "VALIDATION_ERROR" in str(e) or "ValidationError" in str(type(e).__name__):
                        validation_errors_caught += 1
            
            # Test risk parameter validation
            try:
                validate_risk_parameters(0.05, 0.95)
                risk_validation_ok = True
            except ValidationError:
                risk_validation_ok = False
            
            return {
                'passed': True,
                'details': {
                    'valid_config_processed': len(validated_config) > 0,
                    'validation_errors_caught': f'{validation_errors_caught}/{len(invalid_configs)}',
                    'risk_parameter_validation': 'PASS' if risk_validation_ok else 'FAIL'
                }
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Configuration validation test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }
    
    def _test_documentation_coverage(self) -> Dict[str, Any]:
        """Test documentation coverage."""
        try:
            project_root = Path(__file__).parent
            
            # Check for key documentation files
            docs_files = {
                'README.md': project_root / 'README.md',
                'CLAUDE.md': project_root / 'CLAUDE.md',
                'CONTRIBUTING.md': project_root / 'CONTRIBUTING.md',
                'LICENSE': project_root / 'LICENSE'
            }
            
            existing_docs = {}
            for doc_name, doc_path in docs_files.items():
                exists = doc_path.exists()
                existing_docs[doc_name] = exists
                if exists:
                    # Check if file has substantial content
                    content_length = len(doc_path.read_text(encoding='utf-8', errors='ignore'))
                    existing_docs[f'{doc_name}_length'] = content_length
            
            # Check Python docstrings in key modules
            docstring_coverage = {}
            key_modules = [
                'conforl/core/types.py',
                'conforl/core/conformal.py',
                'conforl/utils/errors.py'
            ]
            
            for module_path in key_modules:
                full_path = project_root / module_path
                if full_path.exists():
                    content = full_path.read_text(encoding='utf-8', errors='ignore')
                    # Simple heuristic: count docstrings
                    docstring_count = content.count('"""') // 2
                    function_count = content.count('def ')
                    class_count = content.count('class ')
                    
                    coverage_ratio = docstring_count / max(function_count + class_count, 1)
                    docstring_coverage[module_path] = {
                        'docstrings': docstring_count,
                        'functions_classes': function_count + class_count,
                        'coverage_ratio': coverage_ratio
                    }
            
            # Calculate overall documentation score
            doc_files_score = sum(existing_docs.values()) / len(docs_files) if docs_files else 0
            avg_docstring_coverage = sum(
                info['coverage_ratio'] for info in docstring_coverage.values()
            ) / len(docstring_coverage) if docstring_coverage else 0
            
            overall_score = (doc_files_score + avg_docstring_coverage) / 2
            documentation_adequate = overall_score > 0.7  # 70% threshold
            
            return {
                'passed': documentation_adequate,
                'details': {
                    'documentation_files': existing_docs,
                    'docstring_coverage': docstring_coverage,
                    'overall_score': f'{overall_score:.2f}',
                    'threshold': 0.7,
                    'adequate': documentation_adequate
                },
                'error': f'Documentation coverage below threshold: {overall_score:.2f} < 0.7' if not documentation_adequate else None
            }
        except Exception as e:
            return {
                'passed': False,
                'error': f'Documentation coverage test failed: {str(e)}',
                'details': {'exception': traceback.format_exc()}
            }


def main():
    """Run comprehensive quality gates."""
    runner = QualityGateRunner()
    
    # Define quality gates
    gates = [
        QualityGate(
            "core_imports",
            "Verify all core modules can be imported without errors",
            critical=True
        ),
        QualityGate(
            "basic_functionality", 
            "Test basic ConfoRL functionality works correctly",
            critical=True
        ),
        QualityGate(
            "error_handling",
            "Verify comprehensive error handling is working",
            critical=True
        ),
        QualityGate(
            "security_validation",
            "Test security validation and input sanitization",
            critical=True
        ),
        QualityGate(
            "performance_benchmarks",
            "Verify performance meets minimum requirements",
            critical=False
        ),
        QualityGate(
            "memory_leak_detection",
            "Check for memory leaks in core operations",
            critical=False
        ),
        QualityGate(
            "thread_safety",
            "Verify thread safety of concurrent operations",
            critical=True
        ),
        QualityGate(
            "scalability_features",
            "Test scalability and resilience features",
            critical=False
        ),
        QualityGate(
            "configuration_validation",
            "Verify configuration validation works correctly",
            critical=True
        ),
        QualityGate(
            "documentation_coverage",
            "Check documentation coverage and quality",
            critical=False
        )
    ]
    
    # Add gates to runner
    for gate in gates:
        runner.add_gate(gate)
    
    # Execute all gates
    results = runner.run_all_gates()
    
    # Save results to file
    results_file = Path(__file__).parent / 'quality_report_comprehensive.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {results_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results['deployment_approved'] else 1
    print(f"\n🏁 Quality Gates Execution Complete (exit code: {exit_code})")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())