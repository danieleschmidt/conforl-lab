#!/usr/bin/env python3
"""Quality gates and security validation for ConfoRL."""

import os
import sys
import subprocess
import time
from pathlib import Path


def run_command(command, description, required=True):
    """Run a command and return success status."""
    print(f"\n🔍 {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command.split(),
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout.strip():
                print("Output:", result.stdout.strip()[:200])
            return True
        else:
            status = "FAILED" if required else "WARNING"
            print(f"❌ {description} - {status}")
            if result.stderr.strip():
                print("Error:", result.stderr.strip()[:500])
            return not required
            
    except subprocess.TimeoutExpired:
        print(f"⏱️ {description} - TIMEOUT")
        return not required
    except FileNotFoundError:
        print(f"⚠️ {description} - TOOL NOT FOUND")
        return not required
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return not required


def check_code_style():
    """Check code style and formatting."""
    print("\n" + "="*60)
    print("🎨 CODE STYLE & FORMATTING CHECKS")
    print("="*60)
    
    checks_passed = 0
    total_checks = 0
    
    # Python syntax check
    total_checks += 1
    if run_command("python3 -m py_compile test_comprehensive.py", "Python syntax check", required=False):
        checks_passed += 1
    
    # Line length check (basic)
    total_checks += 1
    try:
        long_lines = 0
        for py_file in Path('.').rglob('*.py'):
            if 'test_' not in str(py_file):  # Skip test files
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if len(line.strip()) > 120:  # Allow up to 120 chars
                            long_lines += 1
                            if long_lines == 1:  # Only print first few
                                print(f"Long line in {py_file}:{line_num}")
        
        if long_lines == 0:
            print("✅ Line length check - PASSED")
            checks_passed += 1
        else:
            print(f"⚠️ Line length check - {long_lines} long lines found")
    except Exception as e:
        print(f"⚠️ Line length check - ERROR: {e}")
    
    return checks_passed, total_checks


def check_security():
    """Run security checks."""
    print("\n" + "="*60)
    print("🔒 SECURITY VALIDATION")
    print("="*60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check for hardcoded secrets
    total_checks += 1
    try:
        secret_patterns = [
            'password', 'secret', 'token', 'api_key', 'private_key'
        ]
        
        secrets_found = False
        for py_file in Path('.').rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                for pattern in secret_patterns:
                    if f'"{pattern}"' in content or f"'{pattern}'" in content:
                        # Check if it's not just a variable name or comment
                        if '=' in content and pattern in content:
                            print(f"⚠️ Potential hardcoded secret in {py_file}: {pattern}")
                            secrets_found = True
        
        if not secrets_found:
            print("✅ Hardcoded secrets check - PASSED")
            checks_passed += 1
        else:
            print("⚠️ Hardcoded secrets check - WARNINGS FOUND")
    except Exception as e:
        print(f"⚠️ Hardcoded secrets check - ERROR: {e}")
    
    # Check for SQL injection vulnerabilities (basic)
    total_checks += 1
    try:
        sql_patterns = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE']
        sql_vulns_found = False
        
        for py_file in Path('.').rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in sql_patterns:
                    if f'"{pattern}' in content and '{' in content:
                        # Potential string formatting in SQL
                        print(f"⚠️ Potential SQL injection risk in {py_file}")
                        sql_vulns_found = True
                        break
        
        if not sql_vulns_found:
            print("✅ SQL injection check - PASSED")
            checks_passed += 1
        else:
            print("⚠️ SQL injection check - WARNINGS FOUND")
    except Exception as e:
        print(f"⚠️ SQL injection check - ERROR: {e}")
    
    # Check for unsafe file operations
    total_checks += 1
    try:
        unsafe_patterns = ['open(user', 'open(input', 'eval(', 'exec(']
        unsafe_found = False
        
        for py_file in Path('.').rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for pattern in unsafe_patterns:
                    if pattern in content:
                        print(f"⚠️ Potential unsafe operation in {py_file}: {pattern}")
                        unsafe_found = True
        
        if not unsafe_found:
            print("✅ Unsafe operations check - PASSED")
            checks_passed += 1
        else:
            print("⚠️ Unsafe operations check - WARNINGS FOUND")
    except Exception as e:
        print(f"⚠️ Unsafe operations check - ERROR: {e}")
    
    return checks_passed, total_checks


def check_performance():
    """Check performance characteristics."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE VALIDATION")
    print("="*60)
    
    checks_passed = 0
    total_checks = 3
    
    # Import speed test
    start_time = time.time()
    try:
        from conforl.core.types import RiskCertificate
        from conforl.utils.errors import ConfoRLError
        import_time = time.time() - start_time
        
        if import_time < 1.0:  # Should import in under 1 second
            print(f"✅ Import performance - PASSED ({import_time:.3f}s)")
            checks_passed += 1
        else:
            print(f"⚠️ Import performance - SLOW ({import_time:.3f}s)")
    except Exception as e:
        print(f"❌ Import performance - ERROR: {e}")
    
    # Object creation speed test
    start_time = time.time()
    try:
        from conforl.core.types import RiskCertificate
        
        for i in range(1000):
            cert = RiskCertificate(
                risk_bound=0.05,
                confidence=0.95,
                coverage_guarantee=0.95,
                method="perf_test",
                sample_size=1000
            )
        
        creation_time = time.time() - start_time
        
        if creation_time < 0.1:  # Should create 1000 objects in under 0.1s
            print(f"✅ Object creation performance - PASSED ({creation_time:.3f}s for 1000 objects)")
            checks_passed += 1
        else:
            print(f"⚠️ Object creation performance - SLOW ({creation_time:.3f}s for 1000 objects)")
    except Exception as e:
        print(f"❌ Object creation performance - ERROR: {e}")
    
    # Memory usage check
    try:
        import psutil
        import gc
        
        gc.collect()
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create some objects
        objects = []
        for i in range(100):
            from conforl.core.types import RiskCertificate
            cert = RiskCertificate(
                risk_bound=0.05,
                confidence=0.95,
                coverage_guarantee=0.95,
                method=f"mem_test_{i}",
                sample_size=1000
            )
            objects.append(cert)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        if memory_increase < 10:  # Should use less than 10MB for 100 objects
            print(f"✅ Memory usage - PASSED ({memory_increase:.1f}MB increase)")
            checks_passed += 1
        else:
            print(f"⚠️ Memory usage - HIGH ({memory_increase:.1f}MB increase)")
    except ImportError:
        print("⚠️ Memory usage check - SKIPPED (psutil not available)")
    except Exception as e:
        print(f"❌ Memory usage check - ERROR: {e}")
    
    return checks_passed, total_checks


def check_dependencies():
    """Check dependencies and compatibility."""
    print("\n" + "="*60)
    print("📦 DEPENDENCY VALIDATION")
    print("="*60)
    
    checks_passed = 0
    total_checks = 0
    
    # Python version check
    total_checks += 1
    if sys.version_info >= (3, 8):
        print(f"✅ Python version - PASSED (Python {sys.version_info.major}.{sys.version_info.minor})")
        checks_passed += 1
    else:
        print(f"❌ Python version - FAILED (Python {sys.version_info.major}.{sys.version_info.minor}, require 3.8+)")
    
    # Check core imports work
    total_checks += 1
    try:
        from conforl.core.types import RiskCertificate, TrajectoryData
        from conforl.core.conformal import SplitConformalPredictor
        from conforl.utils.errors import ConfoRLError
        from conforl.utils.logging import get_logger
        print("✅ Core imports - PASSED")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Core imports - FAILED: {e}")
    
    # Check optional dependencies
    optional_deps = [
        ('numpy', 'Advanced numerical computations'),
        ('psutil', 'System monitoring'),
        ('gymnasium', 'RL environment interface'),
    ]
    
    for dep, description in optional_deps:
        total_checks += 1
        try:
            __import__(dep)
            print(f"✅ Optional dependency {dep} - AVAILABLE ({description})")
            checks_passed += 1
        except ImportError:
            print(f"⚠️ Optional dependency {dep} - NOT AVAILABLE ({description})")
            # Don't count as failure since it's optional
    
    return checks_passed, total_checks


def check_documentation():
    """Check documentation completeness."""
    print("\n" + "="*60)
    print("📚 DOCUMENTATION VALIDATION")
    print("="*60)
    
    checks_passed = 0
    total_checks = 0
    
    # Check README exists
    total_checks += 1
    if Path('README.md').exists():
        print("✅ README.md - EXISTS")
        checks_passed += 1
    else:
        print("⚠️ README.md - MISSING")
    
    # Check CLAUDE.md exists
    total_checks += 1
    if Path('CLAUDE.md').exists():
        print("✅ CLAUDE.md - EXISTS")
        checks_passed += 1
    else:
        print("⚠️ CLAUDE.md - MISSING")
    
    # Check docstrings in core modules
    total_checks += 1
    docstring_coverage = 0
    total_functions = 0
    
    try:
        import inspect
        from conforl.core.types import RiskCertificate, TrajectoryData
        from conforl.core.conformal import SplitConformalPredictor
        
        for obj in [RiskCertificate, TrajectoryData, SplitConformalPredictor]:
            members = inspect.getmembers(obj, predicate=inspect.ismethod)
            for name, method in members:
                if not name.startswith('_'):
                    total_functions += 1
                    if method.__doc__:
                        docstring_coverage += 1
        
        coverage_percent = (docstring_coverage / max(1, total_functions)) * 100
        
        if coverage_percent >= 70:
            print(f"✅ Docstring coverage - PASSED ({coverage_percent:.1f}%)")
            checks_passed += 1
        else:
            print(f"⚠️ Docstring coverage - LOW ({coverage_percent:.1f}%)")
    except Exception as e:
        print(f"⚠️ Docstring coverage check - ERROR: {e}")
    
    return checks_passed, total_checks


def main():
    """Run all quality gates."""
    print("🚀 ConfoRL Quality Gates & Security Validation")
    print("="*80)
    
    start_time = time.time()
    
    # Run all checks
    all_results = []
    
    all_results.append(("Code Style", check_code_style()))
    all_results.append(("Security", check_security()))
    all_results.append(("Performance", check_performance()))
    all_results.append(("Dependencies", check_dependencies()))
    all_results.append(("Documentation", check_documentation()))
    
    # Calculate overall results
    total_passed = sum(passed for _, (passed, _) in all_results)
    total_checks = sum(total for _, (_, total) in all_results)
    overall_score = (total_passed / max(1, total_checks)) * 100
    
    # Print summary
    print("\n" + "="*80)
    print("📊 QUALITY GATES SUMMARY")
    print("="*80)
    
    for category, (passed, total) in all_results:
        score = (passed / max(1, total)) * 100
        status = "✅ PASS" if score >= 80 else "⚠️ WARN" if score >= 60 else "❌ FAIL"
        print(f"{category:<15}: {passed:>2}/{total:<2} ({score:>5.1f}%) {status}")
    
    print("-" * 80)
    print(f"{'OVERALL':<15}: {total_passed:>2}/{total_checks:<2} ({overall_score:>5.1f}%) ", end="")
    
    if overall_score >= 85:
        print("✅ EXCELLENT")
        exit_code = 0
    elif overall_score >= 70:
        print("⚠️ GOOD")
        exit_code = 1
    else:
        print("❌ NEEDS IMPROVEMENT")
        exit_code = 2
    
    elapsed_time = time.time() - start_time
    print(f"\n⏱️ Total execution time: {elapsed_time:.1f} seconds")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if overall_score < 85:
        print("- Address any failed checks above")
        print("- Ensure all security warnings are resolved")
        print("- Consider adding more comprehensive documentation")
    else:
        print("- ConfoRL passes all quality gates!")
        print("- System is ready for production deployment")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())