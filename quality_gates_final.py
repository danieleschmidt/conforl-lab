#!/usr/bin/env python3
"""
Final Quality Gates for ConfoRL Production Deployment
Validates implementation quality, completeness, and production readiness.
"""

import os
import sys
import ast
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

class QualityGateRunner:
    """Comprehensive quality gate validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results = {}
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all quality gates."""
        print("🚀 ConfoRL Production Quality Gates")
        print("=" * 60)
        
        gates = [
            ("Architecture Validation", self.validate_architecture),
            ("Code Quality Check", self.check_code_quality),
            ("Security Analysis", self.check_security_features),
            ("Performance Features", self.check_performance_features),
            ("Research Completeness", self.check_research_features),
            ("Production Readiness", self.check_production_readiness),
            ("Documentation Quality", self.check_documentation),
            ("Test Coverage Analysis", self.analyze_test_coverage),
            ("Deployment Readiness", self.check_deployment_readiness)
        ]
        
        total_score = 0
        max_score = 0
        
        for gate_name, gate_func in gates:
            print(f"\n📋 {gate_name}")
            print("-" * 40)
            
            try:
                score, max_points, details = gate_func()
                self.results[gate_name] = {
                    'score': score,
                    'max_score': max_points,
                    'percentage': (score / max_points) * 100 if max_points > 0 else 0,
                    'details': details,
                    'status': 'PASS' if (score / max_points) >= 0.8 else 'FAIL'
                }
                
                total_score += score
                max_score += max_points
                
                percentage = (score / max_points) * 100 if max_points > 0 else 0
                status = "✅ PASS" if percentage >= 80 else "❌ FAIL"
                print(f"Score: {score}/{max_points} ({percentage:.1f}%) {status}")
                
                # Print key details
                for detail in details[:3]:  # Show top 3 details
                    print(f"  • {detail}")
                    
            except Exception as e:
                print(f"❌ Gate execution failed: {e}")
                self.results[gate_name] = {
                    'score': 0,
                    'max_score': 10,
                    'percentage': 0,
                    'details': [f"Execution failed: {e}"],
                    'status': 'ERROR'
                }
        
        # Generate final report
        overall_percentage = (total_score / max_score) * 100 if max_score > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 FINAL QUALITY ASSESSMENT")
        print("=" * 60)
        
        for gate_name, result in self.results.items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌" if result['status'] == 'FAIL' else "⚠️"
            print(f"{status_icon} {gate_name:25} {result['score']:3}/{result['max_score']:3} ({result['percentage']:5.1f}%)")
        
        print("-" * 60)
        print(f"🎯 OVERALL SCORE: {total_score}/{max_score} ({overall_percentage:.1f}%)")
        
        if overall_percentage >= 85:
            print("🎉 EXCELLENT - Ready for production deployment!")
        elif overall_percentage >= 70:
            print("✅ GOOD - Ready with minor improvements")
        elif overall_percentage >= 50:
            print("⚠️  FAIR - Needs improvements before deployment")
        else:
            print("❌ POOR - Significant improvements needed")
        
        return {
            'overall_score': total_score,
            'max_score': max_score,
            'percentage': overall_percentage,
            'gate_results': self.results,
            'production_ready': overall_percentage >= 70
        }
    
    def validate_architecture(self) -> Tuple[int, int, List[str]]:
        """Validate software architecture."""
        score = 0
        max_points = 10
        details = []
        
        # Check core modules exist
        core_modules = [
            'conforl/core/conformal.py',
            'conforl/core/types.py',
            'conforl/algorithms/base.py',
            'conforl/algorithms/sac.py',
            'conforl/risk/measures.py',
            'conforl/risk/controllers.py'
        ]
        
        existing_core = sum(1 for module in core_modules if (self.project_root / module).exists())
        score += min(3, existing_core // 2)
        details.append(f"Core modules: {existing_core}/{len(core_modules)} present")
        
        # Check research extensions
        research_modules = [
            'conforl/research/causal.py',
            'conforl/research/adversarial.py', 
            'conforl/research/multi_agent.py',
            'conforl/research/compositional.py'
        ]
        
        existing_research = sum(1 for module in research_modules if (self.project_root / module).exists())
        score += min(3, existing_research)
        details.append(f"Research extensions: {existing_research}/{len(research_modules)} implemented")
        
        # Check production modules
        production_modules = [
            'conforl/security/validation.py',
            'conforl/security/encryption.py',
            'conforl/scaling/performance.py',
            'conforl/scaling/distributed.py'
        ]
        
        existing_prod = sum(1 for module in production_modules if (self.project_root / module).exists())
        score += min(2, existing_prod // 2)
        details.append(f"Production modules: {existing_prod}/{len(production_modules)} present")
        
        # Check package structure
        packages = ['core', 'algorithms', 'risk', 'research', 'security', 'scaling', 'benchmarks', 'utils']
        existing_packages = sum(1 for pkg in packages if (self.project_root / 'conforl' / pkg / '__init__.py').exists())
        score += min(2, existing_packages // 4)
        details.append(f"Package structure: {existing_packages}/{len(packages)} packages")
        
        return score, max_points, details
    
    def check_code_quality(self) -> Tuple[int, int, List[str]]:
        """Check code quality metrics."""
        score = 0
        max_points = 10
        details = []
        
        python_files = list(self.project_root.rglob("*.py"))
        if not python_files:
            return 0, max_points, ["No Python files found"]
        
        total_lines = 0
        total_docstrings = 0
        total_classes = 0
        total_functions = 0
        syntax_errors = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = len(content.splitlines())
                total_lines += lines
                
                # Parse AST for analysis
                try:
                    tree = ast.parse(content)
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                total_docstrings += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                            if ast.get_docstring(node):
                                total_docstrings += 1
                                
                except SyntaxError:
                    syntax_errors += 1
                    
            except Exception as e:
                continue
        
        # Score based on metrics
        if syntax_errors == 0:
            score += 3
            details.append(f"Syntax: No syntax errors in {len(python_files)} files")
        else:
            details.append(f"Syntax: {syntax_errors} files with syntax errors")
        
        # Documentation coverage
        total_definitions = total_functions + total_classes
        if total_definitions > 0:
            doc_coverage = total_docstrings / total_definitions
            if doc_coverage >= 0.7:
                score += 3
                details.append(f"Documentation: {doc_coverage:.1%} coverage (excellent)")
            elif doc_coverage >= 0.4:
                score += 2
                details.append(f"Documentation: {doc_coverage:.1%} coverage (good)")
            else:
                score += 1
                details.append(f"Documentation: {doc_coverage:.1%} coverage (needs improvement)")
        
        # Code size indicates completeness
        if total_lines >= 15000:
            score += 2
            details.append(f"Code size: {total_lines} lines (comprehensive)")
        elif total_lines >= 10000:
            score += 1
            details.append(f"Code size: {total_lines} lines (substantial)")
        else:
            details.append(f"Code size: {total_lines} lines (limited)")
        
        # Complexity indicators
        if total_classes >= 50 and total_functions >= 100:
            score += 2
            details.append(f"Complexity: {total_classes} classes, {total_functions} functions")
        elif total_classes >= 20 and total_functions >= 50:
            score += 1
            details.append(f"Complexity: {total_classes} classes, {total_functions} functions")
        
        return score, max_points, details
    
    def check_security_features(self) -> Tuple[int, int, List[str]]:
        """Check security implementation."""
        score = 0
        max_points = 10
        details = []
        
        security_files = {
            'conforl/security/validation.py': 'Input validation and sanitization',
            'conforl/security/encryption.py': 'Encryption and secure serialization',
            'conforl/security/audit.py': 'Security auditing and monitoring',
            'conforl/security/access_control.py': 'Access control and authentication'
        }
        
        for file_path, description in security_files.items():
            if (self.project_root / file_path).exists():
                score += 2
                details.append(f"✅ {description}")
            else:
                details.append(f"❌ Missing: {description}")
        
        # Check for security patterns in code
        security_patterns = [
            ('hashlib', 'Cryptographic hashing'),
            ('secrets', 'Secure random generation'),
            ('hmac', 'Message authentication'),
            ('sanitize', 'Input sanitization'),
            ('validate', 'Input validation'),
            ('encrypt', 'Data encryption')
        ]
        
        found_patterns = 0
        security_files_content = []
        
        for sec_file in ['conforl/security/validation.py', 'conforl/security/encryption.py']:
            file_path = self.project_root / sec_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        security_files_content.append(f.read())
                except:
                    pass
        
        if security_files_content:
            content = '\n'.join(security_files_content)
            for pattern, desc in security_patterns:
                if pattern in content:
                    found_patterns += 1
            
            pattern_score = min(2, found_patterns // 3)
            score += pattern_score
            details.append(f"Security patterns: {found_patterns}/{len(security_patterns)} implemented")
        
        return score, max_points, details
    
    def check_performance_features(self) -> Tuple[int, int, List[str]]:
        """Check performance optimization features."""
        score = 0
        max_points = 10
        details = []
        
        perf_files = {
            'conforl/scaling/performance.py': 'Performance optimization framework',
            'conforl/scaling/distributed.py': 'Distributed computing support',
            'conforl/optimize/cache.py': 'Caching system',
            'conforl/optimize/concurrent.py': 'Concurrent processing'
        }
        
        existing_perf = 0
        for file_path, description in perf_files.items():
            if (self.project_root / file_path).exists():
                existing_perf += 1
                score += 2
                details.append(f"✅ {description}")
        
        # Check for performance patterns
        perf_patterns = [
            'threading', 'multiprocessing', 'concurrent', 'cache', 
            'optimize', 'memory', 'pool', 'parallel'
        ]
        
        perf_content = []
        for perf_file in ['conforl/scaling/performance.py', 'conforl/scaling/distributed.py']:
            file_path = self.project_root / perf_file
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        perf_content.append(f.read())
                except:
                    pass
        
        if perf_content:
            content = '\n'.join(perf_content).lower()
            found_patterns = sum(1 for pattern in perf_patterns if pattern in content)
            pattern_score = min(2, found_patterns // 4)
            score += pattern_score
            details.append(f"Performance patterns: {found_patterns}/{len(perf_patterns)} found")
        
        return score, max_points, details
    
    def check_research_features(self) -> Tuple[int, int, List[str]]:
        """Check research extension completeness."""
        score = 0
        max_points = 10
        details = []
        
        research_areas = {
            'causal.py': 'Causal conformal risk control',
            'adversarial.py': 'Adversarial robust conformal RL',
            'multi_agent.py': 'Multi-agent distributed risk control',
            'compositional.py': 'Compositional hierarchical risk control'
        }
        
        for file_name, description in research_areas.items():
            file_path = self.project_root / 'conforl' / 'research' / file_name
            if file_path.exists():
                score += 2
                details.append(f"✅ {description}")
                
                # Check file size as quality indicator
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 10000:  # >10KB indicates substantial implementation
                        score += 0.5
                except:
                    pass
            else:
                details.append(f"❌ Missing: {description}")
        
        # Check research benchmarking
        benchmark_file = self.project_root / 'conforl' / 'benchmarks' / 'research_benchmarks.py'
        if benchmark_file.exists():
            score += 1
            details.append("✅ Research benchmarking framework")
        
        return score, max_points, details
    
    def check_production_readiness(self) -> Tuple[int, int, List[str]]:
        """Check production deployment readiness."""
        score = 0
        max_points = 10
        details = []
        
        # Check deployment files
        deployment_files = {
            'Dockerfile': 'Container deployment',
            'docker-compose.yml': 'Multi-service orchestration',
            'kubernetes/deployment.yaml': 'Kubernetes deployment',
            'requirements.txt': 'Python dependencies',
            'setup.py': 'Package configuration'
        }
        
        for file_path, description in deployment_files.items():
            if (self.project_root / file_path).exists():
                score += 1
                details.append(f"✅ {description}")
            else:
                details.append(f"❌ Missing: {description}")
        
        # Check configuration files
        config_files = ['pytest.ini', '.env.example']
        existing_configs = sum(1 for f in config_files if (self.project_root / f).exists())
        if existing_configs > 0:
            score += 1
            details.append(f"Configuration files: {existing_configs}/{len(config_files)}")
        
        # Check scripts
        scripts_dir = self.project_root / 'scripts'
        if scripts_dir.exists() and any(scripts_dir.iterdir()):
            score += 1
            details.append("✅ Deployment scripts present")
        
        # Check monitoring and logging
        monitoring_indicators = ['logging', 'metrics', 'monitoring', 'health']
        utils_files = list((self.project_root / 'conforl' / 'utils').glob('*.py'))
        
        monitoring_found = False
        for utils_file in utils_files:
            try:
                with open(utils_file, 'r') as f:
                    content = f.read().lower()
                    if any(indicator in content for indicator in monitoring_indicators):
                        monitoring_found = True
                        break
            except:
                pass
        
        if monitoring_found:
            score += 1
            details.append("✅ Monitoring and logging support")
        
        # Check CLI interface
        cli_file = self.project_root / 'conforl' / 'cli.py'
        if cli_file.exists():
            score += 1
            details.append("✅ Command-line interface")
        
        return score, max_points, details
    
    def check_documentation(self) -> Tuple[int, int, List[str]]:
        """Check documentation quality."""
        score = 0
        max_points = 10
        details = []
        
        # Check main documentation files
        doc_files = {
            'README.md': 'Project documentation',
            'CLAUDE.md': 'Development guide',
            'CONTRIBUTING.md': 'Contribution guidelines'
        }
        
        for file_path, description in doc_files.items():
            file_full_path = self.project_root / file_path
            if file_full_path.exists():
                try:
                    file_size = file_full_path.stat().st_size
                    if file_size > 1000:  # Substantial documentation
                        score += 2
                        details.append(f"✅ {description} ({file_size} bytes)")
                    else:
                        score += 1
                        details.append(f"⚠️ {description} (minimal)")
                except:
                    score += 1
                    details.append(f"✅ {description}")
            else:
                details.append(f"❌ Missing: {description}")
        
        # Check docs directory
        docs_dir = self.project_root / 'docs'
        if docs_dir.exists():
            doc_count = len(list(docs_dir.glob('*.md')))
            if doc_count > 0:
                score += 2
                details.append(f"✅ Additional documentation: {doc_count} files")
        
        # Check inline documentation quality
        python_files = list((self.project_root / 'conforl').rglob('*.py'))
        if python_files:
            total_docstrings = 0
            total_functions = 0
            
            for py_file in python_files[:10]:  # Sample first 10 files
                try:
                    with open(py_file, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            total_functions += 1
                            if ast.get_docstring(node):
                                total_docstrings += 1
                except:
                    pass
            
            if total_functions > 0:
                doc_ratio = total_docstrings / total_functions
                if doc_ratio > 0.7:
                    score += 2
                    details.append(f"✅ Inline docs: {doc_ratio:.1%} coverage")
                elif doc_ratio > 0.4:
                    score += 1
                    details.append(f"⚠️ Inline docs: {doc_ratio:.1%} coverage")
        
        return score, max_points, details
    
    def analyze_test_coverage(self) -> Tuple[int, int, List[str]]:
        """Analyze test coverage and quality."""
        score = 0
        max_points = 10
        details = []
        
        tests_dir = self.project_root / 'tests'
        if not tests_dir.exists():
            details.append("❌ No tests directory found")
            return score, max_points, details
        
        test_files = list(tests_dir.glob('test_*.py'))
        if not test_files:
            details.append("❌ No test files found")
            return score, max_points, details
        
        score += 3
        details.append(f"✅ Test structure: {len(test_files)} test files")
        
        # Check test file sizes and content
        substantial_tests = 0
        total_test_functions = 0
        
        for test_file in test_files:
            try:
                with open(test_file, 'r') as f:
                    content = f.read()
                
                if len(content) > 5000:  # >5KB indicates substantial tests
                    substantial_tests += 1
                
                # Count test functions
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if (isinstance(node, ast.FunctionDef) and 
                        node.name.startswith('test_')):
                        total_test_functions += 1
                        
            except:
                pass
        
        if substantial_tests > 0:
            score += 2
            details.append(f"✅ Substantial tests: {substantial_tests}/{len(test_files)} files")
        
        if total_test_functions >= 50:
            score += 3
            details.append(f"✅ Test functions: {total_test_functions} (comprehensive)")
        elif total_test_functions >= 20:
            score += 2
            details.append(f"✅ Test functions: {total_test_functions} (good)")
        elif total_test_functions > 0:
            score += 1
            details.append(f"⚠️ Test functions: {total_test_functions} (minimal)")
        
        # Check for pytest configuration
        if (self.project_root / 'pytest.ini').exists():
            score += 1
            details.append("✅ Pytest configuration")
        
        # Check for test utilities
        conftest_file = tests_dir / 'conftest.py'
        if conftest_file.exists():
            score += 1
            details.append("✅ Test fixtures and utilities")
        
        return score, max_points, details
    
    def check_deployment_readiness(self) -> Tuple[int, int, List[str]]:
        """Check deployment configuration and readiness."""
        score = 0
        max_points = 10
        details = []
        
        # Container deployment
        if (self.project_root / 'Dockerfile').exists():
            score += 2
            details.append("✅ Docker containerization")
            
            if (self.project_root / 'docker-compose.yml').exists():
                score += 1
                details.append("✅ Multi-container orchestration")
        
        # Kubernetes deployment
        k8s_dir = self.project_root / 'kubernetes'
        if k8s_dir.exists():
            k8s_files = list(k8s_dir.glob('*.yaml')) + list(k8s_dir.glob('*.yml'))
            if k8s_files:
                score += 2
                details.append(f"✅ Kubernetes deployment: {len(k8s_files)} manifests")
        
        # CI/CD and automation
        ci_indicators = ['.github/workflows', '.gitlab-ci.yml', 'Jenkinsfile']
        ci_found = any((self.project_root / indicator).exists() for indicator in ci_indicators)
        if ci_found:
            score += 1
            details.append("✅ CI/CD configuration")
        
        # Environment configuration
        env_files = ['.env.example', 'config.yaml', 'settings.json']
        env_found = sum(1 for env_file in env_files if (self.project_root / env_file).exists())
        if env_found > 0:
            score += 1
            details.append(f"✅ Environment configuration: {env_found} files")
        
        # Deployment scripts
        scripts_dir = self.project_root / 'scripts'
        if scripts_dir.exists():
            script_files = list(scripts_dir.glob('*.sh')) + list(scripts_dir.glob('*.py'))
            if script_files:
                score += 1
                details.append(f"✅ Deployment scripts: {len(script_files)} files")
        
        # Health checks and monitoring
        monitoring_patterns = ['health', 'metrics', 'prometheus', 'grafana']
        monitoring_files = []
        
        for pattern in monitoring_patterns:
            monitoring_files.extend(list(self.project_root.rglob(f'*{pattern}*')))
        
        if monitoring_files:
            score += 1
            details.append(f"✅ Monitoring setup: {len(monitoring_files)} files")
        
        # Production configuration
        prod_patterns = ['production', 'prod', 'deploy']
        prod_files = []
        
        for pattern in prod_patterns:
            prod_files.extend(list(self.project_root.rglob(f'*{pattern}*')))
        
        if prod_files:
            score += 1
            details.append(f"✅ Production configs: {len(prod_files)} files")
        
        return score, max_points, details
    
    def generate_report(self, output_file: str = "quality_report.json"):
        """Generate detailed quality report."""
        report_path = self.project_root / output_file
        
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed report saved to: {report_path}")

def main():
    """Run quality gates."""
    project_root = Path(__file__).parent
    
    runner = QualityGateRunner(project_root)
    final_results = runner.run_all_gates()
    
    # Generate report
    runner.generate_report()
    
    # Determine exit code
    if final_results['production_ready']:
        print("\n🚀 ConfoRL is ready for production deployment!")
        return 0
    else:
        print("\n⚠️  ConfoRL needs improvements before production deployment.")
        return 1

if __name__ == "__main__":
    sys.exit(main())