#!/usr/bin/env python3
"""Comprehensive security scan for ConfoRL - Production Ready Security Validation."""

import ast
import os
import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Set
import re

class SecurityScanner:
    """Comprehensive security scanner for ConfoRL."""
    
    def __init__(self):
        self.issues = []
        self.warnings = []
        self.info = []
        self.scanned_files = 0
        
    def scan_all(self) -> Dict[str, Any]:
        """Run comprehensive security scan."""
        print("🔒 Starting comprehensive security scan...")
        print("=" * 60)
        
        results = {
            'timestamp': time.time(),
            'scan_duration': 0,
            'files_scanned': 0,
            'critical_issues': [],
            'warnings': [],
            'info': [],
            'categories': {
                'injection': [],
                'hardcoded_secrets': [],
                'insecure_imports': [],
                'file_operations': [],
                'network_security': [],
                'crypto_issues': [],
                'input_validation': [],
                'permissions': []
            },
            'summary': {}
        }
        
        start_time = time.time()
        
        # Scan Python files
        python_files = list(Path('.').rglob('*.py'))
        total_files = len(python_files)
        
        print(f"📁 Scanning {total_files} Python files...")
        
        for i, file_path in enumerate(python_files):
            if i % 10 == 0:
                print(f"Progress: {i}/{total_files} files ({i/total_files*100:.1f}%)")
            
            try:
                self._scan_file(file_path, results)
                self.scanned_files += 1
            except Exception as e:
                self.warnings.append(f"Failed to scan {file_path}: {e}")
        
        # Scan configuration files
        self._scan_config_files(results)
        
        # Scan dependencies
        self._scan_dependencies(results)
        
        # Check file permissions
        self._check_file_permissions(results)
        
        # Generate summary
        results['scan_duration'] = time.time() - start_time
        results['files_scanned'] = self.scanned_files
        results['critical_issues'] = self.issues
        results['warnings'] = self.warnings
        results['info'] = self.info
        
        self._generate_summary(results)
        
        return results
    
    def _scan_file(self, file_path: Path, results: Dict[str, Any]):
        """Scan individual Python file for security issues."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST for deeper analysis
            try:
                tree = ast.parse(content)
                self._analyze_ast(tree, file_path, results)
            except SyntaxError:
                pass  # Skip files with syntax errors
            
            # Pattern-based security checks
            self._check_hardcoded_secrets(content, file_path, results)
            self._check_sql_injection_patterns(content, file_path, results)
            self._check_command_injection(content, file_path, results)
            self._check_insecure_imports(content, file_path, results)
            self._check_file_operations(content, file_path, results)
            self._check_network_security(content, file_path, results)
            self._check_crypto_usage(content, file_path, results)
            
        except Exception as e:
            self.warnings.append(f"Error scanning {file_path}: {e}")
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path, results: Dict[str, Any]):
        """Analyze AST for security patterns."""
        for node in ast.walk(tree):
            # Check for eval/exec usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile']:
                        issue = {
                            'type': 'dangerous_function',
                            'severity': 'critical',
                            'file': str(file_path),
                            'line': getattr(node, 'lineno', 0),
                            'description': f"Dangerous function '{node.func.id}' can execute arbitrary code",
                            'function': node.func.id
                        }
                        self.issues.append(issue)
                        results['categories']['injection'].append(issue)
            
            # Check for subprocess calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'subprocess'):
                        self._check_subprocess_usage(node, file_path, results)
            
            # Check for pickle usage
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if (isinstance(node.func.value, ast.Name) and 
                        node.func.value.id == 'pickle' and
                        node.func.attr in ['load', 'loads']):
                        warning = {
                            'type': 'insecure_deserialization',
                            'severity': 'warning',
                            'file': str(file_path),
                            'line': getattr(node, 'lineno', 0),
                            'description': 'Pickle deserialization can execute arbitrary code'
                        }
                        self.warnings.append(warning)
                        results['categories']['injection'].append(warning)
    
    def _check_subprocess_usage(self, node: ast.Call, file_path: Path, results: Dict[str, Any]):
        """Check subprocess usage for command injection vulnerabilities."""
        if node.func.attr in ['call', 'run', 'Popen', 'check_call', 'check_output']:
            # Check if shell=True is used
            for keyword in node.keywords:
                if keyword.arg == 'shell' and isinstance(keyword.value, ast.Constant):
                    if keyword.value.value is True:
                        issue = {
                            'type': 'command_injection',
                            'severity': 'critical',
                            'file': str(file_path),
                            'line': getattr(node, 'lineno', 0),
                            'description': 'subprocess with shell=True can lead to command injection',
                            'recommendation': 'Use shell=False or validate/sanitize inputs'
                        }
                        self.issues.append(issue)
                        results['categories']['injection'].append(issue)
    
    def _check_hardcoded_secrets(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for hardcoded secrets and sensitive data."""
        secret_patterns = [
            (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{8,}["\']', 'hardcoded_password'),
            (r'(?i)(api[_-]?key|apikey)\s*=\s*["\'][^"\']{16,}["\']', 'api_key'),
            (r'(?i)(secret[_-]?key|secretkey)\s*=\s*["\'][^"\']{16,}["\']', 'secret_key'),
            (r'(?i)(access[_-]?token|accesstoken)\s*=\s*["\'][^"\']{16,}["\']', 'access_token'),
            (r'(?i)(private[_-]?key|privatekey)\s*=\s*["\'][^"\']{32,}["\']', 'private_key'),
            (r'["\'][0-9a-f]{32,}["\']', 'potential_hash'),
            (r'sk_[a-zA-Z0-9]{24,}', 'stripe_secret_key'),
            (r'pk_[a-zA-Z0-9]{24,}', 'stripe_public_key'),
            (r'AKIA[0-9A-Z]{16}', 'aws_access_key'),
            (r'[0-9a-zA-Z/+]{40}', 'aws_secret_key'),
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, secret_type in secret_patterns:
                if re.search(pattern, line):
                    # Skip obvious test/example values
                    if any(skip in line.lower() for skip in ['test', 'example', 'dummy', 'placeholder', 'xxx']):
                        continue
                    
                    issue = {
                        'type': 'hardcoded_secret',
                        'severity': 'critical',
                        'file': str(file_path),
                        'line': line_num,
                        'description': f'Potential hardcoded {secret_type} found',
                        'secret_type': secret_type,
                        'recommendation': 'Use environment variables or secure configuration'
                    }
                    self.issues.append(issue)
                    results['categories']['hardcoded_secrets'].append(issue)
    
    def _check_sql_injection_patterns(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for SQL injection vulnerabilities."""
        sql_patterns = [
            r'cursor\.execute\s*\(\s*[\'\""][^\'\"]*%[sd][^\'\"]*[\'\""][^)]*\)',
            r'cursor\.execute\s*\(\s*[\'\""][^\'\"]*\+[^\'\"]*[\'\""][^)]*\)',
            r'\.execute\s*\(\s*[\'\""][^\'\"]*\.format\([^)]*\)[^\'\"]*[\'\""][^)]*\)',
            r'SELECT\s+.*\s+WHERE\s+.*=\s*[\'\""][^\'\"]*\+',
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issue = {
                        'type': 'sql_injection',
                        'severity': 'critical',
                        'file': str(file_path),
                        'line': line_num,
                        'description': 'Potential SQL injection vulnerability',
                        'recommendation': 'Use parameterized queries or ORM'
                    }
                    self.issues.append(issue)
                    results['categories']['injection'].append(issue)
    
    def _check_command_injection(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for command injection patterns."""
        command_patterns = [
            r'os\.system\s*\([^)]*\+',
            r'os\.popen\s*\([^)]*\+',
            r'subprocess\.[^(]*\([^)]*shell\s*=\s*True[^)]*\+',
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for pattern in command_patterns:
                if re.search(pattern, line):
                    issue = {
                        'type': 'command_injection',
                        'severity': 'critical',
                        'file': str(file_path),
                        'line': line_num,
                        'description': 'Potential command injection vulnerability',
                        'recommendation': 'Validate and sanitize inputs, avoid shell=True'
                    }
                    self.issues.append(issue)
                    results['categories']['injection'].append(issue)
    
    def _check_insecure_imports(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for insecure imports and modules."""
        insecure_imports = [
            ('import pickle', 'Pickle can execute arbitrary code during deserialization'),
            ('import yaml', 'YAML loading can be unsafe, use yaml.safe_load()'),
            ('from yaml import load', 'yaml.load() is unsafe, use safe_load()'),
            ('import marshal', 'Marshal can execute arbitrary code'),
            ('import subprocess', 'Subprocess requires careful input validation'),
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for import_pattern, description in insecure_imports:
                if import_pattern in line:
                    warning = {
                        'type': 'insecure_import',
                        'severity': 'warning',
                        'file': str(file_path),
                        'line': line_num,
                        'description': description,
                        'import': import_pattern
                    }
                    self.warnings.append(warning)
                    results['categories']['insecure_imports'].append(warning)
    
    def _check_file_operations(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for insecure file operations."""
        file_patterns = [
            (r'open\s*\([^)]*\+[^)]*[\'\"]\w+[\'\"]\s*\)', 'File path concatenation'),
            (r'os\.path\.join\s*\([^)]*\+', 'Path traversal vulnerability'),
            (r'with\s+open\s*\([^)]*input\([^)]*\)', 'User input in file operations'),
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in file_patterns:
                if re.search(pattern, line):
                    warning = {
                        'type': 'insecure_file_operation',
                        'severity': 'warning',
                        'file': str(file_path),
                        'line': line_num,
                        'description': description,
                        'recommendation': 'Validate file paths and prevent directory traversal'
                    }
                    self.warnings.append(warning)
                    results['categories']['file_operations'].append(warning)
    
    def _check_network_security(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for network security issues."""
        network_patterns = [
            (r'urllib\.request\.urlopen\s*\([^)]*verify\s*=\s*False', 'SSL verification disabled'),
            (r'requests\.[^(]*\([^)]*verify\s*=\s*False', 'SSL verification disabled'),
            (r'ssl\.create_default_context\s*\([^)]*check_hostname\s*=\s*False', 'Hostname verification disabled'),
            (r'http://', 'Unencrypted HTTP connection'),
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in network_patterns:
                if re.search(pattern, line):
                    severity = 'critical' if 'disabled' in description else 'warning'
                    issue = {
                        'type': 'network_security',
                        'severity': severity,
                        'file': str(file_path),
                        'line': line_num,
                        'description': description,
                        'recommendation': 'Enable SSL verification and use HTTPS'
                    }
                    if severity == 'critical':
                        self.issues.append(issue)
                    else:
                        self.warnings.append(issue)
                    results['categories']['network_security'].append(issue)
    
    def _check_crypto_usage(self, content: str, file_path: Path, results: Dict[str, Any]):
        """Check for cryptographic issues."""
        crypto_patterns = [
            (r'hashlib\.md5\s*\(', 'MD5 is cryptographically broken'),
            (r'hashlib\.sha1\s*\(', 'SHA1 is cryptographically weak'),
            (r'random\.random\s*\(', 'Use secrets module for cryptographic randomness'),
            (r'Crypto\.Random', 'Consider using secrets module instead'),
        ]
        
        lines = content.split('\\n')
        for line_num, line in enumerate(lines, 1):
            for pattern, description in crypto_patterns:
                if re.search(pattern, line):
                    warning = {
                        'type': 'crypto_issue',
                        'severity': 'warning',
                        'file': str(file_path),
                        'line': line_num,
                        'description': description,
                        'recommendation': 'Use stronger cryptographic algorithms'
                    }
                    self.warnings.append(warning)
                    results['categories']['crypto_issues'].append(warning)
    
    def _scan_config_files(self, results: Dict[str, Any]):
        """Scan configuration files for security issues."""
        config_files = [
            'config.json', 'config.yaml', 'config.yml', 
            '.env', '.env.local', '.env.production',
            'settings.py', 'settings.json'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r') as f:
                        content = f.read()
                    
                    # Check for sensitive data in config
                    if re.search(r'(password|secret|key|token).*[:=].*[^\\s]', content, re.IGNORECASE):
                        warning = {
                            'type': 'sensitive_config',
                            'severity': 'warning',
                            'file': config_file,
                            'description': 'Configuration file may contain sensitive data'
                        }
                        self.warnings.append(warning)
                        
                except Exception as e:
                    self.warnings.append(f"Failed to scan config file {config_file}: {e}")
    
    def _scan_dependencies(self, results: Dict[str, Any]):
        """Scan dependencies for known vulnerabilities."""
        try:
            # Check if requirements.txt exists
            if os.path.exists('requirements.txt'):
                with open('requirements.txt', 'r') as f:
                    requirements = f.read()
                
                # Check for known vulnerable packages (simplified)
                vulnerable_patterns = [
                    (r'django==1\\.[0-9]+', 'Django 1.x has known vulnerabilities'),
                    (r'flask==0\\.[0-9]+', 'Flask 0.x has known vulnerabilities'),
                    (r'requests==2\\.[0-9]\\.[0-9]', 'Old requests version may have vulnerabilities'),
                    (r'pyyaml==3\\.[0-9]+', 'PyYAML 3.x has known vulnerabilities'),
                ]
                
                for pattern, description in vulnerable_patterns:
                    if re.search(pattern, requirements):
                        warning = {
                            'type': 'vulnerable_dependency',
                            'severity': 'warning',
                            'file': 'requirements.txt',
                            'description': description,
                            'recommendation': 'Update to latest secure version'
                        }
                        self.warnings.append(warning)
                        
        except Exception as e:
            self.warnings.append(f"Failed to scan dependencies: {e}")
    
    def _check_file_permissions(self, results: Dict[str, Any]):
        """Check file permissions for security issues."""
        try:
            # Check for overly permissive files
            sensitive_files = ['.env', 'config.json', 'private_key.pem', 'id_rsa']
            
            for sensitive_file in sensitive_files:
                if os.path.exists(sensitive_file):
                    stat_info = os.stat(sensitive_file)
                    mode = stat_info.st_mode & 0o777
                    
                    if mode & 0o044:  # World or group readable
                        issue = {
                            'type': 'file_permissions',
                            'severity': 'warning',
                            'file': sensitive_file,
                            'description': f'Sensitive file {sensitive_file} is readable by others',
                            'permissions': oct(mode),
                            'recommendation': 'Restrict file permissions (e.g., chmod 600)'
                        }
                        self.warnings.append(issue)
                        results['categories']['permissions'].append(issue)
                        
        except Exception as e:
            self.warnings.append(f"Failed to check file permissions: {e}")
    
    def _generate_summary(self, results: Dict[str, Any]):
        """Generate security scan summary."""
        total_issues = len(results['critical_issues'])
        total_warnings = len(results['warnings'])
        
        results['summary'] = {
            'scan_status': 'completed',
            'total_files_scanned': results['files_scanned'],
            'scan_duration_seconds': results['scan_duration'],
            'critical_issues': total_issues,
            'warnings': total_warnings,
            'categories_affected': len([cat for cat, issues in results['categories'].items() if issues]),
            'security_score': max(0, 100 - (total_issues * 10) - (total_warnings * 2)),
            'recommendations': [
                'Review and fix all critical security issues immediately',
                'Address security warnings when possible',
                'Implement secure coding practices',
                'Regular security scans in CI/CD pipeline',
                'Keep dependencies updated',
                'Use environment variables for secrets'
            ]
        }
    
    def print_report(self, results: Dict[str, Any]):
        """Print comprehensive security report."""
        print("\\n" + "=" * 60)
        print("🔒 CONFORL SECURITY SCAN REPORT")
        print("=" * 60)
        
        summary = results['summary']
        print(f"📊 SUMMARY:")
        print(f"  Files Scanned: {summary['total_files_scanned']}")
        print(f"  Scan Duration: {summary['scan_duration_seconds']:.2f} seconds")
        print(f"  Critical Issues: {summary['critical_issues']}")
        print(f"  Warnings: {summary['warnings']}")
        print(f"  Security Score: {summary['security_score']}/100")
        
        if results['critical_issues']:
            print(f"\\n🚨 CRITICAL ISSUES ({len(results['critical_issues'])}):")
            for issue in results['critical_issues']:
                print(f"  ❌ {issue['type']} in {issue['file']}:{issue.get('line', '?')}")
                print(f"     {issue['description']}")
                if 'recommendation' in issue:
                    print(f"     💡 {issue['recommendation']}")
        
        if results['warnings']:
            print(f"\\n⚠️  WARNINGS ({len(results['warnings'])}):")
            for warning in results['warnings'][:10]:  # Show first 10
                print(f"  ⚠️  {warning.get('type', 'warning')} in {warning.get('file', 'unknown')}")
                print(f"     {warning.get('description', warning)}")
        
        print(f"\\n✅ RECOMMENDATIONS:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")
        
        # Security status
        if summary['critical_issues'] == 0:
            print(f"\\n🎉 No critical security issues found!")
            if summary['warnings'] == 0:
                print(f"🔒 ConfoRL passes comprehensive security scan!")
            else:
                print(f"⚠️  {summary['warnings']} warnings should be reviewed.")
        else:
            print(f"\\n🚨 {summary['critical_issues']} critical issues must be fixed before production!")

def main():
    """Run comprehensive security scan."""
    print("ConfoRL Comprehensive Security Scanner")
    print("Checking for vulnerabilities, secrets, and security best practices...")
    print()
    
    scanner = SecurityScanner()
    results = scanner.scan_all()
    
    # Print report
    scanner.print_report(results)
    
    # Save detailed results
    with open('security_scan_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n📄 Detailed results saved to security_scan_results.json")
    
    # Return appropriate exit code
    if results['summary']['critical_issues'] > 0:
        return 1
    elif results['summary']['warnings'] > 10:
        return 1
    else:
        return 0

if __name__ == "__main__":
    sys.exit(main())