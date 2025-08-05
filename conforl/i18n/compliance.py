"""Compliance checking and privacy protection utilities."""

import re
import hashlib
from typing import Dict, List, Any, Optional, Set, Callable
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import json

from ..utils.logging import get_logger
from ..utils.security import hash_sensitive_data

logger = get_logger(__name__)


class ComplianceChecker(ABC):
    """Base class for compliance checking."""
    
    def __init__(self, regulation_name: str):
        """Initialize compliance checker.
        
        Args:
            regulation_name: Name of the regulation (e.g., 'GDPR', 'CCPA')
        """
        self.regulation_name = regulation_name
        self.violations = []
        self.checks_performed = 0
    
    @abstractmethod
    def check_data_processing(self, data: Dict[str, Any]) -> bool:
        """Check if data processing complies with regulation.
        
        Args:
            data: Data being processed
            
        Returns:
            True if compliant
        """
        pass
    
    @abstractmethod
    def check_data_storage(self, storage_config: Dict[str, Any]) -> bool:
        """Check if data storage complies with regulation.
        
        Args:
            storage_config: Storage configuration
            
        Returns:
            True if compliant
        """
        pass
    
    @abstractmethod
    def check_user_rights(self, user_rights_config: Dict[str, Any]) -> bool:
        """Check if user rights implementation complies with regulation.
        
        Args:
            user_rights_config: User rights configuration
            
        Returns:
            True if compliant
        """
        pass
    
    def add_violation(self, violation_type: str, description: str, severity: str = "medium"):
        """Add a compliance violation.
        
        Args:
            violation_type: Type of violation
            description: Description of the violation
            severity: Severity level (low, medium, high, critical)
        """
        violation = {
            'type': violation_type,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'regulation': self.regulation_name
        }
        
        self.violations.append(violation)
        logger.warning(f"{self.regulation_name} violation [{severity}]: {description}")
    
    def get_violations(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get compliance violations.
        
        Args:
            severity_filter: Filter by severity level
            
        Returns:
            List of violations
        """
        if severity_filter:
            return [v for v in self.violations if v['severity'] == severity_filter]
        return self.violations.copy()
    
    def is_compliant(self) -> bool:
        """Check if system is compliant (no high or critical violations).
        
        Returns:
            True if compliant
        """
        critical_violations = [
            v for v in self.violations 
            if v['severity'] in ['high', 'critical']
        ]
        return len(critical_violations) == 0
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate compliance report.
        
        Returns:
            Compliance report
        """
        severity_counts = {}
        for violation in self.violations:
            severity = violation['severity']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'regulation': self.regulation_name,
            'compliant': self.is_compliant(),
            'checks_performed': self.checks_performed,
            'total_violations': len(self.violations),
            'severity_breakdown': severity_counts,
            'violations': self.violations,
            'generated_at': datetime.now().isoformat()
        }


class GDPRCompliance(ComplianceChecker):
    """GDPR (General Data Protection Regulation) compliance checker."""
    
    def __init__(self):
        """Initialize GDPR compliance checker."""
        super().__init__("GDPR")
        
        # GDPR-specific configuration
        self.sensitive_data_patterns = [
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',  # IP address
        ]
        
        self.required_user_rights = {
            'access', 'rectification', 'erasure', 'restrict_processing',
            'data_portability', 'object', 'withdraw_consent'
        }
        
        self.data_retention_limits = {
            'personal_data': timedelta(days=365 * 6),  # 6 years default
            'sensitive_data': timedelta(days=365 * 3),  # 3 years for sensitive
            'logs': timedelta(days=90),  # 90 days for logs
        }
    
    def check_data_processing(self, data: Dict[str, Any]) -> bool:
        """Check GDPR compliance for data processing.
        
        Args:
            data: Data being processed
            
        Returns:
            True if compliant
        """
        self.checks_performed += 1
        compliant = True
        
        # Check for explicit consent
        if not data.get('consent_given', False):
            self.add_violation(
                'missing_consent',
                'Processing personal data without explicit consent',
                'critical'
            )
            compliant = False
        
        # Check for lawful basis
        lawful_bases = [
            'consent', 'contract', 'legal_obligation',
            'vital_interests', 'public_task', 'legitimate_interests'
        ]
        
        if not data.get('lawful_basis') in lawful_bases:
            self.add_violation(
                'missing_lawful_basis',
                'No valid lawful basis for processing personal data',
                'critical'
            )
            compliant = False
        
        # Check for data minimization
        if self._detect_excessive_data_collection(data):
            self.add_violation(
                'data_minimization',
                'Collecting more personal data than necessary',
                'high'
            )
            compliant = False
        
        # Check for sensitive data patterns
        if self._detect_sensitive_data(data):
            if not data.get('special_category_consent', False):
                self.add_violation(
                    'sensitive_data_no_consent',
                    'Processing sensitive personal data without explicit consent',
                    'critical'
                )
                compliant = False
        
        return compliant
    
    def check_data_storage(self, storage_config: Dict[str, Any]) -> bool:
        """Check GDPR compliance for data storage.
        
        Args:
            storage_config: Storage configuration
            
        Returns:
            True if compliant
        """
        self.checks_performed += 1
        compliant = True
        
        # Check encryption
        if not storage_config.get('encryption_enabled', False):
            self.add_violation(
                'no_encryption',
                'Personal data stored without encryption',
                'high'
            )
            compliant = False
        
        # Check data retention policies
        retention_policy = storage_config.get('retention_policy', {})
        if not retention_policy:
            self.add_violation(
                'no_retention_policy',
                'No data retention policy defined',
                'medium'
            )
            compliant = False
        
        # Check geographic restrictions
        storage_location = storage_config.get('location', 'unknown')
        eu_countries = {
            'AT', 'BE', 'BG', 'HR', 'CY', 'CZ', 'DK', 'EE', 'FI', 'FR',
            'DE', 'GR', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL',
            'PL', 'PT', 'RO', 'SK', 'SI', 'ES', 'SE'
        }
        
        if storage_location not in eu_countries and not storage_config.get('adequacy_decision', False):
            self.add_violation(
                'inadequate_country',
                f'Data stored in {storage_location} without adequacy decision',
                'high'
            )
            compliant = False
        
        # Check access controls
        if not storage_config.get('access_controls', {}).get('enabled', False):
            self.add_violation(
                'weak_access_controls',
                'Insufficient access controls for personal data storage',
                'medium'
            )
            compliant = False
        
        return compliant
    
    def check_user_rights(self, user_rights_config: Dict[str, Any]) -> bool:
        """Check GDPR compliance for user rights implementation.
        
        Args:
            user_rights_config: User rights configuration
            
        Returns:
            True if compliant
        """
        self.checks_performed += 1
        compliant = True
        
        implemented_rights = set(user_rights_config.get('implemented_rights', []))
        missing_rights = self.required_user_rights - implemented_rights
        
        if missing_rights:
            self.add_violation(
                'missing_user_rights',
                f'Missing implementation of user rights: {", ".join(missing_rights)}',
                'critical'
            )
            compliant = False
        
        # Check response time for user requests
        response_time_days = user_rights_config.get('response_time_days', 0)
        if response_time_days > 30:
            self.add_violation(
                'slow_response_time',
                f'User request response time ({response_time_days} days) exceeds 30-day limit',
                'medium'
            )
            compliant = False
        
        # Check data portability format
        if 'data_portability' in implemented_rights:
            portable_formats = user_rights_config.get('portable_formats', [])
            if 'json' not in portable_formats and 'xml' not in portable_formats:
                self.add_violation(
                    'no_machine_readable_format',
                    'Data portability does not provide machine-readable format',
                    'medium'
                )
                compliant = False
        
        return compliant
    
    def _detect_sensitive_data(self, data: Dict[str, Any]) -> bool:
        """Detect sensitive personal data patterns.
        
        Args:
            data: Data to check
            
        Returns:
            True if sensitive data detected
        """
        data_str = json.dumps(data, default=str)
        
        for pattern in self.sensitive_data_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                return True
        
        # Check for explicit sensitive data fields
        sensitive_fields = {
            'race', 'ethnicity', 'religion', 'political_opinion',
            'health_data', 'biometric_data', 'genetic_data', 'sexual_orientation'
        }
        
        return any(field in data for field in sensitive_fields)
    
    def _detect_excessive_data_collection(self, data: Dict[str, Any]) -> bool:
        """Detect potentially excessive data collection.
        
        Args:
            data: Data being collected
            
        Returns:
            True if excessive collection detected
        """
        # Check for large number of fields
        if len(data) > 50:
            return True
        
        # Check for unnecessary personal identifiers
        unnecessary_fields = {
            'mother_maiden_name', 'fathers_name', 'childhood_pet',
            'favorite_color', 'first_school'
        }
        
        return any(field in data for field in unnecessary_fields)


class CCPACompliance(ComplianceChecker):
    """CCPA (California Consumer Privacy Act) compliance checker."""
    
    def __init__(self):
        """Initialize CCPA compliance checker."""
        super().__init__("CCPA")
        
        self.required_disclosures = {
            'categories_collected', 'sources', 'business_purpose',
            'categories_shared', 'categories_sold'
        }
        
        self.consumer_rights = {
            'know', 'delete', 'opt_out', 'non_discrimination'
        }
    
    def check_data_processing(self, data: Dict[str, Any]) -> bool:
        """Check CCPA compliance for data processing.
        
        Args:
            data: Data being processed
            
        Returns:
            True if compliant
        """
        self.checks_performed += 1
        compliant = True
        
        # Check for proper notice at collection
        if not data.get('notice_at_collection', False):
            self.add_violation(
                'missing_notice',
                'No notice provided at time of collection',
                'high'
            )
            compliant = False
        
        # Check if data is being sold without opt-out
        if data.get('data_sold', False) and not data.get('opt_out_provided', False):
            self.add_violation(
                'sale_without_opt_out',
                'Personal information sold without providing opt-out mechanism',
                'critical'
            )
            compliant = False
        
        # Check business purpose limitation
        if not data.get('business_purpose'):
            self.add_violation(
                'no_business_purpose',
                'No business purpose specified for data collection',
                'medium'
            )
            compliant = False
        
        return compliant
    
    def check_data_storage(self, storage_config: Dict[str, Any]) -> bool:
        """Check CCPA compliance for data storage.
        
        Args:
            storage_config: Storage configuration
            
        Returns:
            True if compliant
        """
        self.checks_performed += 1
        compliant = True
        
        # Check reasonable security measures
        security_measures = storage_config.get('security_measures', [])
        required_measures = {'encryption', 'access_controls', 'audit_logging'}
        
        if not required_measures.issubset(set(security_measures)):
            missing = required_measures - set(security_measures)
            self.add_violation(
                'insufficient_security',
                f'Missing security measures: {", ".join(missing)}',
                'high'
            )
            compliant = False
        
        # Check data retention policy
        if not storage_config.get('retention_policy'):
            self.add_violation(
                'no_retention_policy',
                'No data retention policy specified',
                'medium'
            )
            compliant = False
        
        return compliant
    
    def check_user_rights(self, user_rights_config: Dict[str, Any]) -> bool:
        """Check CCPA compliance for consumer rights implementation.
        
        Args:
            user_rights_config: Consumer rights configuration
            
        Returns:
            True if compliant
        """
        self.checks_performed += 1
        compliant = True
        
        implemented_rights = set(user_rights_config.get('implemented_rights', []))
        missing_rights = self.consumer_rights - implemented_rights
        
        if missing_rights:
            self.add_violation(
                'missing_consumer_rights',
                f'Missing consumer rights: {", ".join(missing_rights)}',
                'critical'
            )
            compliant = False
        
        # Check response time (45 days for CCPA)
        response_time_days = user_rights_config.get('response_time_days', 0)
        if response_time_days > 45:
            self.add_violation(
                'slow_response_time',
                f'Consumer request response time ({response_time_days} days) exceeds 45-day limit',
                'medium'
            )
            compliant = False
        
        # Check verification methods
        verification_methods = user_rights_config.get('verification_methods', [])
        if not verification_methods:
            self.add_violation(
                'no_verification_method',
                'No verification method specified for consumer requests',
                'high'
            )
            compliant = False
        
        return compliant


class PrivacyByDesign:
    """Privacy by Design implementation helper."""
    
    def __init__(self):
        """Initialize Privacy by Design helper."""
        self.principles = {
            'proactive': 'Proactive not Reactive; Preventative not Remedial',
            'default': 'Privacy as the Default Setting',
            'built_in': 'Full Functionality — Positive-Sum, not Zero-Sum',
            'end_to_end': 'End-to-End Security — Full Lifecycle Protection',
            'visibility': 'Visibility and Transparency — Ensure all Stakeholders',
            'respect': 'Respect for User Privacy — Keep User-Centric',
            'embedded': 'Privacy Embedded into Design'
        }
        
        logger.info("Privacy by Design helper initialized")
    
    def assess_design(self, design_config: Dict[str, Any]) -> Dict[str, Any]:
        """Assess design against Privacy by Design principles.
        
        Args:
            design_config: Design configuration to assess
            
        Returns:
            Assessment results
        """
        assessment = {
            'overall_score': 0,
            'principle_scores': {},
            'recommendations': []
        }
        
        # Assess each principle
        total_score = 0
        
        for principle, description in self.principles.items():
            score = self._assess_principle(principle, design_config)
            assessment['principle_scores'][principle] = {
                'score': score,
                'description': description
            }
            total_score += score
        
        assessment['overall_score'] = total_score / len(self.principles)
        
        # Generate recommendations
        if assessment['overall_score'] < 0.7:
            assessment['recommendations'].append(
                'Overall privacy design needs improvement'
            )
        
        for principle, result in assessment['principle_scores'].items():
            if result['score'] < 0.6:
                assessment['recommendations'].append(
                    f'Improve {principle}: {result["description"]}'
                )
        
        return assessment
    
    def _assess_principle(self, principle: str, config: Dict[str, Any]) -> float:
        """Assess a specific Privacy by Design principle.
        
        Args:
            principle: Principle to assess
            config: Configuration to assess
            
        Returns:
            Score between 0 and 1
        """
        if principle == 'proactive':
            # Check for proactive privacy measures
            measures = config.get('proactive_measures', [])
            return min(1.0, len(measures) / 5)  # Up to 5 measures
        
        elif principle == 'default':
            # Check if privacy is default
            return 1.0 if config.get('privacy_by_default', False) else 0.0
        
        elif principle == 'built_in':
            # Check if privacy is built into architecture
            return 1.0 if config.get('privacy_built_in', False) else 0.0
        
        elif principle == 'end_to_end':
            # Check for end-to-end security
            security_features = config.get('security_features', [])
            required_features = {'encryption', 'access_control', 'audit'}
            coverage = len(set(security_features) & required_features) / len(required_features)
            return coverage
        
        elif principle == 'visibility':
            # Check for transparency measures
            transparency = config.get('transparency_measures', [])
            return min(1.0, len(transparency) / 3)  # Up to 3 measures
        
        elif principle == 'respect':
            # Check user-centric design
            return 1.0 if config.get('user_centric', False) else 0.0
        
        elif principle == 'embedded':
            # Check if privacy is embedded in design process
            return 1.0 if config.get('privacy_embedded', False) else 0.0
        
        return 0.0