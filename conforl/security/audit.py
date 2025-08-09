"""Security auditing and threat detection for ConfoRL.

Comprehensive security monitoring, audit logging, and threat detection
for production safe RL deployments.
"""

import json
import time
import threading
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import hashlib
import ipaddress
import re

from ..utils.logging import get_logger
from ..utils.errors import SecurityError

logger = get_logger(__name__)


class SecurityEventType(Enum):
    """Types of security events."""
    
    # Authentication events
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    SESSION_EXPIRED = "session_expired"
    
    # Authorization events
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    
    # Data access events
    MODEL_ACCESSED = "model_accessed"
    MODEL_MODIFIED = "model_modified"
    DATA_EXPORTED = "data_exported"
    CONFIG_CHANGED = "config_changed"
    
    # Security violations
    INJECTION_ATTEMPT = "injection_attempt"
    BRUTE_FORCE_ATTACK = "brute_force_attack"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    
    # System events
    SERVICE_STARTED = "service_started"
    SERVICE_STOPPED = "service_stopped"
    ERROR_OCCURRED = "error_occurred"


class SecurityLevel(Enum):
    """Security levels for events."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ALERT = "alert"


@dataclass
class SecurityEvent:
    """Security audit event."""
    
    event_type: SecurityEventType
    security_level: SecurityLevel
    timestamp: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    outcome: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    risk_score: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class SecurityAuditor:
    """Comprehensive security auditing system."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_events: int = 100000,
        retention_days: int = 90
    ):
        """Initialize security auditor.
        
        Args:
            log_file: Path to audit log file
            max_events: Maximum events to keep in memory
            retention_days: Days to retain audit logs
        """
        self.log_file = Path(log_file) if log_file else None
        self.max_events = max_events
        self.retention_days = retention_days
        
        # Event storage
        self.events = deque(maxlen=max_events)
        self.event_counts = defaultdict(int)
        self.user_activity = defaultdict(list)
        self.ip_activity = defaultdict(list)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Event handlers
        self.event_handlers = []
        
        # Initialize logging
        self._initialize_logging()
        
        logger.info(f"Security auditor initialized (log_file={bool(log_file)})")
    
    def _initialize_logging(self):
        """Initialize audit logging."""
        if self.log_file:
            # Ensure log directory exists
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Log auditor initialization
            init_event = SecurityEvent(
                event_type=SecurityEventType.SERVICE_STARTED,
                security_level=SecurityLevel.INFO,
                timestamp=time.time(),
                resource="security_auditor",
                details={"max_events": self.max_events, "retention_days": self.retention_days}
            )
            
            self._write_event_to_file(init_event)
    
    def log_event(
        self,
        event_type: SecurityEventType,
        security_level: SecurityLevel = SecurityLevel.INFO,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None
    ) -> SecurityEvent:
        """Log security event.
        
        Args:
            event_type: Type of security event
            security_level: Security level
            user_id: User identifier
            session_id: Session identifier
            ip_address: Client IP address
            resource: Resource accessed
            action: Action performed
            outcome: Outcome of action
            details: Additional event details
            risk_score: Risk score (0-1, higher = riskier)
            
        Returns:
            Created security event
        """
        event = SecurityEvent(
            event_type=event_type,
            security_level=security_level,
            timestamp=time.time(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            resource=resource,
            action=action,
            outcome=outcome,
            details=details or {},
            risk_score=risk_score
        )
        
        with self._lock:
            # Store event
            self.events.append(event)
            self.event_counts[event_type] += 1
            
            # Track user and IP activity
            if user_id:
                self.user_activity[user_id].append(event.timestamp)
                # Keep only recent activity
                cutoff = time.time() - 86400  # 24 hours
                self.user_activity[user_id] = [
                    ts for ts in self.user_activity[user_id] if ts > cutoff
                ]
            
            if ip_address:
                self.ip_activity[ip_address].append(event.timestamp)
                # Keep only recent activity
                self.ip_activity[ip_address] = [
                    ts for ts in self.ip_activity[ip_address] if ts > cutoff
                ]
            
            # Write to file if configured
            if self.log_file:
                self._write_event_to_file(event)
            
            # Trigger event handlers
            for handler in self.event_handlers:
                try:
                    handler(event)
                except Exception as e:
                    logger.error(f"Event handler failed: {e}")
        
        # Log to standard logger based on security level
        if security_level == SecurityLevel.CRITICAL:
            logger.critical(f"SECURITY: {event_type.value} - {resource} by {user_id}")
        elif security_level == SecurityLevel.ALERT:
            logger.error(f"SECURITY: {event_type.value} - {resource} by {user_id}")
        elif security_level == SecurityLevel.WARNING:
            logger.warning(f"SECURITY: {event_type.value} - {resource} by {user_id}")
        else:
            logger.info(f"SECURITY: {event_type.value} - {resource} by {user_id}")
        
        return event
    
    def _write_event_to_file(self, event: SecurityEvent):
        """Write event to audit log file."""
        try:
            event_json = json.dumps(asdict(event), default=str)
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(event_json + '\n')
                
        except Exception as e:
            logger.error(f"Failed to write audit log: {e}")
    
    def add_event_handler(self, handler: Callable[[SecurityEvent], None]):
        """Add event handler for real-time processing.
        
        Args:
            handler: Function to call for each event
        """
        self.event_handlers.append(handler)
        logger.info("Security event handler added")
    
    def query_events(
        self,
        event_type: Optional[SecurityEventType] = None,
        security_level: Optional[SecurityLevel] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: Optional[int] = None
    ) -> List[SecurityEvent]:
        """Query audit events with filtering.
        
        Args:
            event_type: Filter by event type
            security_level: Filter by security level
            user_id: Filter by user ID
            ip_address: Filter by IP address
            start_time: Filter events after this time
            end_time: Filter events before this time
            limit: Maximum number of events to return
            
        Returns:
            List of matching events
        """
        with self._lock:
            matching_events = []
            
            for event in self.events:
                # Apply filters
                if event_type and event.event_type != event_type:
                    continue
                if security_level and event.security_level != security_level:
                    continue
                if user_id and event.user_id != user_id:
                    continue
                if ip_address and event.ip_address != ip_address:
                    continue
                if start_time and event.timestamp < start_time:
                    continue
                if end_time and event.timestamp > end_time:
                    continue
                
                matching_events.append(event)
                
                # Apply limit
                if limit and len(matching_events) >= limit:
                    break
            
            return matching_events
    
    def get_user_activity_summary(self, user_id: str, hours: int = 24) -> Dict[str, Any]:
        """Get activity summary for a user.
        
        Args:
            user_id: User identifier
            hours: Hours of history to analyze
            
        Returns:
            User activity summary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        user_events = self.query_events(
            user_id=user_id,
            start_time=cutoff_time
        )
        
        # Analyze activity
        event_counts = defaultdict(int)
        security_levels = defaultdict(int)
        resources_accessed = set()
        
        for event in user_events:
            event_counts[event.event_type] += 1
            security_levels[event.security_level] += 1
            if event.resource:
                resources_accessed.add(event.resource)
        
        return {
            'user_id': user_id,
            'total_events': len(user_events),
            'event_counts': dict(event_counts),
            'security_levels': dict(security_levels),
            'resources_accessed': list(resources_accessed),
            'analysis_period_hours': hours,
            'high_risk_events': len([e for e in user_events if e.risk_score and e.risk_score > 0.7])
        }
    
    def get_ip_activity_summary(self, ip_address: str, hours: int = 24) -> Dict[str, Any]:
        """Get activity summary for an IP address.
        
        Args:
            ip_address: IP address
            hours: Hours of history to analyze
            
        Returns:
            IP activity summary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        ip_events = self.query_events(
            ip_address=ip_address,
            start_time=cutoff_time
        )
        
        # Analyze activity
        users = set()
        event_counts = defaultdict(int)
        failed_logins = 0
        
        for event in ip_events:
            if event.user_id:
                users.add(event.user_id)
            event_counts[event.event_type] += 1
            if event.event_type == SecurityEventType.LOGIN_FAILURE:
                failed_logins += 1
        
        return {
            'ip_address': ip_address,
            'total_events': len(ip_events),
            'unique_users': len(users),
            'event_counts': dict(event_counts),
            'failed_logins': failed_logins,
            'analysis_period_hours': hours,
            'suspicious_activity': failed_logins > 5 or len(users) > 10
        }
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive security report.
        
        Args:
            hours: Hours of history to analyze
            
        Returns:
            Security report
        """
        cutoff_time = time.time() - (hours * 3600)
        
        recent_events = self.query_events(start_time=cutoff_time)
        
        # Overall statistics
        total_events = len(recent_events)
        event_counts = defaultdict(int)
        security_levels = defaultdict(int)
        unique_users = set()
        unique_ips = set()
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            security_levels[event.security_level] += 1
            if event.user_id:
                unique_users.add(event.user_id)
            if event.ip_address:
                unique_ips.add(event.ip_address)
        
        # High-risk events
        critical_events = [e for e in recent_events if e.security_level == SecurityLevel.CRITICAL]
        alert_events = [e for e in recent_events if e.security_level == SecurityLevel.ALERT]
        
        # Top active users and IPs
        user_activity_counts = defaultdict(int)
        ip_activity_counts = defaultdict(int)
        
        for event in recent_events:
            if event.user_id:
                user_activity_counts[event.user_id] += 1
            if event.ip_address:
                ip_activity_counts[event.ip_address] += 1
        
        top_users = sorted(user_activity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_ips = sorted(ip_activity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'report_period_hours': hours,
            'report_timestamp': time.time(),
            'overview': {
                'total_events': total_events,
                'unique_users': len(unique_users),
                'unique_ips': len(unique_ips),
                'critical_events': len(critical_events),
                'alert_events': len(alert_events)
            },
            'event_breakdown': dict(event_counts),
            'security_level_breakdown': dict(security_levels),
            'top_users': top_users,
            'top_ips': top_ips,
            'critical_events_sample': [asdict(e) for e in critical_events[:5]],
            'recommendations': self._generate_security_recommendations(recent_events)
        }
    
    def _generate_security_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events."""
        recommendations = []
        
        # Count different types of concerning events
        failed_logins = len([e for e in events if e.event_type == SecurityEventType.LOGIN_FAILURE])
        injection_attempts = len([e for e in events if e.event_type == SecurityEventType.INJECTION_ATTEMPT])
        brute_force_attacks = len([e for e in events if e.event_type == SecurityEventType.BRUTE_FORCE_ATTACK])
        critical_events = len([e for e in events if e.security_level == SecurityLevel.CRITICAL])
        
        if failed_logins > 10:
            recommendations.append("High number of login failures detected - consider implementing account lockout")
        
        if injection_attempts > 0:
            recommendations.append("SQL/code injection attempts detected - review input validation")
        
        if brute_force_attacks > 0:
            recommendations.append("Brute force attacks detected - implement rate limiting")
        
        if critical_events > 5:
            recommendations.append("Multiple critical security events - immediate investigation recommended")
        
        # IP-based recommendations
        ip_counts = defaultdict(int)
        for event in events:
            if event.ip_address:
                ip_counts[event.ip_address] += 1
        
        high_activity_ips = [ip for ip, count in ip_counts.items() if count > 50]
        if high_activity_ips:
            recommendations.append(f"High activity from {len(high_activity_ips)} IP addresses - review for automation/bots")
        
        if not recommendations:
            recommendations.append("No immediate security concerns identified")
        
        return recommendations
    
    def cleanup_old_events(self, days: Optional[int] = None) -> int:
        """Clean up old events beyond retention period.
        
        Args:
            days: Days to retain (uses configured retention if None)
            
        Returns:
            Number of events cleaned up
        """
        retention_days = days or self.retention_days
        cutoff_time = time.time() - (retention_days * 86400)
        
        with self._lock:
            original_count = len(self.events)
            
            # Filter out old events
            self.events = deque(
                (event for event in self.events if event.timestamp > cutoff_time),
                maxlen=self.max_events
            )
            
            cleaned_count = original_count - len(self.events)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old security events")
        
        return cleaned_count
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """Get auditing system statistics."""
        with self._lock:
            return {
                'total_events': len(self.events),
                'max_events': self.max_events,
                'event_type_counts': dict(self.event_counts),
                'active_users': len(self.user_activity),
                'active_ips': len(self.ip_activity),
                'event_handlers': len(self.event_handlers),
                'log_file': str(self.log_file) if self.log_file else None,
                'retention_days': self.retention_days
            }


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self, auditor: SecurityAuditor):
        """Initialize threat detector.
        
        Args:
            auditor: Security auditor to monitor
        """
        self.auditor = auditor
        self.threat_patterns = {}
        self.detection_rules = []
        self.alert_thresholds = {}
        self.detected_threats = deque(maxlen=10000)
        
        # Setup default detection rules
        self._setup_default_rules()
        
        # Register with auditor
        self.auditor.add_event_handler(self._analyze_event)
        
        logger.info("Threat detector initialized")
    
    def _setup_default_rules(self):
        """Setup default threat detection rules."""
        # Brute force detection
        self.alert_thresholds['failed_logins_per_ip'] = 5
        self.alert_thresholds['failed_logins_per_user'] = 3
        
        # Anomaly detection
        self.alert_thresholds['events_per_minute'] = 100
        self.alert_thresholds['unique_users_per_ip'] = 10
        
        # Injection detection patterns
        self.threat_patterns['sql_injection'] = [
            r"(\bunion\s+select)",
            r"(\bdrop\s+table)",
            r"(\binsert\s+into)",
            r"(';.*--)",
        ]
        
        self.threat_patterns['command_injection'] = [
            r"(;\s*rm\s+-rf)",
            r"(\|\s*nc\s+)",
            r"(\$\(.*\))",
            r"(`.*`)",
        ]
    
    def _analyze_event(self, event: SecurityEvent):
        """Analyze security event for threats."""
        try:
            # Check for brute force attacks
            if event.event_type == SecurityEventType.LOGIN_FAILURE:
                self._check_brute_force(event)
            
            # Check for injection attempts
            if event.details:
                self._check_injection_patterns(event)
            
            # Check for anomalous behavior
            self._check_anomalous_behavior(event)
            
        except Exception as e:
            logger.error(f"Threat analysis failed: {e}")
    
    def _check_brute_force(self, event: SecurityEvent):
        """Check for brute force attacks."""
        current_time = time.time()
        window = 300  # 5 minutes
        
        # Check failures by IP
        if event.ip_address:
            recent_failures = self.auditor.query_events(
                event_type=SecurityEventType.LOGIN_FAILURE,
                ip_address=event.ip_address,
                start_time=current_time - window
            )
            
            if len(recent_failures) >= self.alert_thresholds['failed_logins_per_ip']:
                self._create_threat_alert(
                    threat_type="brute_force_by_ip",
                    description=f"Brute force attack from IP {event.ip_address}",
                    related_events=recent_failures[-5:],
                    risk_score=0.8
                )
        
        # Check failures by user
        if event.user_id:
            recent_failures = self.auditor.query_events(
                event_type=SecurityEventType.LOGIN_FAILURE,
                user_id=event.user_id,
                start_time=current_time - window
            )
            
            if len(recent_failures) >= self.alert_thresholds['failed_logins_per_user']:
                self._create_threat_alert(
                    threat_type="brute_force_by_user",
                    description=f"Brute force attack on user {event.user_id}",
                    related_events=recent_failures[-3:],
                    risk_score=0.7
                )
    
    def _check_injection_patterns(self, event: SecurityEvent):
        """Check event details for injection patterns."""
        details_str = json.dumps(event.details).lower()
        
        # Check SQL injection patterns
        for pattern in self.threat_patterns.get('sql_injection', []):
            if re.search(pattern, details_str, re.IGNORECASE):
                self._create_threat_alert(
                    threat_type="sql_injection_attempt",
                    description="SQL injection pattern detected",
                    related_events=[event],
                    risk_score=0.9
                )
                break
        
        # Check command injection patterns
        for pattern in self.threat_patterns.get('command_injection', []):
            if re.search(pattern, details_str):
                self._create_threat_alert(
                    threat_type="command_injection_attempt",
                    description="Command injection pattern detected",
                    related_events=[event],
                    risk_score=0.9
                )
                break
    
    def _check_anomalous_behavior(self, event: SecurityEvent):
        """Check for anomalous behavior patterns."""
        current_time = time.time()
        window = 60  # 1 minute
        
        # Check event rate
        recent_events = self.auditor.query_events(start_time=current_time - window)
        
        if len(recent_events) > self.alert_thresholds['events_per_minute']:
            self._create_threat_alert(
                threat_type="high_event_rate",
                description=f"Unusually high event rate: {len(recent_events)} events/minute",
                related_events=[],
                risk_score=0.6
            )
        
        # Check multiple users from same IP
        if event.ip_address:
            ip_events = self.auditor.query_events(
                ip_address=event.ip_address,
                start_time=current_time - 3600  # 1 hour
            )
            
            unique_users = set(e.user_id for e in ip_events if e.user_id)
            
            if len(unique_users) > self.alert_thresholds['unique_users_per_ip']:
                self._create_threat_alert(
                    threat_type="multiple_users_per_ip",
                    description=f"Multiple users ({len(unique_users)}) from IP {event.ip_address}",
                    related_events=[],
                    risk_score=0.5
                )
    
    def _create_threat_alert(
        self,
        threat_type: str,
        description: str,
        related_events: List[SecurityEvent],
        risk_score: float
    ):
        """Create and log threat alert."""
        threat_alert = {
            'threat_type': threat_type,
            'description': description,
            'risk_score': risk_score,
            'timestamp': time.time(),
            'related_events_count': len(related_events),
            'related_event_ids': [id(e) for e in related_events]
        }
        
        self.detected_threats.append(threat_alert)
        
        # Log threat event
        self.auditor.log_event(
            event_type=SecurityEventType.ANOMALOUS_BEHAVIOR,
            security_level=SecurityLevel.ALERT if risk_score > 0.7 else SecurityLevel.WARNING,
            resource=threat_type,
            details=threat_alert,
            risk_score=risk_score
        )
        
        logger.warning(f"THREAT DETECTED: {threat_type} - {description} (risk: {risk_score})")
    
    def get_threat_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat detection summary.
        
        Args:
            hours: Hours of history to analyze
            
        Returns:
            Threat summary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        recent_threats = [
            t for t in self.detected_threats 
            if t['timestamp'] > cutoff_time
        ]
        
        # Categorize by threat type
        threat_counts = defaultdict(int)
        risk_levels = defaultdict(int)
        
        for threat in recent_threats:
            threat_counts[threat['threat_type']] += 1
            
            # Categorize by risk
            risk_score = threat['risk_score']
            if risk_score >= 0.8:
                risk_levels['high'] += 1
            elif risk_score >= 0.6:
                risk_levels['medium'] += 1
            else:
                risk_levels['low'] += 1
        
        return {
            'analysis_period_hours': hours,
            'total_threats': len(recent_threats),
            'threat_type_counts': dict(threat_counts),
            'risk_level_counts': dict(risk_levels),
            'recent_high_risk_threats': [
                t for t in recent_threats[-10:] 
                if t['risk_score'] >= 0.8
            ],
            'detection_rules_active': len(self.detection_rules),
            'alert_thresholds': self.alert_thresholds
        }
    
    def add_custom_pattern(self, pattern_name: str, regex_patterns: List[str]):
        """Add custom threat detection pattern.
        
        Args:
            pattern_name: Name of the pattern
            regex_patterns: List of regex patterns to match
        """
        self.threat_patterns[pattern_name] = regex_patterns
        logger.info(f"Added custom threat pattern: {pattern_name}")
    
    def update_threshold(self, threshold_name: str, value: Union[int, float]):
        """Update alert threshold.
        
        Args:
            threshold_name: Name of threshold
            value: New threshold value
        """
        old_value = self.alert_thresholds.get(threshold_name, 'undefined')
        self.alert_thresholds[threshold_name] = value
        
        logger.info(f"Updated threshold {threshold_name}: {old_value} -> {value}")


# Global instances
_global_auditor = None
_global_threat_detector = None


def get_security_auditor() -> SecurityAuditor:
    """Get global security auditor instance."""
    global _global_auditor
    if _global_auditor is None:
        _global_auditor = SecurityAuditor()
    return _global_auditor


def get_threat_detector() -> ThreatDetector:
    """Get global threat detector instance.""" 
    global _global_threat_detector
    if _global_threat_detector is None:
        _global_threat_detector = ThreatDetector(get_security_auditor())
    return _global_threat_detector