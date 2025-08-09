"""Access control and authentication for ConfoRL.

Comprehensive access control system with role-based permissions,
authentication, and session management for secure RL deployments.
"""

import hashlib
import secrets
import time
import jwt
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import json
import threading

from .audit import get_security_auditor, SecurityEventType, SecurityLevel
from ..utils.logging import get_logger
from ..utils.errors import SecurityError, ValidationError

logger = get_logger(__name__)


class Permission(Enum):
    """System permissions."""
    
    # Model permissions
    MODEL_READ = "model:read"
    MODEL_WRITE = "model:write"
    MODEL_DELETE = "model:delete"
    MODEL_EXECUTE = "model:execute"
    
    # Data permissions
    DATA_READ = "data:read"
    DATA_WRITE = "data:write" 
    DATA_DELETE = "data:delete"
    DATA_EXPORT = "data:export"
    
    # Configuration permissions
    CONFIG_READ = "config:read"
    CONFIG_WRITE = "config:write"
    
    # System permissions
    SYSTEM_ADMIN = "system:admin"
    USER_MANAGEMENT = "user:management"
    AUDIT_ACCESS = "audit:access"
    
    # Research permissions
    RESEARCH_ACCESS = "research:access"
    EXPERIMENT_RUN = "experiment:run"
    BENCHMARK_ACCESS = "benchmark:access"


class Role(Enum):
    """User roles."""
    
    ADMIN = "admin"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    VIEWER = "viewer"
    GUEST = "guest"


# Role-Permission mappings
ROLE_PERMISSIONS = {
    Role.ADMIN: {
        Permission.MODEL_READ, Permission.MODEL_WRITE, Permission.MODEL_DELETE, Permission.MODEL_EXECUTE,
        Permission.DATA_READ, Permission.DATA_WRITE, Permission.DATA_DELETE, Permission.DATA_EXPORT,
        Permission.CONFIG_READ, Permission.CONFIG_WRITE,
        Permission.SYSTEM_ADMIN, Permission.USER_MANAGEMENT, Permission.AUDIT_ACCESS,
        Permission.RESEARCH_ACCESS, Permission.EXPERIMENT_RUN, Permission.BENCHMARK_ACCESS
    },
    Role.RESEARCHER: {
        Permission.MODEL_READ, Permission.MODEL_WRITE, Permission.MODEL_EXECUTE,
        Permission.DATA_READ, Permission.DATA_WRITE,
        Permission.CONFIG_READ,
        Permission.RESEARCH_ACCESS, Permission.EXPERIMENT_RUN, Permission.BENCHMARK_ACCESS
    },
    Role.OPERATOR: {
        Permission.MODEL_READ, Permission.MODEL_EXECUTE,
        Permission.DATA_READ,
        Permission.CONFIG_READ
    },
    Role.VIEWER: {
        Permission.MODEL_READ,
        Permission.DATA_READ,
        Permission.CONFIG_READ
    },
    Role.GUEST: {
        Permission.MODEL_READ
    }
}


@dataclass
class User:
    """User account."""
    
    user_id: str
    username: str
    email: str
    password_hash: str
    roles: Set[Role]
    permissions: Set[Permission]
    created_at: float
    last_login: Optional[float] = None
    failed_login_attempts: int = 0
    locked_until: Optional[float] = None
    is_active: bool = True
    session_timeout: int = 3600  # 1 hour default
    
    def __post_init__(self):
        # Convert string roles/permissions back to enums if needed
        if self.roles and isinstance(next(iter(self.roles)), str):
            self.roles = {Role(r) for r in self.roles}
        if self.permissions and isinstance(next(iter(self.permissions)), str):
            self.permissions = {Permission(p) for p in self.permissions}


@dataclass
class Session:
    """User session."""
    
    session_id: str
    user_id: str
    created_at: float
    last_activity: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True
    
    def is_expired(self, timeout: int) -> bool:
        """Check if session is expired."""
        return time.time() - self.last_activity > timeout


class PasswordManager:
    """Secure password management."""
    
    @staticmethod
    def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, bytes]:
        """Hash password with salt.
        
        Args:
            password: Plain text password
            salt: Salt bytes (generates if None)
            
        Returns:
            Tuple of (hash_hex, salt_bytes)
        """
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 with SHA-256
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt,
            100000  # 100,000 iterations
        )
        
        # Combine salt and hash
        combined = salt + password_hash
        return combined.hex(), salt
    
    @staticmethod
    def verify_password(password: str, hash_hex: str) -> bool:
        """Verify password against hash.
        
        Args:
            password: Plain text password
            hash_hex: Stored password hash
            
        Returns:
            True if password matches
        """
        try:
            combined = bytes.fromhex(hash_hex)
            salt = combined[:32]
            stored_hash = combined[32:]
            
            # Compute hash with stored salt
            password_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            
            # Constant-time comparison
            return secrets.compare_digest(stored_hash, password_hash)
            
        except Exception:
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate secure random password.
        
        Args:
            length: Password length
            
        Returns:
            Random password
        """
        import string
        
        # Use cryptographically secure random
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        
        return password
    
    @staticmethod
    def check_password_strength(password: str) -> Dict[str, Any]:
        """Check password strength.
        
        Args:
            password: Password to check
            
        Returns:
            Strength assessment
        """
        import re
        
        strength_score = 0
        feedback = []
        
        # Length check
        if len(password) >= 12:
            strength_score += 2
        elif len(password) >= 8:
            strength_score += 1
        else:
            feedback.append("Password should be at least 8 characters")
        
        # Character variety checks
        if re.search(r'[a-z]', password):
            strength_score += 1
        else:
            feedback.append("Add lowercase letters")
        
        if re.search(r'[A-Z]', password):
            strength_score += 1
        else:
            feedback.append("Add uppercase letters")
        
        if re.search(r'[0-9]', password):
            strength_score += 1
        else:
            feedback.append("Add numbers")
        
        if re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            strength_score += 1
        else:
            feedback.append("Add special characters")
        
        # Common password check (simplified)
        common_passwords = {'password', '123456', 'qwerty', 'admin', 'letmein'}
        if password.lower() in common_passwords:
            strength_score = 0
            feedback.append("Avoid common passwords")
        
        # Determine strength level
        if strength_score >= 5:
            strength_level = "strong"
        elif strength_score >= 3:
            strength_level = "medium"
        else:
            strength_level = "weak"
        
        return {
            'strength_level': strength_level,
            'score': strength_score,
            'max_score': 6,
            'feedback': feedback
        }


class AccessController:
    """Main access control system."""
    
    def __init__(
        self,
        user_store_file: Optional[str] = None,
        jwt_secret: Optional[str] = None,
        session_timeout: int = 3600
    ):
        """Initialize access controller.
        
        Args:
            user_store_file: File to store user data
            jwt_secret: Secret for JWT tokens
            session_timeout: Default session timeout in seconds
        """
        self.user_store_file = Path(user_store_file) if user_store_file else None
        self.jwt_secret = jwt_secret or secrets.token_urlsafe(32)
        self.session_timeout = session_timeout
        
        # In-memory storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Password manager
        self.password_manager = PasswordManager()
        
        # Security auditor
        self.auditor = get_security_auditor()
        
        # Load existing users
        self._load_users()
        
        # Create default admin if no users exist
        if not self.users:
            self._create_default_admin()
        
        logger.info(f"Access controller initialized ({len(self.users)} users loaded)")
    
    def _load_users(self):
        """Load users from storage."""
        if not self.user_store_file or not self.user_store_file.exists():
            return
        
        try:
            with open(self.user_store_file, 'r') as f:
                users_data = json.load(f)
            
            for user_data in users_data:
                # Convert role and permission strings back to enums
                user_data['roles'] = {Role(r) for r in user_data['roles']}
                user_data['permissions'] = {Permission(p) for p in user_data['permissions']}
                
                user = User(**user_data)
                self.users[user.user_id] = user
            
            logger.info(f"Loaded {len(self.users)} users from {self.user_store_file}")
            
        except Exception as e:
            logger.error(f"Failed to load users: {e}")
    
    def _save_users(self):
        """Save users to storage."""
        if not self.user_store_file:
            return
        
        try:
            # Ensure directory exists
            self.user_store_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert users to JSON-serializable format
            users_data = []
            for user in self.users.values():
                user_dict = asdict(user)
                # Convert enums to strings
                user_dict['roles'] = [r.value for r in user.roles]
                user_dict['permissions'] = [p.value for p in user.permissions]
                users_data.append(user_dict)
            
            with open(self.user_store_file, 'w') as f:
                json.dump(users_data, f, indent=2)
            
            logger.debug(f"Saved {len(self.users)} users to {self.user_store_file}")
            
        except Exception as e:
            logger.error(f"Failed to save users: {e}")
    
    def _create_default_admin(self):
        """Create default admin user."""
        admin_password = self.password_manager.generate_secure_password()
        
        self.create_user(
            username="admin",
            email="admin@conforl.ai",
            password=admin_password,
            roles={Role.ADMIN}
        )
        
        logger.warning(f"Created default admin user with password: {admin_password}")
        logger.warning("Please change the admin password immediately!")
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: Optional[Set[Role]] = None
    ) -> User:
        """Create new user.
        
        Args:
            username: Username
            email: Email address
            password: Plain text password
            roles: User roles
            
        Returns:
            Created user
        """
        with self._lock:
            # Validate inputs
            if not username or not email or not password:
                raise ValidationError("Username, email, and password are required")
            
            # Check for existing user
            for user in self.users.values():
                if user.username == username:
                    raise ValidationError(f"Username '{username}' already exists")
                if user.email == email:
                    raise ValidationError(f"Email '{email}' already exists")
            
            # Check password strength
            strength = self.password_manager.check_password_strength(password)
            if strength['strength_level'] == 'weak':
                raise ValidationError(f"Password too weak: {', '.join(strength['feedback'])}")
            
            # Generate user ID
            user_id = secrets.token_urlsafe(16)
            
            # Hash password
            password_hash, _ = self.password_manager.hash_password(password)
            
            # Default roles and permissions
            roles = roles or {Role.VIEWER}
            permissions = set()
            for role in roles:
                permissions.update(ROLE_PERMISSIONS.get(role, set()))
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                roles=roles,
                permissions=permissions,
                created_at=time.time()
            )
            
            self.users[user_id] = user
            self._save_users()
            
            # Log event
            self.auditor.log_event(
                event_type=SecurityEventType.SERVICE_STARTED,  # User creation
                security_level=SecurityLevel.INFO,
                resource="user_management",
                action="create_user",
                outcome="success",
                details={"username": username, "roles": [r.value for r in roles]}
            )
            
            logger.info(f"Created user: {username} with roles {[r.value for r in roles]}")
            return user
    
    def authenticate(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[str]:
        """Authenticate user and create session.
        
        Args:
            username: Username
            password: Password
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Session token if authentication successful
        """
        with self._lock:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username:
                    user = u
                    break
            
            if not user:
                self.auditor.log_event(
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    security_level=SecurityLevel.WARNING,
                    ip_address=ip_address,
                    resource="authentication",
                    action="login",
                    outcome="user_not_found",
                    details={"username": username}
                )
                return None
            
            # Check if account is locked
            if user.locked_until and time.time() < user.locked_until:
                self.auditor.log_event(
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    security_level=SecurityLevel.WARNING,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    resource="authentication",
                    action="login",
                    outcome="account_locked"
                )
                return None
            
            # Check if account is active
            if not user.is_active:
                self.auditor.log_event(
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    security_level=SecurityLevel.WARNING,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    resource="authentication",
                    action="login",
                    outcome="account_disabled"
                )
                return None
            
            # Verify password
            if not self.password_manager.verify_password(password, user.password_hash):
                # Increment failed attempts
                user.failed_login_attempts += 1
                
                # Lock account after too many failures
                if user.failed_login_attempts >= 5:
                    user.locked_until = time.time() + 1800  # Lock for 30 minutes
                    logger.warning(f"Account locked due to failed attempts: {username}")
                
                self._save_users()
                
                self.auditor.log_event(
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    security_level=SecurityLevel.WARNING,
                    user_id=user.user_id,
                    ip_address=ip_address,
                    resource="authentication",
                    action="login",
                    outcome="wrong_password",
                    details={"failed_attempts": user.failed_login_attempts}
                )
                return None
            
            # Authentication successful - reset failed attempts
            user.failed_login_attempts = 0
            user.last_login = time.time()
            user.locked_until = None
            
            # Create session
            session_id = secrets.token_urlsafe(32)
            session = Session(
                session_id=session_id,
                user_id=user.user_id,
                created_at=time.time(),
                last_activity=time.time(),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            self.sessions[session_id] = session
            self._save_users()
            
            # Log successful login
            self.auditor.log_event(
                event_type=SecurityEventType.LOGIN_SUCCESS,
                security_level=SecurityLevel.INFO,
                user_id=user.user_id,
                session_id=session_id,
                ip_address=ip_address,
                resource="authentication",
                action="login",
                outcome="success"
            )
            
            logger.info(f"User authenticated: {username}")
            return session_id
    
    def logout(self, session_id: str) -> bool:
        """Logout user session.
        
        Args:
            session_id: Session ID
            
        Returns:
            True if logout successful
        """
        with self._lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            user_id = session.user_id
            
            # Deactivate session
            session.is_active = False
            
            # Log logout
            self.auditor.log_event(
                event_type=SecurityEventType.LOGOUT,
                security_level=SecurityLevel.INFO,
                user_id=user_id,
                session_id=session_id,
                resource="authentication",
                action="logout",
                outcome="success"
            )
            
            # Remove session
            del self.sessions[session_id]
            
            logger.info(f"User logged out: {user_id}")
            return True
    
    def validate_session(self, session_id: str) -> Optional[User]:
        """Validate session and return user.
        
        Args:
            session_id: Session ID
            
        Returns:
            User if session is valid
        """
        if not session_id or session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        if not session.is_active:
            return None
        
        # Check if session is expired
        user = self.users.get(session.user_id)
        if not user:
            return None
        
        session_timeout = user.session_timeout
        if session.is_expired(session_timeout):
            # Session expired
            self.auditor.log_event(
                event_type=SecurityEventType.SESSION_EXPIRED,
                security_level=SecurityLevel.INFO,
                user_id=session.user_id,
                session_id=session_id,
                resource="authentication"
            )
            
            del self.sessions[session_id]
            return None
        
        # Update last activity
        session.last_activity = time.time()
        
        return user
    
    def check_permission(
        self,
        user: User,
        permission: Permission,
        resource: Optional[str] = None
    ) -> bool:
        """Check if user has permission.
        
        Args:
            user: User to check
            permission: Required permission
            resource: Resource being accessed (optional)
            
        Returns:
            True if user has permission
        """
        # Check if user is active
        if not user.is_active:
            return False
        
        # Check if user has required permission
        has_permission = permission in user.permissions
        
        # Log access attempt
        self.auditor.log_event(
            event_type=SecurityEventType.ACCESS_GRANTED if has_permission else SecurityEventType.ACCESS_DENIED,
            security_level=SecurityLevel.INFO if has_permission else SecurityLevel.WARNING,
            user_id=user.user_id,
            resource=resource or "unknown",
            action=permission.value,
            outcome="granted" if has_permission else "denied"
        )
        
        return has_permission
    
    def require_permission(self, permission: Permission):
        """Decorator to require permission for function access.
        
        Args:
            permission: Required permission
            
        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                # Extract session from kwargs or context
                session_id = kwargs.get('session_id') or getattr(func, '_current_session', None)
                
                if not session_id:
                    raise SecurityError("Authentication required")
                
                user = self.validate_session(session_id)
                if not user:
                    raise SecurityError("Invalid or expired session")
                
                if not self.check_permission(user, permission, resource=func.__name__):
                    raise SecurityError(f"Permission denied: {permission.value}")
                
                return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def update_user_roles(self, user_id: str, roles: Set[Role]) -> bool:
        """Update user roles.
        
        Args:
            user_id: User ID
            roles: New roles
            
        Returns:
            True if update successful
        """
        with self._lock:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            old_roles = user.roles.copy()
            
            # Update roles and permissions
            user.roles = roles
            user.permissions = set()
            for role in roles:
                user.permissions.update(ROLE_PERMISSIONS.get(role, set()))
            
            self._save_users()
            
            # Log role change
            self.auditor.log_event(
                event_type=SecurityEventType.PRIVILEGE_ESCALATION if len(roles) > len(old_roles) else SecurityEventType.CONFIG_CHANGED,
                security_level=SecurityLevel.WARNING,
                user_id=user_id,
                resource="user_management",
                action="update_roles",
                details={
                    "old_roles": [r.value for r in old_roles],
                    "new_roles": [r.value for r in roles]
                }
            )
            
            logger.info(f"Updated roles for user {user.username}: {[r.value for r in roles]}")
            return True
    
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        with self._lock:
            expired_sessions = []
            
            for session_id, session in self.sessions.items():
                user = self.users.get(session.user_id)
                if not user or session.is_expired(user.session_timeout):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.sessions[session_id]
            
            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
            
            return len(expired_sessions)
    
    def get_user_info(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user information (safe for API).
        
        Args:
            user_id: User ID
            
        Returns:
            User information (no password)
        """
        user = self.users.get(user_id)
        if not user:
            return None
        
        return {
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'roles': [r.value for r in user.roles],
            'permissions': [p.value for p in user.permissions],
            'created_at': user.created_at,
            'last_login': user.last_login,
            'is_active': user.is_active,
            'failed_login_attempts': user.failed_login_attempts,
            'locked_until': user.locked_until
        }
    
    def get_access_control_stats(self) -> Dict[str, Any]:
        """Get access control statistics."""
        with self._lock:
            active_sessions = len([s for s in self.sessions.values() if s.is_active])
            
            role_counts = {}
            for user in self.users.values():
                for role in user.roles:
                    role_counts[role.value] = role_counts.get(role.value, 0) + 1
            
            return {
                'total_users': len(self.users),
                'active_users': len([u for u in self.users.values() if u.is_active]),
                'locked_users': len([u for u in self.users.values() if u.locked_until and time.time() < u.locked_until]),
                'active_sessions': active_sessions,
                'total_sessions': len(self.sessions),
                'role_distribution': role_counts,
                'session_timeout': self.session_timeout
            }


class RoleManager:
    """Manage roles and permissions."""
    
    def __init__(self):
        """Initialize role manager."""
        self.custom_roles: Dict[str, Set[Permission]] = {}
        
        logger.info("Role manager initialized")
    
    def create_custom_role(self, role_name: str, permissions: Set[Permission]) -> None:
        """Create custom role with specific permissions.
        
        Args:
            role_name: Name of custom role
            permissions: Set of permissions
        """
        self.custom_roles[role_name] = permissions
        logger.info(f"Created custom role: {role_name} with {len(permissions)} permissions")
    
    def get_role_permissions(self, role: Union[Role, str]) -> Set[Permission]:
        """Get permissions for a role.
        
        Args:
            role: Role enum or custom role name
            
        Returns:
            Set of permissions
        """
        if isinstance(role, Role):
            return ROLE_PERMISSIONS.get(role, set())
        else:
            return self.custom_roles.get(role, set())
    
    def list_all_permissions(self) -> List[Permission]:
        """List all available permissions."""
        return list(Permission)
    
    def list_all_roles(self) -> Dict[str, List[str]]:
        """List all roles and their permissions."""
        all_roles = {}
        
        # Built-in roles
        for role, permissions in ROLE_PERMISSIONS.items():
            all_roles[role.value] = [p.value for p in permissions]
        
        # Custom roles
        for role_name, permissions in self.custom_roles.items():
            all_roles[role_name] = [p.value for p in permissions]
        
        return all_roles


# Global instances
_global_access_controller = None
_global_role_manager = None


def get_access_controller() -> AccessController:
    """Get global access controller instance."""
    global _global_access_controller
    if _global_access_controller is None:
        _global_access_controller = AccessController()
    return _global_access_controller


def get_role_manager() -> RoleManager:
    """Get global role manager instance."""
    global _global_role_manager
    if _global_role_manager is None:
        _global_role_manager = RoleManager()
    return _global_role_manager


# Convenience decorators
def require_permission(permission: Permission):
    """Decorator to require permission."""
    return get_access_controller().require_permission(permission)


def require_role(role: Role):
    """Decorator to require role."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would need to be integrated with the session context
            # For now, just return the original function
            return func(*args, **kwargs)
        return wrapper
    return decorator