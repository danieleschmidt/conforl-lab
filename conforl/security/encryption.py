"""Secure encryption and model serialization for ConfoRL.

Provides secure model serialization, encryption, and key management
for production deployment of safe RL systems.
"""

import os
import json
import pickle
import hashlib
import hmac
import secrets
from typing import Any, Dict, Optional, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, asdict
import base64
import time

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

from ..utils.logging import get_logger
from ..utils.errors import ConfoRLError, SecurityError

logger = get_logger(__name__)


@dataclass
class EncryptionMetadata:
    """Metadata for encrypted data."""
    
    algorithm: str
    key_hash: str  # Hash of key used (for verification)
    salt: str
    iv: Optional[str] = None
    timestamp: float = None
    version: str = "1.0"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class EncryptionManager:
    """Manages encryption keys and operations."""
    
    def __init__(self, key_file: Optional[str] = None):
        """Initialize encryption manager.
        
        Args:
            key_file: Path to key file (if None, uses in-memory key)
        """
        if not CRYPTO_AVAILABLE:
            logger.warning("Cryptography library not available - encryption disabled")
            self.encryption_enabled = False
            return
        
        self.encryption_enabled = True
        self.key_file = Path(key_file) if key_file else None
        self.master_key = None
        self.fernet = None
        
        # Initialize encryption
        self._initialize_encryption()
        
        logger.info(f"Encryption manager initialized (key_file={bool(key_file)})")
    
    def _initialize_encryption(self):
        """Initialize encryption system."""
        if self.key_file and self.key_file.exists():
            # Load existing key
            self.master_key = self._load_key()
        else:
            # Generate new key
            self.master_key = Fernet.generate_key()
            if self.key_file:
                self._save_key(self.master_key)
        
        self.fernet = Fernet(self.master_key)
        logger.debug("Encryption system initialized")
    
    def _load_key(self) -> bytes:
        """Load encryption key from file."""
        try:
            with open(self.key_file, 'rb') as f:
                key = f.read()
            
            # Verify key format
            try:
                Fernet(key)  # Test if key is valid
                return key
            except Exception:
                raise SecurityError("Invalid encryption key format")
                
        except Exception as e:
            raise SecurityError(f"Failed to load encryption key: {e}")
    
    def _save_key(self, key: bytes):
        """Save encryption key to file."""
        try:
            # Ensure directory exists
            self.key_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Write key with secure permissions
            with open(self.key_file, 'wb') as f:
                f.write(key)
            
            # Set secure file permissions (Unix only)
            try:
                os.chmod(self.key_file, 0o600)  # Read/write for owner only
            except (OSError, AttributeError):
                logger.warning("Could not set secure file permissions")
            
            logger.info(f"Encryption key saved to {self.key_file}")
            
        except Exception as e:
            raise SecurityError(f"Failed to save encryption key: {e}")
    
    def encrypt_data(self, data: bytes) -> Tuple[bytes, EncryptionMetadata]:
        """Encrypt data with metadata.
        
        Args:
            data: Raw data to encrypt
            
        Returns:
            Tuple of (encrypted_data, metadata)
        """
        if not self.encryption_enabled:
            raise SecurityError("Encryption not available")
        
        try:
            # Generate salt for this encryption
            salt = os.urandom(16)
            
            # Encrypt data
            encrypted_data = self.fernet.encrypt(data)
            
            # Create metadata
            key_hash = hashlib.sha256(self.master_key).hexdigest()[:16]
            metadata = EncryptionMetadata(
                algorithm="fernet",
                key_hash=key_hash,
                salt=base64.b64encode(salt).decode(),
                timestamp=time.time()
            )
            
            logger.debug(f"Data encrypted ({len(data)} -> {len(encrypted_data)} bytes)")
            return encrypted_data, metadata
            
        except Exception as e:
            raise SecurityError(f"Encryption failed: {e}")
    
    def decrypt_data(self, encrypted_data: bytes, metadata: EncryptionMetadata) -> bytes:
        """Decrypt data using metadata.
        
        Args:
            encrypted_data: Encrypted data
            metadata: Encryption metadata
            
        Returns:
            Decrypted data
        """
        if not self.encryption_enabled:
            raise SecurityError("Encryption not available")
        
        try:
            # Verify key matches
            current_key_hash = hashlib.sha256(self.master_key).hexdigest()[:16]
            if current_key_hash != metadata.key_hash:
                raise SecurityError("Encryption key mismatch")
            
            # Decrypt data
            decrypted_data = self.fernet.decrypt(encrypted_data)
            
            logger.debug(f"Data decrypted ({len(encrypted_data)} -> {len(decrypted_data)} bytes)")
            return decrypted_data
            
        except Exception as e:
            raise SecurityError(f"Decryption failed: {e}")
    
    def generate_hmac(self, data: bytes, key: Optional[bytes] = None) -> str:
        """Generate HMAC for data integrity verification.
        
        Args:
            data: Data to sign
            key: Key for HMAC (uses master key if None)
            
        Returns:
            HMAC hex digest
        """
        signing_key = key or self.master_key
        if not signing_key:
            raise SecurityError("No signing key available")
        
        mac = hmac.new(signing_key, data, hashlib.sha256)
        return mac.hexdigest()
    
    def verify_hmac(self, data: bytes, mac_digest: str, key: Optional[bytes] = None) -> bool:
        """Verify HMAC for data integrity.
        
        Args:
            data: Original data
            mac_digest: HMAC digest to verify
            key: Key for HMAC verification
            
        Returns:
            True if HMAC is valid
        """
        try:
            expected_mac = self.generate_hmac(data, key)
            return hmac.compare_digest(mac_digest, expected_mac)
        except Exception:
            return False
    
    def rotate_key(self) -> None:
        """Rotate encryption key (generate new key)."""
        if not self.encryption_enabled:
            raise SecurityError("Encryption not available")
        
        old_key = self.master_key
        new_key = Fernet.generate_key()
        
        self.master_key = new_key
        self.fernet = Fernet(new_key)
        
        if self.key_file:
            self._save_key(new_key)
        
        logger.info("Encryption key rotated")
    
    def get_key_info(self) -> Dict[str, Any]:
        """Get information about current key."""
        if not self.encryption_enabled:
            return {'encryption_enabled': False}
        
        key_hash = hashlib.sha256(self.master_key).hexdigest()[:16]
        
        return {
            'encryption_enabled': True,
            'key_hash': key_hash,
            'key_file': str(self.key_file) if self.key_file else None,
            'algorithm': 'fernet'
        }


class SecureModelSerializer:
    """Secure serialization for ConfoRL models and data."""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        """Initialize secure model serializer.
        
        Args:
            encryption_manager: Encryption manager (creates default if None)
        """
        self.encryption_manager = encryption_manager or EncryptionManager()
        self.serialization_history = []
        
        logger.info("Secure model serializer initialized")
    
    def serialize_model(
        self,
        model: Any,
        file_path: Union[str, Path],
        encrypt: bool = True,
        include_checksum: bool = True
    ) -> Dict[str, Any]:
        """Securely serialize model to file.
        
        Args:
            model: Model object to serialize
            file_path: Path to save file
            encrypt: Whether to encrypt the serialized data
            include_checksum: Whether to include integrity checksum
            
        Returns:
            Serialization metadata
        """
        file_path = Path(file_path)
        
        try:
            # Serialize model to bytes
            if hasattr(model, '__getstate__') or hasattr(model, '__dict__'):
                # Use pickle for objects
                model_bytes = pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
                serialization_format = 'pickle'
            else:
                # Use JSON for simple data
                model_json = json.dumps(model, default=str)
                model_bytes = model_json.encode('utf-8')
                serialization_format = 'json'
            
            # Generate checksum
            checksum = None
            if include_checksum:
                checksum = hashlib.sha256(model_bytes).hexdigest()
            
            # Encrypt if requested
            encryption_metadata = None
            if encrypt and self.encryption_manager.encryption_enabled:
                model_bytes, encryption_metadata = self.encryption_manager.encrypt_data(model_bytes)
            elif encrypt:
                logger.warning("Encryption requested but not available")
            
            # Write to file
            with open(file_path, 'wb') as f:
                f.write(model_bytes)
            
            # Create metadata
            metadata = {
                'file_path': str(file_path),
                'file_size': len(model_bytes),
                'serialization_format': serialization_format,
                'encrypted': encrypt and self.encryption_manager.encryption_enabled,
                'checksum': checksum,
                'timestamp': time.time(),
                'encryption_metadata': asdict(encryption_metadata) if encryption_metadata else None
            }
            
            # Record serialization
            self.serialization_history.append(metadata.copy())
            
            logger.info(f"Model serialized to {file_path} (encrypted={metadata['encrypted']})")
            return metadata
            
        except Exception as e:
            raise ConfoRLError(f"Model serialization failed: {e}")
    
    def deserialize_model(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        verify_checksum: bool = True
    ) -> Any:
        """Securely deserialize model from file.
        
        Args:
            file_path: Path to model file
            metadata: Serialization metadata (optional)
            verify_checksum: Whether to verify integrity checksum
            
        Returns:
            Deserialized model
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        try:
            # Read file
            with open(file_path, 'rb') as f:
                file_bytes = f.read()
            
            # Decrypt if needed
            if metadata and metadata.get('encrypted', False):
                if not self.encryption_manager.encryption_enabled:
                    raise SecurityError("Cannot decrypt: encryption not available")
                
                encryption_metadata = EncryptionMetadata(**metadata['encryption_metadata'])
                file_bytes = self.encryption_manager.decrypt_data(file_bytes, encryption_metadata)
            
            # Verify checksum
            if verify_checksum and metadata and metadata.get('checksum'):
                computed_checksum = hashlib.sha256(file_bytes).hexdigest()
                if computed_checksum != metadata['checksum']:
                    raise SecurityError("File integrity check failed - checksum mismatch")
            
            # Deserialize based on format
            serialization_format = metadata.get('serialization_format', 'pickle') if metadata else 'pickle'
            
            if serialization_format == 'json':
                model_json = file_bytes.decode('utf-8')
                model = json.loads(model_json)
            else:
                # Default to pickle
                model = pickle.loads(file_bytes)
            
            logger.info(f"Model deserialized from {file_path}")
            return model
            
        except Exception as e:
            raise ConfoRLError(f"Model deserialization failed: {e}")
    
    def secure_copy(
        self,
        source_path: Union[str, Path],
        dest_path: Union[str, Path],
        verify_integrity: bool = True
    ) -> bool:
        """Securely copy model file with integrity verification.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            verify_integrity: Whether to verify copy integrity
            
        Returns:
            True if copy successful
        """
        source_path = Path(source_path)
        dest_path = Path(dest_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")
        
        try:
            # Read source file
            with open(source_path, 'rb') as f:
                source_data = f.read()
            
            # Compute source checksum
            source_checksum = hashlib.sha256(source_data).hexdigest()
            
            # Write to destination
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                f.write(source_data)
            
            # Verify copy if requested
            if verify_integrity:
                with open(dest_path, 'rb') as f:
                    dest_data = f.read()
                
                dest_checksum = hashlib.sha256(dest_data).hexdigest()
                
                if source_checksum != dest_checksum:
                    dest_path.unlink()  # Remove corrupted copy
                    raise SecurityError("File copy integrity check failed")
            
            logger.info(f"Secure copy: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Secure copy failed: {e}")
            return False
    
    def get_file_integrity_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get integrity information for a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            File integrity information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {'exists': False}
        
        try:
            stat = file_path.stat()
            
            # Read file for checksum
            with open(file_path, 'rb') as f:
                file_data = f.read()
            
            return {
                'exists': True,
                'size': stat.st_size,
                'modified_time': stat.st_mtime,
                'checksum_sha256': hashlib.sha256(file_data).hexdigest(),
                'checksum_md5': hashlib.md5(file_data).hexdigest()
            }
            
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    def cleanup_temp_files(self, temp_dir: Union[str, Path] = None) -> int:
        """Clean up temporary serialization files.
        
        Args:
            temp_dir: Directory to clean (default: system temp)
            
        Returns:
            Number of files cleaned up
        """
        import tempfile
        
        temp_dir = Path(temp_dir or tempfile.gettempdir())
        cleaned_count = 0
        
        try:
            # Look for ConfoRL temp files
            for temp_file in temp_dir.glob("conforl_temp_*"):
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Could not clean temp file {temp_file}: {e}")
            
            logger.info(f"Cleaned up {cleaned_count} temporary files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Temp file cleanup failed: {e}")
            return 0
    
    def get_serialization_stats(self) -> Dict[str, Any]:
        """Get serialization statistics."""
        if not self.serialization_history:
            return {'total_serializations': 0}
        
        total_size = sum(s.get('file_size', 0) for s in self.serialization_history)
        encrypted_count = sum(1 for s in self.serialization_history if s.get('encrypted', False))
        
        return {
            'total_serializations': len(self.serialization_history),
            'total_bytes_serialized': total_size,
            'encrypted_files': encrypted_count,
            'encryption_rate': encrypted_count / len(self.serialization_history),
            'recent_serializations': len([s for s in self.serialization_history if time.time() - s['timestamp'] < 3600])
        }


# Global instances
_global_encryption_manager = None
_global_model_serializer = None


def get_encryption_manager() -> EncryptionManager:
    """Get global encryption manager instance."""
    global _global_encryption_manager
    if _global_encryption_manager is None:
        _global_encryption_manager = EncryptionManager()
    return _global_encryption_manager


def get_model_serializer() -> SecureModelSerializer:
    """Get global model serializer instance."""
    global _global_model_serializer
    if _global_model_serializer is None:
        _global_model_serializer = SecureModelSerializer(get_encryption_manager())
    return _global_model_serializer