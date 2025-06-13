"""
Security utilities for TalentScout AI Interview Assistant
Handles encryption, data management, and consent in a simplified way.
"""
import os
import json
import base64
import logging
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from typing import Dict, Any, Optional, List
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecurityManager:
    """Simplified security manager for handling encryption and sensitive data."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SecurityManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        # Create necessary directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize encryption
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        # Initialize audit log
        self.audit_log_file = os.path.join(self.base_dir, 'audit.log')
        self._initialized = True
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get existing encryption key or generate a new one."""
        key_file = os.path.join(self.base_dir, '.encryption_key')
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Restrict permissions
            return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data."""
        if not data:
            return ""
        return self.cipher_suite.encrypt(data.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data."""
        if not encrypted_data:
            return ""
        try:
            return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            return ""
    
    def log_audit_event(self, action: str, resource: str, status: str = "success", details: Optional[Dict] = None):
        """Log security-relevant events."""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'resource': resource,
            'status': status,
            'details': details or {}
        }
        
        try:
            with open(self.audit_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write to audit log: {e}")

class DataManager:
    """Manages storage and retrieval of candidate data with encryption."""
    
    def __init__(self):
        self.security = SecurityManager()
        self.candidates_file = os.path.join(self.security.data_dir, 'candidates.json')
        self._ensure_data_file()
    
    def _ensure_data_file(self):
        """Ensure the data file exists."""
        if not os.path.exists(self.candidates_file):
            with open(self.candidates_file, 'w') as f:
                json.dump([], f)
    
    def save_candidate_data(self, candidate_data: Dict[str, Any]) -> bool:
        """Save candidate data with encrypted sensitive fields."""
        try:
            # Encrypt sensitive data
            encrypted_data = candidate_data.copy()
            for field in ['name', 'email', 'phone', 'address']:
                if field in encrypted_data and encrypted_data[field]:
                    encrypted_data[f'encrypted_{field}'] = self.security.encrypt_data(encrypted_data[field])
                    encrypted_data[field] = ''  # Remove plaintext
            
            # Add metadata
            encrypted_data['last_updated'] = datetime.utcnow().isoformat()
            
            # Load existing data
            if os.path.exists(self.candidates_file):
                with open(self.candidates_file, 'r') as f:
                    data = json.load(f)
            else:
                data = []
            
            # Update or add candidate
            updated = False
            for i, candidate in enumerate(data):
                if candidate.get('candidate_id') == encrypted_data.get('candidate_id'):
                    data[i] = encrypted_data
                    updated = True
                    break
            
            if not updated:
                data.append(encrypted_data)
            
            # Save back to file
            with open(self.candidates_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            self.security.log_audit_event(
                'candidate_data_saved',
                f"candidate:{encrypted_data.get('candidate_id')}",
                'success',
                {'fields': list(encrypted_data.keys())}
            )
            return True
            
        except Exception as e:
            logger.error(f"Error saving candidate data: {e}", exc_info=True)
            self.security.log_audit_event(
                'candidate_data_save_failed',
                f"candidate:{candidate_data.get('candidate_id', 'unknown')}",
                'error',
                {'error': str(e), 'traceback': traceback.format_exc()}
            )
            return False
            self.security.log_audit_event(
                'candidate_data_save_failed',
                'candidate:unknown',
                'error',
                {'error': str(e)}
            )
            return False
    
    def get_candidate_data(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve candidate data with decrypted sensitive fields."""
        try:
            if not os.path.exists(self.candidates_file):
                return None
                
            with open(self.candidates_file, 'r') as f:
                candidates = json.load(f)
            
            for candidate in candidates:
                if candidate.get('candidate_id') == candidate_id:
                    # Decrypt sensitive data
                    decrypted = candidate.copy()
                    for field in ['name', 'email', 'phone', 'address']:
                        enc_field = f'encrypted_{field}'
                        if enc_field in decrypted and decrypted[enc_field]:
                            decrypted[field] = self.security.decrypt_data(decrypted[enc_field])
                            del decrypted[enc_field]
                    return decrypted
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving candidate data: {e}")
            return None

# Singleton instances
security_manager = SecurityManager()
data_manager = DataManager()
