"""
Consent management for TalentScout AI Interview Assistant.
Handles collection, storage, and verification of user consent.
"""
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from security_utils import security_manager

class ConsentManager:
    """Manages user consent for data processing."""
    
    def __init__(self):
        self.consents_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'data',
            'consents.json'
        )
        self._ensure_consents_file()
        self.required_consents = [
            {
                'id': 'data_processing',
                'title': 'Data Processing Consent',
                'description': 'I consent to the processing of my personal data for the interview process.',
                'required': True,
                'expires_after_days': 365
            },
            {
                'id': 'data_storage',
                'title': 'Data Storage Consent',
                'description': 'I consent to the storage of my interview data for evaluation purposes.',
                'required': True,
                'expires_after_days': 90
            }
        ]
    
    def _ensure_consents_file(self):
        """Ensure the consents file exists."""
        os.makedirs(os.path.dirname(self.consents_file), exist_ok=True)
        if not os.path.exists(self.consents_file):
            with open(self.consents_file, 'w') as f:
                json.dump([], f)
    
    def get_required_consents(self) -> List[Dict[str, Any]]:
        """Get the list of required consents."""
        return self.required_consents
    
    def record_consent(self, user_id: str, consent_type: str, granted: bool, purpose: str) -> bool:
        """Record user consent.
        
        Args:
            user_id: Unique identifier for the user
            consent_type: Type of consent (e.g., 'data_processing')
            granted: Whether consent was granted
            purpose: Purpose of the consent
            
        Returns:
            bool: True if consent was recorded successfully, False otherwise
        """
        try:
            consent_data = {
                'user_id': user_id,
                'consent_type': consent_type,
                'granted': granted,
                'purpose': purpose,
                'timestamp': datetime.utcnow().isoformat(),
                'expires_at': (datetime.utcnow() + timedelta(days=365)).isoformat()
            }
            return self.save_consent(user_id, consent_data)
        except Exception as e:
            logger.error(f"Error recording consent: {e}", exc_info=True)
            return False
            
    def save_consent(self, candidate_id: str, consent_data: Dict[str, Any]) -> bool:
        """Save user consent."""
        try:
            # Load existing consents
            if os.path.exists(self.consents_file):
                with open(self.consents_file, 'r') as f:
                    consents = json.load(f)
            else:
                consents = []
            
            # Update or add consent
            existing = next(
                (c for c in consents if c.get('candidate_id') == candidate_id),
                None
            )
            
            consent_entry = {
                'candidate_id': candidate_id,
                'consent_granted': True,
                'consent_timestamp': datetime.utcnow().isoformat(),
                'consents': {
                    consent['id']: {
                        'granted': True,
                        'timestamp': datetime.utcnow().isoformat(),
                        'expires_at': (
                            datetime.utcnow() + 
                            timedelta(days=consent.get('expires_after_days', 90))
                        ).isoformat()
                    }
                    for consent in self.required_consents
                }
            }
            
            if existing:
                consents.remove(existing)
            consents.append(consent_entry)
            
            # Save back to file
            with open(self.consents_file, 'w') as f:
                json.dump(consents, f, indent=2)
            
            security_manager.log_audit_event(
                'consent_saved',
                f'candidate:{candidate_id}',
                'success',
                {'consent_ids': list(consent_entry['consents'].keys())}
            )
            
            return True
            
        except Exception as e:
            security_manager.log_audit_event(
                'consent_save_failed',
                f'candidate:{candidate_id}',
                'error',
                {'error': str(e)}
            )
            return False
    
    def has_consent(self, candidate_id: str) -> bool:
        """Check if a candidate has given all required consents."""
        try:
            if not os.path.exists(self.consents_file):
                return False
                
            with open(self.consents_file, 'r') as f:
                consents = json.load(f)
            
            consent_entry = next(
                (c for c in consents if c.get('candidate_id') == candidate_id),
                None
            )
            
            if not consent_entry or not consent_entry.get('consent_granted', False):
                return False
            
            # Check if all required consents are present and not expired
            required_ids = {c['id'] for c in self.required_consents}
            given_ids = set(consent_entry.get('consents', {}).keys())
            
            if not required_ids.issubset(given_ids):
                return False
            
            # Check if any consent is expired
            now = datetime.utcnow()
            for consent in consent_entry['consents'].values():
                expires_at = datetime.fromisoformat(consent['expires_at'])
                if now > expires_at:
                    return False
            
            return True
            
        except Exception as e:
            security_manager.log_audit_event(
                'consent_check_failed',
                f'candidate:{candidate_id}',
                'error',
                {'error': str(e)}
            )
            return False

# Singleton instance
consent_manager = ConsentManager()
