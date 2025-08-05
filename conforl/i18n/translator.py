"""Multi-language translation support for ConfoRL."""

import json
import os
from typing import Dict, Optional, Any, List
from pathlib import Path
import threading

from ..utils.logging import get_logger

logger = get_logger(__name__)


class Translator:
    """Multi-language translator for ConfoRL messages and UI."""
    
    def __init__(
        self,
        default_language: str = "en",
        translations_dir: Optional[str] = None
    ):
        """Initialize translator.
        
        Args:
            default_language: Default language code (ISO 639-1)
            translations_dir: Directory containing translation files
        """
        self.default_language = default_language
        self.current_language = default_language
        
        # Set translations directory
        if translations_dir is None:
            translations_dir = Path(__file__).parent / "translations"
        self.translations_dir = Path(translations_dir)
        
        # Translation storage
        self.translations = {}
        self.fallback_translations = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load translations
        self._load_translations()
        
        logger.info(f"Translator initialized with default language: {default_language}")
    
    def _load_translations(self):
        """Load translation files from directory."""
        if not self.translations_dir.exists():
            logger.warning(f"Translations directory not found: {self.translations_dir}")
            self._create_default_translations()
            return
        
        # Load all JSON translation files
        for lang_file in self.translations_dir.glob("*.json"):
            lang_code = lang_file.stem
            
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    translations = json.load(f)
                
                self.translations[lang_code] = translations
                logger.debug(f"Loaded translations for language: {lang_code}")
                
                # Set default language as fallback
                if lang_code == self.default_language:
                    self.fallback_translations = translations
                    
            except Exception as e:
                logger.error(f"Failed to load translations for {lang_code}: {e}")
        
        if not self.translations:
            logger.warning("No translations loaded, creating defaults")
            self._create_default_translations()
    
    def _create_default_translations(self):
        """Create default English translations."""
        default_translations = {
            # General messages
            "general": {
                "error": "Error",
                "warning": "Warning", 
                "info": "Information",
                "success": "Success",
                "loading": "Loading...",
                "complete": "Complete",
                "failed": "Failed",
                "unknown": "Unknown"
            },
            
            # Training messages
            "training": {
                "started": "Training started",
                "completed": "Training completed",
                "episode": "Episode",
                "step": "Step",
                "reward": "Reward",
                "risk": "Risk",
                "bound": "Risk Bound",
                "coverage": "Coverage",
                "progress": "Progress"
            },
            
            # Risk messages
            "risk": {
                "low": "Low Risk",
                "medium": "Medium Risk", 
                "high": "High Risk",
                "critical": "Critical Risk",
                "violation": "Risk Violation",
                "certificate": "Risk Certificate",
                "guarantee": "Safety Guarantee",
                "bound_exceeded": "Risk bound exceeded",
                "fallback_activated": "Fallback policy activated"
            },
            
            # Deployment messages
            "deployment": {
                "starting": "Starting deployment",
                "stopping": "Stopping deployment",  
                "monitoring": "Monitoring active",
                "alert": "Alert",
                "emergency_stop": "Emergency stop activated",
                "performance": "Performance",
                "safety_intervention": "Safety intervention"
            },
            
            # Configuration messages
            "config": {
                "invalid": "Invalid configuration",
                "missing": "Missing configuration",
                "loaded": "Configuration loaded",
                "saved": "Configuration saved",
                "parameter": "Parameter",
                "value": "Value"
            },
            
            # Error messages
            "errors": {
                "initialization_failed": "Initialization failed",
                "connection_failed": "Connection failed",
                "timeout": "Operation timed out",
                "insufficient_data": "Insufficient data",
                "invalid_input": "Invalid input",
                "permission_denied": "Permission denied",
                "file_not_found": "File not found",
                "network_error": "Network error"
            }
        }
        
        self.translations[self.default_language] = default_translations
        self.fallback_translations = default_translations
        
        # Save default translations
        self.translations_dir.mkdir(exist_ok=True)
        default_file = self.translations_dir / f"{self.default_language}.json"
        
        with open(default_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created default translations: {default_file}")
    
    def set_language(self, language_code: str) -> bool:
        """Set current language.
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            True if language was set successfully
        """
        with self._lock:
            if language_code in self.translations:
                self.current_language = language_code
                logger.info(f"Language set to: {language_code}")
                return True
            else:
                logger.warning(f"Language not available: {language_code}")
                return False
    
    def get_language(self) -> str:
        """Get current language code.
        
        Returns:
            Current language code
        """
        return self.current_language
    
    def get_available_languages(self) -> List[str]:
        """Get list of available language codes.
        
        Returns:
            List of available language codes
        """
        return list(self.translations.keys())
    
    def translate(
        self,
        key: str,
        language: Optional[str] = None,
        fallback: Optional[str] = None,
        **kwargs
    ) -> str:
        """Translate a message key.
        
        Args:
            key: Translation key (e.g., 'training.started')
            language: Language code (uses current if None)
            fallback: Fallback text if key not found
            **kwargs: Format parameters for the translated string
            
        Returns:
            Translated string
        """
        with self._lock:
            target_language = language or self.current_language
            
            # Get translation from target language
            translation = self._get_translation(key, target_language)
            
            # Fall back to default language
            if translation is None and target_language != self.default_language:
                translation = self._get_translation(key, self.default_language)
            
            # Use provided fallback
            if translation is None:
                translation = fallback or key
            
            # Format with parameters
            try:
                if kwargs:
                    translation = translation.format(**kwargs)
                return translation
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting failed for key '{key}': {e}")
                return translation
    
    def _get_translation(self, key: str, language: str) -> Optional[str]:
        """Get translation for key in specified language.
        
        Args:
            key: Translation key (dot-separated path)
            language: Language code
            
        Returns:
            Translation string or None if not found
        """
        if language not in self.translations:
            return None
        
        # Navigate nested dictionary using dot notation
        current = self.translations[language]
        for part in key.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current if isinstance(current, str) else None
    
    def add_translation(
        self,
        language: str,
        key: str,
        value: str,
        save: bool = True
    ):
        """Add or update a translation.
        
        Args:
            language: Language code
            key: Translation key (dot-separated path)
            value: Translation value
            save: Whether to save to file
        """
        with self._lock:
            if language not in self.translations:
                self.translations[language] = {}
            
            # Navigate and create nested structure
            current = self.translations[language]
            parts = key.split('.')
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
            
            if save:
                self._save_language_file(language)
            
            logger.debug(f"Added translation: {language}.{key} = {value}")
    
    def _save_language_file(self, language: str):
        """Save translations for a language to file.
        
        Args:
            language: Language code to save
        """
        if language not in self.translations:
            return
        
        self.translations_dir.mkdir(exist_ok=True)
        lang_file = self.translations_dir / f"{language}.json"
        
        try:
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(
                    self.translations[language],
                    f,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=True
                )
            logger.debug(f"Saved translations for language: {language}")
        except Exception as e:
            logger.error(f"Failed to save translations for {language}: {e}")
    
    def get_language_info(self, language: str) -> Dict[str, Any]:
        """Get information about a language.
        
        Args:
            language: Language code
            
        Returns:
            Dictionary with language information
        """
        language_names = {
            'en': {'name': 'English', 'native': 'English'},
            'es': {'name': 'Spanish', 'native': 'Español'},
            'fr': {'name': 'French', 'native': 'Français'},
            'de': {'name': 'German', 'native': 'Deutsch'},
            'ja': {'name': 'Japanese', 'native': '日本語'},
            'zh': {'name': 'Chinese', 'native': '中文'},
            'ru': {'name': 'Russian', 'native': 'Русский'},
            'pt': {'name': 'Portuguese', 'native': 'Português'},
            'it': {'name': 'Italian', 'native': 'Italiano'},
            'ko': {'name': 'Korean', 'native': '한국어'}
        }
        
        info = language_names.get(language, {
            'name': language.upper(),
            'native': language.upper()
        })
        
        info.update({
            'code': language,
            'available': language in self.translations,
            'translation_count': self._count_translations(language)
        })
        
        return info
    
    def _count_translations(self, language: str) -> int:
        """Count number of translations for a language.
        
        Args:
            language: Language code
            
        Returns:
            Number of translation keys
        """
        if language not in self.translations:
            return 0
        
        def count_keys(obj):
            if isinstance(obj, dict):
                return sum(count_keys(v) for v in obj.values())
            else:
                return 1
        
        return count_keys(self.translations[language])
    
    def export_translations(self, output_dir: str):
        """Export all translations to a directory.
        
        Args:
            output_dir: Directory to export to
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for language, translations in self.translations.items():
            lang_file = output_path / f"{language}.json"
            
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, indent=2, ensure_ascii=False, sort_keys=True)
        
        logger.info(f"Exported translations to: {output_dir}")


# Global translator instance
_global_translator = None
_translator_lock = threading.Lock()


def get_translator() -> Translator:
    """Get global translator instance.
    
    Returns:
        Global Translator instance
    """
    global _global_translator
    
    if _global_translator is None:
        with _translator_lock:
            if _global_translator is None:
                _global_translator = Translator()
    
    return _global_translator


def t(key: str, **kwargs) -> str:
    """Convenience function for translation.
    
    Args:
        key: Translation key
        **kwargs: Format parameters
        
    Returns:
        Translated string
    """
    return get_translator().translate(key, **kwargs)


def set_language(language_code: str) -> bool:
    """Set global language.
    
    Args:
        language_code: Language code to set
        
    Returns:
        True if successful
    """
    return get_translator().set_language(language_code)