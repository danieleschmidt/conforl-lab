"""Localized formatting utilities for numbers, dates, and text."""

import locale
from typing import Dict, Any, Optional, Union
from datetime import datetime, date, time
from decimal import Decimal
import re

from ..utils.logging import get_logger

logger = get_logger(__name__)


class LocalizedFormatter:
    """Base class for localized formatting."""
    
    def __init__(self, language_code: str = "en", region_code: Optional[str] = None):
        """Initialize localized formatter.
        
        Args:
            language_code: ISO 639-1 language code
            region_code: ISO 3166-1 alpha-2 country code
        """
        self.language_code = language_code
        self.region_code = region_code or self._get_default_region(language_code)
        self.locale_code = f"{language_code}_{self.region_code}"
        
        # Try to set locale
        self._setup_locale()
        
        logger.debug(f"Initialized formatter for locale: {self.locale_code}")
    
    def _get_default_region(self, language_code: str) -> str:
        """Get default region for language code.
        
        Args:
            language_code: Language code
            
        Returns:
            Default region code
        """
        defaults = {
            'en': 'US',
            'es': 'ES',
            'fr': 'FR',
            'de': 'DE',
            'ja': 'JP',
            'zh': 'CN',
            'ru': 'RU',
            'pt': 'BR',
            'it': 'IT',
            'ko': 'KR'
        }
        return defaults.get(language_code, 'US')
    
    def _setup_locale(self):
        """Setup system locale if available."""
        try:
            locale.setlocale(locale.LC_ALL, f"{self.locale_code}.UTF-8")
            logger.debug(f"Set system locale to: {self.locale_code}.UTF-8")
        except locale.Error:
            try:
                # Try without encoding
                locale.setlocale(locale.LC_ALL, self.locale_code)
                logger.debug(f"Set system locale to: {self.locale_code}")
            except locale.Error:
                logger.warning(f"Could not set locale {self.locale_code}, using default")


class NumberFormatter(LocalizedFormatter):
    """Localized number formatting."""
    
    def __init__(self, language_code: str = "en", region_code: Optional[str] = None):
        """Initialize number formatter."""
        super().__init__(language_code, region_code)
        
        # Locale-specific formatting rules
        self.decimal_separators = {
            'en_US': '.', 'en_GB': '.', 'es_ES': ',', 'fr_FR': ',',
            'de_DE': ',', 'ja_JP': '.', 'zh_CN': '.', 'ru_RU': ',',
            'pt_BR': ',', 'it_IT': ',', 'ko_KR': '.'
        }
        
        self.thousand_separators = {
            'en_US': ',', 'en_GB': ',', 'es_ES': '.', 'fr_FR': ' ',
            'de_DE': '.', 'ja_JP': ',', 'zh_CN': ',', 'ru_RU': ' ',
            'pt_BR': '.', 'it_IT': '.', 'ko_KR': ','
        }
        
        self.currency_symbols = {
            'en_US': '$', 'en_GB': '£', 'es_ES': '€', 'fr_FR': '€',
            'de_DE': '€', 'ja_JP': '¥', 'zh_CN': '¥', 'ru_RU': '₽',
            'pt_BR': 'R$', 'it_IT': '€', 'ko_KR': '₩'
        }
    
    def format_number(
        self,
        number: Union[int, float, Decimal],
        decimal_places: Optional[int] = None,
        use_grouping: bool = True
    ) -> str:
        """Format number according to locale.
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places (auto-detect if None)
            use_grouping: Whether to use thousand separators
            
        Returns:
            Formatted number string
        """
        try:
            # Try system locale formatting first
            if decimal_places is not None:
                return locale.format_string(f"%.{decimal_places}f", float(number), grouping=use_grouping)
            else:
                return locale.format_string("%.10g", float(number), grouping=use_grouping)
        
        except (locale.Error, ValueError):
            # Fall back to manual formatting
            return self._manual_number_format(number, decimal_places, use_grouping)
    
    def _manual_number_format(
        self,
        number: Union[int, float, Decimal],
        decimal_places: Optional[int],
        use_grouping: bool
    ) -> str:
        """Manual number formatting when locale is not available.
        
        Args:
            number: Number to format
            decimal_places: Number of decimal places
            use_grouping: Whether to use grouping
            
        Returns:
            Formatted number string
        """
        # Convert to string
        if decimal_places is not None:
            number_str = f"{float(number):.{decimal_places}f}"
        else:
            number_str = str(float(number))
        
        # Split integer and decimal parts
        if '.' in number_str:
            integer_part, decimal_part = number_str.split('.')
        else:
            integer_part, decimal_part = number_str, ''
        
        # Add thousand separators
        if use_grouping and len(integer_part) > 3:
            thousand_sep = self.thousand_separators.get(self.locale_code, ',')
            
            # Add separators from right to left
            formatted_integer = ''
            for i, digit in enumerate(reversed(integer_part)):
                if i > 0 and i % 3 == 0:
                    formatted_integer = thousand_sep + formatted_integer
                formatted_integer = digit + formatted_integer
            
            integer_part = formatted_integer
        
        # Reconstruct number with appropriate decimal separator
        if decimal_part:
            decimal_sep = self.decimal_separators.get(self.locale_code, '.')
            return f"{integer_part}{decimal_sep}{decimal_part}"
        else:
            return integer_part
    
    def format_currency(
        self,
        amount: Union[int, float, Decimal],
        currency_code: Optional[str] = None,
        decimal_places: int = 2
    ) -> str:
        """Format currency amount.
        
        Args:
            amount: Amount to format
            currency_code: Currency code (e.g., 'USD', 'EUR')
            decimal_places: Number of decimal places
            
        Returns:
            Formatted currency string
        """
        formatted_number = self.format_number(amount, decimal_places, use_grouping=True)
        
        # Get currency symbol
        if currency_code:
            symbol = self._get_currency_symbol(currency_code)
        else:
            symbol = self.currency_symbols.get(self.locale_code, '$')
        
        # Format based on locale conventions
        if self.language_code in ['en']:
            return f"{symbol}{formatted_number}"
        elif self.language_code in ['fr', 'es', 'it']:
            return f"{formatted_number} {symbol}"
        elif self.language_code in ['de']:
            return f"{formatted_number} {symbol}"
        else:
            return f"{symbol}{formatted_number}"
    
    def _get_currency_symbol(self, currency_code: str) -> str:
        """Get currency symbol for currency code.
        
        Args:
            currency_code: ISO 4217 currency code
            
        Returns:
            Currency symbol
        """
        symbols = {
            'USD': '$', 'EUR': '€', 'GBP': '£', 'JPY': '¥',
            'CNY': '¥', 'RUB': '₽', 'BRL': 'R$', 'KRW': '₩',
            'INR': '₹', 'CAD': 'C$', 'AUD': 'A$', 'CHF': 'CHF'
        }
        return symbols.get(currency_code, currency_code)
    
    def format_percentage(
        self,
        value: Union[int, float, Decimal],
        decimal_places: int = 1
    ) -> str:
        """Format percentage value.
        
        Args:
            value: Percentage value (0.05 = 5%)
            decimal_places: Number of decimal places
            
        Returns:
            Formatted percentage string
        """
        percentage = float(value) * 100
        formatted_number = self.format_number(percentage, decimal_places, use_grouping=False)
        
        if self.language_code in ['fr']:
            return f"{formatted_number} %"
        else:
            return f"{formatted_number}%"


class DateTimeFormatter(LocalizedFormatter):
    """Localized date and time formatting."""
    
    def __init__(self, language_code: str = "en", region_code: Optional[str] = None):
        """Initialize datetime formatter."""
        super().__init__(language_code, region_code)
        
        # Locale-specific format patterns
        self.date_formats = {
            'en_US': '%m/%d/%Y',
            'en_GB': '%d/%m/%Y',
            'es_ES': '%d/%m/%Y',
            'fr_FR': '%d/%m/%Y',
            'de_DE': '%d.%m.%Y',
            'ja_JP': '%Y/%m/%d',
            'zh_CN': '%Y-%m-%d',
            'ru_RU': '%d.%m.%Y',
            'pt_BR': '%d/%m/%Y',
            'it_IT': '%d/%m/%Y',
            'ko_KR': '%Y.%m.%d'
        }
        
        self.time_formats = {
            'en_US': '%I:%M %p',
            'en_GB': '%H:%M',
            'es_ES': '%H:%M',
            'fr_FR': '%H:%M',
            'de_DE': '%H:%M',
            'ja_JP': '%H:%M',
            'zh_CN': '%H:%M',
            'ru_RU': '%H:%M',
            'pt_BR': '%H:%M',
            'it_IT': '%H:%M',
            'ko_KR': '%H:%M'
        }
        
        self.datetime_formats = {
            'en_US': '%m/%d/%Y %I:%M %p',
            'en_GB': '%d/%m/%Y %H:%M',
            'es_ES': '%d/%m/%Y %H:%M',
            'fr_FR': '%d/%m/%Y %H:%M',
            'de_DE': '%d.%m.%Y %H:%M',
            'ja_JP': '%Y/%m/%d %H:%M',
            'zh_CN': '%Y-%m-%d %H:%M',
            'ru_RU': '%d.%m.%Y %H:%M',
            'pt_BR': '%d/%m/%Y %H:%M',
            'it_IT': '%d/%m/%Y %H:%M',
            'ko_KR': '%Y.%m.%d %H:%M'
        }
    
    def format_date(self, date_obj: Union[datetime, date]) -> str:
        """Format date according to locale.
        
        Args:
            date_obj: Date object to format
            
        Returns:
            Formatted date string
        """
        format_pattern = self.date_formats.get(self.locale_code, '%Y-%m-%d')
        return date_obj.strftime(format_pattern)
    
    def format_time(self, time_obj: Union[datetime, time]) -> str:
        """Format time according to locale.
        
        Args:
            time_obj: Time object to format
            
        Returns:
            Formatted time string
        """
        format_pattern = self.time_formats.get(self.locale_code, '%H:%M')
        return time_obj.strftime(format_pattern)
    
    def format_datetime(self, datetime_obj: datetime) -> str:
        """Format datetime according to locale.
        
        Args:
            datetime_obj: Datetime object to format
            
        Returns:
            Formatted datetime string
        """
        format_pattern = self.datetime_formats.get(self.locale_code, '%Y-%m-%d %H:%M')
        return datetime_obj.strftime(format_pattern)
    
    def format_relative_time(self, datetime_obj: datetime) -> str:
        """Format relative time (e.g., '2 hours ago').
        
        Args:
            datetime_obj: Datetime object to format
            
        Returns:
            Relative time string
        """
        now = datetime.now()
        diff = now - datetime_obj
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return self._get_relative_text('seconds', int(seconds))
        elif seconds < 3600:
            return self._get_relative_text('minutes', int(seconds // 60))
        elif seconds < 86400:
            return self._get_relative_text('hours', int(seconds // 3600))
        elif seconds < 2592000:  # 30 days
            return self._get_relative_text('days', int(seconds // 86400))
        elif seconds < 31536000:  # 365 days
            return self._get_relative_text('months', int(seconds // 2592000))
        else:
            return self._get_relative_text('years', int(seconds // 31536000))
    
    def _get_relative_text(self, unit: str, value: int) -> str:
        """Get relative time text for language.
        
        Args:
            unit: Time unit (seconds, minutes, hours, days, months, years)
            value: Numeric value
            
        Returns:
            Localized relative time text
        """
        if self.language_code == 'en':
            unit_names = {
                'seconds': 'second' if value == 1 else 'seconds',
                'minutes': 'minute' if value == 1 else 'minutes',
                'hours': 'hour' if value == 1 else 'hours',
                'days': 'day' if value == 1 else 'days',
                'months': 'month' if value == 1 else 'months',
                'years': 'year' if value == 1 else 'years'
            }
            return f"{value} {unit_names[unit]} ago"
        
        elif self.language_code == 'es':
            unit_names = {
                'seconds': 'segundo' if value == 1 else 'segundos',
                'minutes': 'minuto' if value == 1 else 'minutos',
                'hours': 'hora' if value == 1 else 'horas',
                'days': 'día' if value == 1 else 'días',
                'months': 'mes' if value == 1 else 'meses',
                'years': 'año' if value == 1 else 'años'
            }
            return f"hace {value} {unit_names[unit]}"
        
        elif self.language_code == 'fr':
            unit_names = {
                'seconds': 'seconde' if value == 1 else 'secondes',
                'minutes': 'minute' if value == 1 else 'minutes',
                'hours': 'heure' if value == 1 else 'heures',
                'days': 'jour' if value == 1 else 'jours',
                'months': 'mois', 
                'years': 'an' if value == 1 else 'ans'
            }
            return f"il y a {value} {unit_names[unit]}"
        
        elif self.language_code == 'de':
            unit_names = {
                'seconds': 'Sekunde' if value == 1 else 'Sekunden',
                'minutes': 'Minute' if value == 1 else 'Minuten',
                'hours': 'Stunde' if value == 1 else 'Stunden',
                'days': 'Tag' if value == 1 else 'Tage',
                'months': 'Monat' if value == 1 else 'Monate',
                'years': 'Jahr' if value == 1 else 'Jahre'
            }
            return f"vor {value} {unit_names[unit]}"
        
        else:
            # Default to English
            unit_names = {
                'seconds': 'second' if value == 1 else 'seconds',
                'minutes': 'minute' if value == 1 else 'minutes',
                'hours': 'hour' if value == 1 else 'hours',
                'days': 'day' if value == 1 else 'days',
                'months': 'month' if value == 1 else 'months',
                'years': 'year' if value == 1 else 'years'
            }
            return f"{value} {unit_names[unit]} ago"


class TextFormatter(LocalizedFormatter):
    """Localized text formatting utilities."""
    
    def __init__(self, language_code: str = "en", region_code: Optional[str] = None):
        """Initialize text formatter."""
        super().__init__(language_code, region_code)
        
        # Text direction
        self.rtl_languages = {'ar', 'he', 'fa', 'ur'}
        self.is_rtl = language_code in self.rtl_languages
    
    def format_list(self, items: list, conjunction: str = 'and') -> str:
        """Format list of items with proper conjunction.
        
        Args:
            items: List of items to format
            conjunction: Conjunction word ('and', 'or')
            
        Returns:
            Formatted list string
        """
        if not items:
            return ""
        
        if len(items) == 1:
            return str(items[0])
        
        conjunction_word = self._get_conjunction(conjunction)
        
        if len(items) == 2:
            return f"{items[0]} {conjunction_word} {items[1]}"
        
        # Oxford comma handling by language
        if self.language_code == 'en':
            return f"{', '.join(map(str, items[:-1]))}, {conjunction_word} {items[-1]}"
        else:
            return f"{', '.join(map(str, items[:-1]))} {conjunction_word} {items[-1]}"
    
    def _get_conjunction(self, conjunction: str) -> str:
        """Get localized conjunction word.
        
        Args:
            conjunction: English conjunction
            
        Returns:
            Localized conjunction
        """
        translations = {
            'en': {'and': 'and', 'or': 'or'},
            'es': {'and': 'y', 'or': 'o'},
            'fr': {'and': 'et', 'or': 'ou'},
            'de': {'and': 'und', 'or': 'oder'},
            'ja': {'and': 'と', 'or': 'または'},
            'zh': {'and': '和', 'or': '或'},
            'ru': {'and': 'и', 'or': 'или'},
            'pt': {'and': 'e', 'or': 'ou'},
            'it': {'and': 'e', 'or': 'o'},
            'ko': {'and': '그리고', 'or': '또는'}
        }
        
        lang_translations = translations.get(self.language_code, translations['en'])
        return lang_translations.get(conjunction, conjunction)
    
    def pluralize(self, count: int, singular: str, plural: Optional[str] = None) -> str:
        """Get pluralized form based on count.
        
        Args:
            count: Number of items
            singular: Singular form
            plural: Plural form (auto-generated if None)
            
        Returns:
            Appropriate form based on count
        """
        if count == 1:
            return singular
        
        if plural is not None:
            return plural
        
        # Simple English pluralization rules
        if self.language_code == 'en':
            if singular.endswith(('s', 'sh', 'ch', 'x', 'z')):
                return singular + 'es'
            elif singular.endswith('y') and len(singular) > 1 and singular[-2] not in 'aeiou':
                return singular[:-1] + 'ies'
            elif singular.endswith('f'):
                return singular[:-1] + 'ves'
            elif singular.endswith('fe'):
                return singular[:-2] + 'ves'
            else:
                return singular + 's'
        
        # For other languages, just return singular + 's' as fallback
        return singular + 's'