"""Comprehensive logging setup for ConfoRL."""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path
import json
import time


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
    json_logging: bool = False,
    include_console: bool = True
) -> logging.Logger:
    """Setup comprehensive logging for ConfoRL.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Specific log file path
        log_dir: Directory for log files (creates timestamped file)
        json_logging: Whether to use JSON format
        include_console: Whether to include console output
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ConfoRL")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Setup formatters
    if json_logging:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
    
    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file or log_dir:
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(exist_ok=True)
            timestamp = int(time.time())
            log_file = log_path / f"conforl_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.info(f"ConfoRL logging initialized: level={level}, file={log_file}")
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"ConfoRL.{name}")


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        return json.dumps(log_data)


class ContextualLogger:
    """Logger with contextual information for experiments."""
    
    def __init__(self, base_logger: logging.Logger, context: Dict[str, Any]):
        """Initialize contextual logger.
        
        Args:
            base_logger: Base logger instance
            context: Context information to include in logs
        """
        self.base_logger = base_logger
        self.context = context
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context information."""
        extra_data = self.context.copy()
        extra_data.update(kwargs)
        
        # Create log record with extra data
        record = self.base_logger.makeRecord(
            self.base_logger.name,
            level,
            "",  # pathname (not used)
            0,   # lineno (not used)
            message,
            (),  # args
            None,  # exc_info
            None,  # func
            extra_data
        )
        record.extra_data = extra_data
        
        self.base_logger.handle(record)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, **kwargs)


class PerformanceLogger:
    """Logger for performance metrics and timing."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize performance logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        self.timers = {}
    
    def start_timer(self, name: str):
        """Start a named timer.
        
        Args:
            name: Timer name
        """
        self.timers[name] = time.time()
    
    def end_timer(self, name: str, log_level: str = "INFO"):
        """End a named timer and log the duration.
        
        Args:
            name: Timer name
            log_level: Logging level for the duration message
        """
        if name not in self.timers:
            self.logger.warning(f"Timer '{name}' not found")
            return
        
        duration = time.time() - self.timers[name]
        del self.timers[name]
        
        log_func = getattr(self.logger, log_level.lower())
        log_func(f"Timer '{name}': {duration:.4f} seconds")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log performance metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
            prefix: Optional prefix for metric names
        """
        for name, value in metrics.items():
            metric_name = f"{prefix}{name}" if prefix else name
            self.logger.info(f"Metric {metric_name}: {value}")


class SecurityLogger:
    """Logger for security-related events."""
    
    def __init__(self, logger: logging.Logger):
        """Initialize security logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_access_attempt(
        self,
        resource: str,
        user: str = "unknown",
        allowed: bool = True,
        reason: str = ""
    ):
        """Log access attempt to a resource.
        
        Args:
            resource: Resource being accessed
            user: User attempting access
            allowed: Whether access was allowed
            reason: Reason for denial (if applicable)
        """
        status = "ALLOWED" if allowed else "DENIED"
        message = f"Access {status}: user='{user}' resource='{resource}'"
        
        if reason:
            message += f" reason='{reason}'"
        
        if allowed:
            self.logger.info(message)
        else:
            self.logger.warning(message)
    
    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO"
    ):
        """Log security-related event.
        
        Args:
            event_type: Type of security event
            description: Event description
            severity: Event severity (INFO, WARNING, ERROR, CRITICAL)
        """
        message = f"Security Event [{event_type}]: {description}"
        
        log_func = getattr(self.logger, severity.lower())
        log_func(message)
    
    def log_data_access(
        self,
        data_type: str,
        operation: str,
        user: str = "system"
    ):
        """Log data access operation.
        
        Args:
            data_type: Type of data accessed
            operation: Operation performed (read, write, delete)
            user: User performing operation
        """
        message = f"Data Access: user='{user}' operation='{operation}' data_type='{data_type}'"
        self.logger.info(message)