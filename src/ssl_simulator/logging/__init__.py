"""
ssl_simulator logging infrastructure.

Provides:
- Custom log levels (DEBUG_VERBOSE)
- Shared formatters
- LoggerManager for centralized setup
- Decorators
- Utilities for standalone quick configuration
"""

from .decorators import requires_log_level
from .formatters import FORMATTERS, get_formatter
from .levels import DEBUG_VERBOSE, LEVELS, normalize_level
from .manager import LoggerManager
from .utils import set_log_format, set_log_level, setup_logging

__all__ = [  # noqa: RUF022
    # Classes
    "LoggerManager",
    # Convenience functions
    "set_log_level",
    "set_log_format",
    "setup_logging",
    # Utilities
    "get_formatter",
    "normalize_level",
    # Decorators
    "requires_log_level",
    # Constants
    "DEBUG_VERBOSE",
    "FORMATTERS",
    "LEVELS",
]

LoggerManager().setup()
