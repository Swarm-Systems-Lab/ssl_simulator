"""
ssl_simulator logging infrastructure.

Provides:
- Custom log levels (DEBUG_VERBOSE)
- Shared formatters
- LoggerManager for centralized setup
- Decorators
- Utilities for standalone quick configuration
"""

from .decorators import debug_verbose, requires_log_level
from .formatters import FORMATTERS, get_formatter
from .levels import DEBUG_VERBOSE, LEVELS, normalize_level
from .manager import LoggerManager
from .utils import set_log_format, set_log_level

__all__ = [  # noqa: RUF022
    # Classes
    "LoggerManager",
    # Convenience functions
    "set_log_level",
    "set_log_format",
    # Utilities
    "debug_verbose",
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
