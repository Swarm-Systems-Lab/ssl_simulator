"""
Convenience functions for logging setup.
Use these for quick configuration without creating LoggerManager instance.
"""

import logging

from .formatters import get_formatter
from .levels import normalize_level
from .manager import LoggerManager

_logger = logging.getLogger("ssl_simulator")


def set_log_level(level: int | str = logging.INFO) -> None:
    """
    Set ssl_simulator logging level.

    Examples
    --------
    >>> set_log_level("DEBUG")
    >>> set_log_level(logging.INFO)
    """
    _initialize_if_needed()
    _logger.setLevel(normalize_level(level))


def set_log_format(format_type: str = "simple") -> None:
    """
    Set ssl_simulator logging format.

    Examples
    --------
    >>> set_log_format("standard")
    >>> set_log_format("detailed")
    """
    _initialize_if_needed()
    for handler in _logger.handlers:
        handler.setFormatter(get_formatter(format_type))


def _initialize_if_needed() -> None:
    """Auto-initialize logger if not already done."""
    if not _logger.handlers:
        LoggerManager().setup(level=logging.INFO, format_type="simple")
