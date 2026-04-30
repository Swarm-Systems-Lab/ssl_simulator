"""
Convenience functions for logging setup.
Use these for quick configuration without creating LoggerManager instance.
"""

import logging

from ssl_simulator.config import CONFIG

from .formatters import get_formatter
from .levels import normalize_level
from .manager import LoggerManager

_logger = logging.getLogger("ssl_simulator")


def _initialize_if_needed() -> None:
    """Auto-initialize logger if not already done."""
    if not _logger.handlers:
        LoggerManager().setup(level=logging.INFO, format_type="simple")


def setup_logging(
    level: int | str = logging.INFO,
    format_type: str = "standard",
    inline_max_len: int | None = None,
    inline_max_keys: int | None = None,
) -> None:
    """Configure the root logger with a project formatter. Idempotent."""
    _initialize_if_needed()
    for handler in _logger.handlers:
        handler.setFormatter(get_formatter(format_type))
    _logger.setLevel(normalize_level(level))
    if inline_max_len is not None:
        CONFIG["LOG_INLINE_MAX_LEN"] = inline_max_len
    if inline_max_keys is not None:
        CONFIG["LOG_INLINE_MAX_KEYS"] = inline_max_keys


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
