"""Logging configuration and utilities for ssl_simulator."""

import functools
import logging
import sys

__all__: list[str] = [
    "debug_verbose",
    "disable_logging",
    "enable_logging",
    "requires_log_level",
    "set_log_format",
    "set_log_level",
    "setup_logging",
]

DEBUG_VERBOSE = 5
logging.addLevelName(DEBUG_VERBOSE, "DEBUG_VERBOSE")

# Package logger
_logger = logging.getLogger("ssl_simulator")

# Standard formatter presets
_FORMATTERS = {
    "simple": logging.Formatter("%(message)s"),
    "standard": logging.Formatter("%(name)s - [%(levelname)s] %(message)s"),
    "detailed": logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] (%(filename)s:%(funcName)s:%(lineno)d) %(message)s"
    ),
    "compact": logging.Formatter("[%(levelname)s] %(message)s"),
}


def setup_logging() -> None:
    """
    Initialize default logging configuration for ssl_simulator.

    This is called automatically when the package is imported.
    Uses the 'simple' formatter by default.
    Users can override by calling logging.basicConfig() or configuring the logger directly.
    """
    if not _logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_FORMATTERS["simple"])
        _logger.addHandler(handler)
        _logger.setLevel(logging.INFO)
        _logger.propagate = False


def set_log_format(format_type: str = "simple") -> None:
    """
    Set the logging format for ssl_simulator package.

    Parameters
    ----------
    format_type : str
        The format preset to use. Options:
        - "simple": Just the message (default)
        - "simple_level": Level and message only
        - "standard": Timestamp, logger name, level, message
        - "detailed": Timestamp, logger name, level, file, function, line number, message
        - "compact": Level and message only
    """
    if format_type not in _FORMATTERS:
        raise ValueError(
            f"Unknown format '{format_type}'. Available formats: {', '.join(_FORMATTERS.keys())}"
        )

    formatter = _FORMATTERS[format_type]
    for handler in _logger.handlers:
        handler.setFormatter(formatter)


def set_log_level(level: int | str = logging.INFO) -> None:
    """
    Set the logging level for ssl_simulator package.

    Parameters
    ----------
    level : int or str
        Logging level (e.g., logging.DEBUG, logging.INFO, "DEBUG", "INFO").
        Supports custom level "DEBUG_VERBOSE".
        CRITICAL=50, ERROR=40, WARNING=30, INFO=20, DEBUG=10, DEBUG_VERBOSE=5, NOTSET=0
    """
    if isinstance(level, str):
        normalized_level = level.upper()
        if normalized_level == "DEBUG_VERBOSE":
            level = DEBUG_VERBOSE
        else:
            level = getattr(logging, normalized_level)
    _logger.setLevel(level)


def debug_verbose(message: str, *args, **kwargs) -> None:
    """Log a message at DEBUG_VERBOSE level."""
    _logger.log(DEBUG_VERBOSE, message, *args, **kwargs)


def disable_logging() -> None:
    """Disable all logging output from ssl_simulator package."""
    _logger.setLevel(logging.CRITICAL + 1)


def enable_logging() -> None:
    """Re-enable logging output from ssl_simulator package (INFO level)."""
    _logger.setLevel(logging.INFO)


def requires_log_level(minimum_level: int | str):
    """
    Decorator: skip execution if logger doesn't meet minimum level.

    Parameters
    ----------
    minimum_level : int or str
        Minimum log level required to execute the decorated function.
        Example: requires_log_level(logging.DEBUG) or requires_log_level("DEBUG_VERBOSE")

    Returns
    -------
        Decorator function that wraps the target function.

    Examples
    --------
    @requires_log_level(logging.DEBUG)
    def debug_artists(self):
        logger.debug("Detailed debug info...")
    """
    if isinstance(minimum_level, str):
        normalized = minimum_level.upper()
        if normalized == "DEBUG_VERBOSE":
            minimum_level = DEBUG_VERBOSE
        else:
            minimum_level = getattr(logging, normalized)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not _logger.isEnabledFor(minimum_level):
                level_name = logging.getLevelName(minimum_level)
                _logger.warning(
                    f"{func.__name__}() requires {level_name} log level. "
                    f"Use set_log_level('{level_name}')"
                )
                return
            return func(*args, **kwargs)

        return wrapper

    return decorator
