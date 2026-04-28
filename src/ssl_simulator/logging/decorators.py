"""Logging decorators and utilities."""

import functools
import logging

from .levels import normalize_level


def requires_log_level(logger: logging.Logger, minimum_level: int | str):
    """Decorator: skip function if logger doesn't meet minimum level."""
    min_level = normalize_level(minimum_level) if isinstance(minimum_level, str) else minimum_level

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not logger.isEnabledFor(min_level):
                level_name = logging.getLevelName(min_level)
                logger.warning(
                    f"{func.__name__}() requires {level_name} level. "
                    f"Use LoggerManager().set_level('{level_name}') or setup_logging(level='{level_name}') from ssl_simulator.logging to enable."
                )
                return
            return func(*args, **kwargs)

        return wrapper

    return decorator
