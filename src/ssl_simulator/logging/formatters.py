"""Logging formatters used across all projects."""

import logging

FORMATTERS = {
    "simple": logging.Formatter("%(message)s"),
    "compact": logging.Formatter("[%(levelname)s] %(message)s"),
    "standard": logging.Formatter("%(name)s - [%(levelname)s] %(message)s"),
    "detailed": logging.Formatter(
        "%(asctime)s - %(name)s - [%(levelname)s] "
        "(%(filename)s:%(funcName)s:%(lineno)d) %(message)s"
    ),
    "json": logging.Formatter(
        # Could add JSON formatter for structured logging
        "%(message)s"  # placeholder
    ),
}


def get_formatter(format_type: str = "simple") -> logging.Formatter:
    """Get a formatter by name."""
    if format_type not in FORMATTERS:
        raise ValueError(f"Unknown format: {format_type}. Available: {list(FORMATTERS.keys())}")
    return FORMATTERS[format_type]
