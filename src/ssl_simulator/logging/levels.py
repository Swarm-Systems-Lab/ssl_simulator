"""Custom logging levels used across all projects."""

import logging

DEBUG_VERBOSE = 5
logging.addLevelName(DEBUG_VERBOSE, "DEBUG_VERBOSE")

# Standard levels for reference
LEVELS = {
    "CRITICAL": logging.CRITICAL,  # 50
    "ERROR": logging.ERROR,  # 40
    "WARNING": logging.WARNING,  # 30
    "INFO": logging.INFO,  # 20
    "DEBUG": logging.DEBUG,  # 10
    "DEBUG_VERBOSE": DEBUG_VERBOSE,
    "NOTSET": logging.NOTSET,  # 0
}


def normalize_level(level: int | str) -> int:
    """Convert string level names to logging level integers."""
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if normalized not in LEVELS:
        raise ValueError(f"Unknown level: {level}. Options: {list(LEVELS.keys())}")
    return LEVELS[normalized]
