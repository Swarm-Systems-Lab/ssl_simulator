# ssl_simulator/logging/manager.py
"""Centralized logging for ssl_simulator and dependent apps."""

import logging
import sys
from typing import Optional

from ssl_simulator.config import CONFIG

from .formatters import get_formatter
from .levels import normalize_level


class LoggerManager:
    """
    Manages logging specifically for ssl_simulator and ssl_vista.

    Does NOT touch the root logger or other packages.
    Each framework package gets its own handler.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._handlers = {}  # Store handlers per package
        self._initialized = True

    def setup(
        self,
        level: int | str = logging.INFO,
        format_type: str = "compact",
        packages: list[str] | None = None,
        handler: logging.Handler | None = None,
        inline_max_len: int | None = None,
        inline_max_keys: int | None = None,
    ) -> None:
        """
        Configure logging for specific packages (default: ssl_simulator, ssl_vista).

        This configures ONLY these packages, leaving other packages silent.

        Parameters
        ----------
        level : int | str
            Log level for the configured packages
        format_type : str
            Formatter preset (simple, compact, standard, detailed, json)
        packages : list[str], optional
            Package names to configure (default: ["ssl_simulator"])
        handler : logging.Handler, optional
            Custom handler (default: StreamHandler to stdout)
        inline_max_len : int, optional
            Maximum line length for inline dict rendering. Edits CONFIG["LOG_INLINE_MAX_LEN"].
        inline_max_keys : int, optional
            Maximum keys to keep dicts inline. Edits CONFIG["LOG_INLINE_MAX_KEYS"].

        Examples
        --------
        >>> LoggerManager().setup(level="DEBUG")
        >>> # Only ssl_simulator logs at DEBUG
        """
        if packages is None:
            packages = ["ssl_simulator"]

        if inline_max_len is not None:
            CONFIG["LOG_INLINE_MAX_LEN"] = inline_max_len
        if inline_max_keys is not None:
            CONFIG["LOG_INLINE_MAX_KEYS"] = inline_max_keys

        formatter = get_formatter(format_type)
        normalized_level = normalize_level(level)

        if handler is None:
            handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        for package_name in packages:
            logger = logging.getLogger(package_name)

            # Remove existing handlers to avoid duplicates
            for h in logger.handlers[:]:
                logger.removeHandler(h)

            # Add handler only to this package logger
            logger.addHandler(handler)
            logger.setLevel(normalized_level)

            # Prevent propagation to root (so root's config doesn't interfere)
            logger.propagate = False

            self._handlers[package_name] = handler

    def set_level(self, level: int | str, packages: list[str] | None = None) -> None:
        """Set level for specific packages."""
        if packages is None:
            packages = list(self._handlers.keys()) or ["ssl_simulator"]

        normalized_level = normalize_level(level)
        for package_name in packages:
            logger = logging.getLogger(package_name)
            logger.setLevel(normalized_level)

    def set_format(self, format_type: str, packages: list[str] | None = None) -> None:
        """Set format for specific packages."""
        if packages is None:
            packages = list(self._handlers.keys()) or ["ssl_simulator"]

        formatter = get_formatter(format_type)
        for package_name in packages:
            if package_name in self._handlers:
                self._handlers[package_name].setFormatter(formatter)

    def suppress_package(self, package_name: str) -> None:
        """Suppress logging from a specific package."""
        logger = logging.getLogger(package_name)
        logger.setLevel(logging.CRITICAL + 1)

    def enable_third_party(
        self, package_names: list[str], level: int | str = logging.WARNING
    ) -> None:
        """
        Optionally enable logging from third-party packages at a high level.

        Examples
        --------
        >>> manager = LoggerManager()
        >>> manager.setup()
        >>> manager.enable_third_party(["PyQt5", "matplotlib"], level="WARNING")
        """
        normalized_level = normalize_level(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(get_formatter("standard"))

        for package_name in package_names:
            logger = logging.getLogger(package_name)
            logger.addHandler(handler)
            logger.setLevel(normalized_level)
