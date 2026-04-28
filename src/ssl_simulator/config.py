"""
Configuration for ssl_simulator.
"""

import logging
import os

from numpy import pi

_logger = logging.getLogger(__name__)


class Config(dict):
    def __setitem__(self, key, value):
        _logger.info(f"SSL simulator configuration updated: {key} = {value}")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            _logger.info(f"SSL simulator configuration updated: {key} = {value}")
        super().update(*args, **kwargs)


# Initialize the configuration dictionary
CONFIG = Config(
    {
        "SO3_STEP": os.getenv("SSL_SIMULATOR_SO3_STEP", pi / 12),
        "LOG_INLINE_MAX_LEN": int(os.getenv("SSL_LOG_INLINE_MAX_LEN", "100")),
        "LOG_INLINE_MAX_KEYS": int(os.getenv("SSL_LOG_INLINE_MAX_KEYS", "6")),
        "LOG_FANCY_MAX_ROWS": int(os.getenv("SSL_LOG_FANCY_MAX_ROWS", "8")),
        "LOG_FANCY_MAX_COLS": int(os.getenv("SSL_LOG_FANCY_MAX_COLS", "8")),
        "LOG_FANCY_PRECISION": int(os.getenv("SSL_LOG_FANCY_PRECISION", "4")),
    }
)
