"""
Configuration for ssl_simulator.
"""

import logging
import os

from numpy import pi

logger = logging.getLogger(__name__)


class Config(dict):
    def __setitem__(self, key, value):
        logger.info(f"SSL simulator configuration updated: {key} = {value}")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            logger.info(f"SSL simulator configuration updated: {key} = {value}")
        super().update(*args, **kwargs)


# Initialize the configuration dictionary
CONFIG = Config(
    {
        "DEBUG": os.getenv("SSL_SIMULATOR_DEBUG", "False").lower() in ("true", "1", "yes"),
        "SO3_STEP": os.getenv("SSL_SIMULATOR_SO3_STEP", pi / 12),
    }
)
