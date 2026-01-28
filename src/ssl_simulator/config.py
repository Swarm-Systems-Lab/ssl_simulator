# ssl_simulator/config.py

import os

from numpy import pi


class Config(dict):
    def __setitem__(self, key, value):
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        for _key, _value in dict(*args, **kwargs).items():
            pass
        super().update(*args, **kwargs)


# Initialize the configuration dictionary
CONFIG = Config(
    {
        "DEBUG": os.getenv("SSL_SIMULATOR_DEBUG", "False").lower() in ("true", "1", "yes"),
        "SO3_STEP": os.getenv("SSL_SIMULATOR_SO3_STEP", pi / 12),
    }
)
