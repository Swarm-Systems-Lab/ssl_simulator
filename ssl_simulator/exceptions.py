"""
Custom exceptions for ssl_simulator.

Provides a clear hierarchy of exceptions for different error scenarios.
"""

__all__ = [
    "SSLSimulatorError",
    "InitializationError",
    "ConfigurationError",
    "ControllerError",
    "RobotModelError",
    "ValidationError",
]


class SSLSimulatorError(Exception):
    """Base exception class for all ssl_simulator errors."""
    pass


class InitializationError(SSLSimulatorError):
    """Raised when initialization of simulator, robot model, or controller fails."""
    pass


class ConfigurationError(SSLSimulatorError):
    """Raised when configuration is invalid or missing."""
    pass


class ControllerError(SSLSimulatorError):
    """Raised when there's an error in controller setup or execution."""
    pass


class RobotModelError(SSLSimulatorError):
    """Raised when there's an error in robot model setup or execution."""
    pass


class ValidationError(SSLSimulatorError):
    """Raised when data validation fails."""
    pass
