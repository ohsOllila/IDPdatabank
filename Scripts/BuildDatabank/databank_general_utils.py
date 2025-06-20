"""
General utilities for the IDP Databank project.

This module provides shared functionality including logging configuration and other common utilities used across the codebase.
"""

import logging
import colorlog


def setup_colored_logging(
    logger_name: str = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return a logger with colored output.

    Args:
        logger_name: Name of the logger. If None, returns the root logger.
        level: Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Return early if logger already has handlers to avoid duplicate logs
    if logger.handlers:
        return logger

    # Create a colored formatter
    formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={"message": {"ERROR": "red", "CRITICAL": "red"}},
    )

    # Create a stream handler (for console output)
    handler = colorlog.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def setup_colored_warnings():
    """
    Configure the logging system to capture and display warnings with color.

    This should be called early in the application startup.
    """
    # Configure the root logger with colored output
    setup_colored_logging()

    # Capture warnings in the logging system
    logging.captureWarnings(True)

    # Configure the warnings logger to use our handler
    py_warnings_logger = logging.getLogger("py.warnings")
    if not py_warnings_logger.handlers:  # Prevent adding duplicates
        handler = colorlog.StreamHandler()
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s %(message)s",
            log_colors={
                "WARNING": "yellow",
                "ERROR": "red",
            },
        )
        handler.setFormatter(formatter)
        py_warnings_logger.addHandler(handler)

    py_warnings_logger.propagate = False  # Prevent double-logging
