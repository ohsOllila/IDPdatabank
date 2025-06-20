"""
General utilities for the IDP Databank project.

This module provides shared functionality including logging configuration,
file operations, and other common utilities used across the codebase.
"""

import logging
import colorlog
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml


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


class FileHandler:
    """
    Handles file operations including reading and writing YAML files.
    """

    @staticmethod
    def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Read and parse a YAML file.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed YAML content as a dictionary

        Raises:
            RuntimeError: If there's an error reading or parsing the file
        """
        try:
            with open(file_path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise RuntimeError(f"Failed to read YAML file {file_path}: {str(e)}")

    @staticmethod
    def write_yaml(file_path: Union[str, Path], data: Dict[str, Any]) -> None:
        """
        Write a dictionary to a YAML file.

        Args:
            file_path: Path to the YAML file
            data: Dictionary to write to the file

        Raises:
            RuntimeError: If there's an error writing the file
        """
        try:
            with open(file_path, "w") as file:
                yaml.safe_dump(data, file)
        except Exception as e:
            raise RuntimeError(f"Failed to write YAML data to {file_path}: {str(e)}")

    @staticmethod
    def join_paths(*args: str) -> str:
        """
        Join path components using the appropriate path separator.

        Args:
            *args: Path components to join

        Returns:
            str: Joined path
        """
        return str(Path(*args))


# Set up default logging when module is imported
setup_colored_warnings()

# Export commonly used functions and classes
__all__ = ["setup_colored_logging", "setup_colored_warnings", "FileHandler"]
