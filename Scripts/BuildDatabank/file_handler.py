from typing import Any, Dict, Optional
import yaml
import os


class FileHandler:
    """
    Handles file operations including reading and writing YAML files.
    """

    @staticmethod
    def read_yaml(file_path: str) -> Dict[str, Any]:
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
            raise RuntimeError(f"Failed to read YAML data from {file_path}: {str(e)}")

    @staticmethod
    def file_exists(file_path: str) -> bool:
        """Check if a file exists."""
        return os.path.exists(file_path)

    @staticmethod
    def join_path(*args: str) -> str:
        """Join path components."""
        return os.path.join(*args)

    @staticmethod
    def write_yaml(file_path: str, data: Dict[str, Any]) -> None:
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
