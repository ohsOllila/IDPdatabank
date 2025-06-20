"""
Parameter comparison module for matching simulation and experimental parameters.

This module provides a flexible system for comparing numerical parameters with
configurable thresholds and comparison methods.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Any, Optional, Union, TypeVar, Generic

# Type variable for numeric types
Numeric = Union[int, float]


class ThresholdType(Enum):
    """Type of threshold to use for comparison."""

    ABSOLUTE = auto()  # Fixed value difference (e.g., ±0.5)
    PERCENT = auto()  # Percentage of the reference value (e.g., ±10%)


@dataclass
class Threshold:
    """Defines a threshold for parameter comparison."""

    value: float
    threshold_type: ThresholdType
    description: str = ""

    def __post_init__(self):
        if self.threshold_type == ThresholdType.PERCENT and not (
            0 <= self.value <= 100
        ):
            raise ValueError("Percentage threshold must be between 0 and 100")


@dataclass
class ComparisonResult:
    """Result of a parameter comparison operation."""

    match: bool
    difference: float
    within_tolerance: bool
    threshold: Threshold
    tolerance_used: float
    message: str = ""

    def __str__(self) -> str:
        return self.message


class ParameterComparator:
    """
    Handles comparison of simulation and experiment parameters with configurable thresholds.

    Example:
        >>> comparator = ParameterComparator()
        >>> # Configure pH with absolute threshold
        >>> comparator.configure_parameter(
        ...     "ph",
        ...     Threshold(0.5, ThresholdType.ABSOLUTE, "pH difference tolerance")
        ... )
        >>> # Configure temperature with percentage threshold
        >>> comparator.configure_parameter(
        ...     "temperature",
        ...     Threshold(10.0, ThresholdType.PERCENT, "±10% temperature difference")
        ... )
        >>> # Compare values
        >>> result = comparator.compare("ph", 6.8, 7.0)
        >>> print(result.message)
    """

    def __init__(self):
        """Initialize with default parameter configurations."""
        self._thresholds: Dict[str, Threshold] = self._get_default_thresholds()

    @staticmethod
    def _get_default_thresholds() -> Dict[str, Threshold]:
        """Get default threshold configurations for common parameters."""
        return {
            "ph": Threshold(
                value=0.5,
                threshold_type=ThresholdType.ABSOLUTE,
                description="Maximum allowed pH difference",
            ),
            "temperature": Threshold(
                value=10.0,
                threshold_type=ThresholdType.PERCENT,
                description="Maximum allowed temperature difference (±10%)",
            ),
            "ionic_strength": Threshold(
                value=15.0,
                threshold_type=ThresholdType.PERCENT,
                description="Maximum allowed ionic strength difference (±15% or ±0.01M, whichever is larger)",
            ),
        }

    def configure_parameter(self, param_name: str, threshold: Threshold) -> None:
        """
        Configure comparison settings for a parameter.

        Args:
            param_name: Name of the parameter (e.g., 'ph', 'temperature')
            threshold: Threshold configuration for the parameter
        """
        self._thresholds[param_name] = threshold

    def get_parameter_config(self, param_name: str) -> Optional[Threshold]:
        """
        Get the threshold configuration for a parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Threshold configuration or None if not found
        """
        return self._thresholds.get(param_name)

    def compare(
        self,
        param_name: str,
        sim_value: Numeric,
        exp_value: Numeric,
        custom_threshold: Optional[Threshold] = None,
    ) -> ComparisonResult:
        """
        Compare simulation and experimental values for a parameter.

        Args:
            param_name: Name of the parameter being compared
            sim_value: Simulated value
            exp_value: Experimental value
            custom_threshold: Optional override for the parameter's threshold

        Returns:
            ComparisonResult with detailed comparison information

        Raises:
            ValueError: If the parameter is not configured
        """
        if param_name not in self._thresholds and custom_threshold is None:
            raise ValueError(f"No threshold configured for parameter: {param_name}")

        threshold = custom_threshold or self._thresholds[param_name]
        diff = abs(float(sim_value) - float(exp_value))

        if threshold.threshold_type == ThresholdType.ABSOLUTE:
            tolerance = threshold.value
        else:  # PERCENT
            ref_value = abs(exp_value)
            tolerance = (threshold.value / 100.0) * ref_value

        # Special case for ionic strength: minimum absolute difference
        if param_name == "ionic_strength":
            min_abs_diff = 0.01  # 10mM
            tolerance = max(tolerance, min_abs_diff)

        within_tolerance = diff <= tolerance
        match = within_tolerance

        # Generate descriptive message
        if match:
            message = (
                f"Match: {sim_value} ≈ {exp_value} "
                f"(diff: {diff:.3f} ≤ {tolerance:.3f} {self._get_tolerance_units(threshold)})"
            )
        else:
            message = (
                f"Mismatch: {sim_value} ≠ {exp_value} "
                f"(diff: {diff:.3f} > {tolerance:.3f} {self._get_tolerance_units(threshold)})"
            )

        return ComparisonResult(
            match=match,
            difference=diff,
            within_tolerance=within_tolerance,
            threshold=threshold,
            tolerance_used=tolerance,
            message=message,
        )

    @staticmethod
    def _get_tolerance_units(threshold: Threshold) -> str:
        """Get the units string for a threshold."""
        if threshold.threshold_type == ThresholdType.ABSOLUTE:
            return "units"
        return "%"
