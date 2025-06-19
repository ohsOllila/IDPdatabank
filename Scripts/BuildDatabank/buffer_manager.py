from typing import Dict, List, Optional, Union, Any
from buffer_molecule_data import BUFFER_MOLECULE_DATA


class BufferManager:
    """
    Handles all buffer-related calculations and data management.
    Uses composition to provide buffer functionality to other classes.
    """

    def __init__(self, ph: float = 0.0):
        """
        Initialize BufferManager with a specific pH value.

        Args:
            ph: The pH value for buffer calculations
        """
        self.ph = ph
        self._buffer_data: Dict[str, Dict[str, Union[float, str]]] = {}

    def add_component(self, name: str, concentration: float, unit: str) -> None:
        """Add a component to the buffer."""
        self._buffer_data[name] = {
            "concentration_val": concentration,
            "concentration_val_units": unit,
        }

    def get_component(self, name: str) -> Dict[str, Union[float, str]]:
        """Get buffer component data by name."""
        return self._buffer_data.get(name, {})

    def get_all_components(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get all buffer components."""
        return self._buffer_data

    def calculate_ionic_strength(self, component: str) -> float:
        """
        Calculate the ionic strength for a specific buffer component.

        Args:
            component: Name of the buffer component

        Returns:
            float: The calculated ionic strength
        """
        if component not in self._buffer_data:
            raise ValueError(f"Component {component} not found in buffer data")

        if component in ("H2O", "D2O", "entity_1"):
            return 0.0

        charges = self._get_charges_at_ph(component, self.ph)
        concentration = self._get_concentration_in_molar(component)

        # Calculate ionic strength: I = 0.5 * Σ(ci * zi²)
        ionic_strength = 0.0
        for i, j in zip(stoichiometry, charges):
            ionic_strength += 0.5 * concentration * i * j**2
        return ionic_strength

    def _get_charges_at_ph(self, molecule_name: str, ph_value: float) -> List[int]:
        """Get charges for a molecule at a specific pH."""
        molecule_data = self._get_buffer_molecule_data(molecule_name)

        if not molecule_data["ph_dependent"]:
            return molecule_data["charges"]

        ranges = molecule_data["ph_ranges"]
        # Binary search for the correct pH range
        left, right = 0, len(ranges) - 1

        while left <= right:
            mid = (left + right) // 2
            range_data = ranges[mid]
            if range_data["min_ph"] <= ph_value <= range_data["max_ph"]:
                return range_data["charges"]
            elif ph_value < range_data["min_ph"]:
                right = mid - 1
            else:
                left = mid + 1

        raise ValueError(f"No pH range found for {molecule_name} at pH {ph_value}")

    def _get_stoichiometry_at_ph(
        self, molecule_name: str, ph_value: float
    ) -> List[int]:
        """Get stoichiometry for a molecule at a specific pH."""
        molecule_data = self._get_buffer_molecule_data(molecule_name)

        if not molecule_data["ph_dependent"]:
            return molecule_data["stoichiometry"]

        ranges = molecule_data["ph_ranges"]
        # Binary search for the correct pH range
        left, right = 0, len(ranges) - 1

        while left <= right:
            mid = (left + right) // 2
            range_data = ranges[mid]
            if range_data["min_ph"] <= ph_value <= range_data["max_ph"]:
                return range_data["stoichiometry"]
            elif ph_value < range_data["min_ph"]:
                right = mid - 1
            else:
                left = mid + 1

        raise ValueError(f"No pH range found for {molecule_name} at pH {ph_value}")

    def _get_conversion_factor(self, unit: str) -> float:
        """Convert concentration units to Molar."""
        unit = unit.lower()
        conversion_factors = {
            "m": 1e-3,  # milli (mM -> M)
            "μ": 1e-6,  # micro (μM -> M)
            "u": 1e-6,  # micro (uM -> M)
            "n": 1e-9,  # nano (nM -> M)
            "": 1.0,  # already in M
            "molar": 1.0,
            "millimolar": 1e-3,
            "micromolar": 1e-6,
            "nanomolar": 1e-9,
        }

        # Handle cases like 'mM', 'uM', 'nM'
        if unit.endswith("m"):
            return 1e-3
        if unit.endswith(("μ", "u")):
            return 1e-6
        if unit.endswith("n"):
            return 1e-9

        # Handle full unit names
        return conversion_factors.get(unit.lower(), 1.0)

    def _get_concentration_in_molar(self, component: str) -> float:
        """Get concentration in Molar units."""
        comp_data = self._buffer_data[component]
        return comp_data["concentration_val"] * self._get_conversion_factor(
            comp_data["concentration_val_units"]
        )

    @staticmethod
    def _get_buffer_molecule_data(component: str) -> Dict[str, Any]:
        """
        Get buffer molecule data from the BUFFER_MOLECULE_DATA dictionary.

        Args:
            component: Name of the buffer component to look up

        Returns:
            Dict containing the component's data

        Raises:
            ValueError: If the component is not found in BUFFER_MOLECULE_DATA
        """
        if component not in BUFFER_MOLECULE_DATA:
            raise ValueError(f"Unknown component: {component}")

        return BUFFER_MOLECULE_DATA[component]


class SimulationBufferManager(BufferManager):
    """
    Specialized BufferManager for simulation purposes.
    """

    def __init__(self, ph: float = 0.0):
        super().__init__(ph)
        # Add any simulation-specific initialization here

    def _get_concentration_in_molar_from_count(
        self, count_component: int, count_water: int
    ) -> float:
        """Calculate the concentration of a component in the simulation."""
        # Water concentration is 55.5 M
        # we get molar concentration in M.
        water_concentration = 55.5
        return count_component / count_water * water_concentration

    # Override any methods you need to change
    def calculate_ionic_strength(
        self, component: str, concentration_component: float
    ) -> float:
        """Override the base implementation for simulation-specific logic."""
        # Your simulation-specific implementation here
        charge = self._get_charges_at_ph(component, self.ph)
        return 0.5 * charge[0] ** 2 * concentration_component
