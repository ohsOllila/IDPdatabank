from typing import Dict, Any, Optional
from pathlib import Path
from buffer_manager import SimulationBufferManager
from file_handler import FileHandler


class Simulation:
    """
    Handles simulation data processing using composition for buffer and file operations.
    """

    def __init__(self, path: str) -> None:
        """
        Initialize the Simulation with path to info file.

        Args:
            path: Path to the simulation info YAML file
        """
        self.path = Path(path)
        self.file_handler = FileHandler()
        self.buffer_manager: Optional[SimulationBufferManager] = None
        self.info: Dict[str, Any] = {}
        self.composition: Dict[str, Any] = {}
        self.sequence: str = ""
        self.ionic_strength: Optional[float] = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize simulation data and buffer."""
        try:
            # Load simulation info
            self.info = self.file_handler.read_yaml(self.path)
            self.composition = self.info.get("COMPOSITION", {})

            # Extract sequence
            protein_data = self.composition.get("PROTEIN", {})
            self.sequence = protein_data.get("SEQUENCE", "")

            # Initialize buffer if buffer data is available
            self._initialize_buffer()

            # calculate the ionic strength
            simulation_ions = self.buffer_manager.get_all_components()
            simulation_ionic_strength = 0
            for key, val in simulation_ions.items():
                simulation_ionic_strength += (
                    self.buffer_manager.calculate_ionic_strength(
                        key, val["concentration_val"]
                    )
                )
            self.ionic_strength = simulation_ionic_strength

        except Exception as e:
            raise RuntimeError(f"Error initializing simulation: {e}")

    def _initialize_buffer(self) -> None:
        """Initialize buffer manager with simulation data if available."""
        count_water = self.composition.get("SOL", {}).get("COUNT", 0)
        if count_water == 0:
            raise ValueError(
                "No water found in simulation composition. I don't know what to do now."
            )
        buffer_data = {
            k: v for k, v in self.composition.items() if k not in ["PROTEIN", "SOL"]
        }  #
        ph = buffer_data.get("ph", 7.0)  # Default to neutral pH if not specified

        self.buffer_manager = SimulationBufferManager(ph=ph)

        # Add buffer components if they exist
        components = buffer_data.keys()
        for comp in components:
            name = comp
            count = buffer_data[comp].get("COUNT")
            concentration = self.buffer_manager._get_concentration_in_molar_from_count(
                count, count_water
            )

            if name is not None and concentration is not None:
                self.buffer_manager.add_component(name, float(concentration), "M")

    def calculate_ionic_strength(self, component: str) -> float:
        """
        Calculate ionic strength for a buffer component.

        Args:
            component: Name of the buffer component

        Returns:
            float: Ionic strength value

        Raises:
            RuntimeError: If buffer manager is not initialized
        """
        if not self.buffer_manager:
            raise RuntimeError("Buffer manager not initialized")

        return self.buffer_manager.calculate_ionic_strength(component)

    def get_buffer_ph(self) -> float:
        """Get the pH of the buffer."""
        if not self.buffer_manager:
            raise RuntimeError("Buffer manager not initialized")
        return self.buffer_manager.ph

    def get_sequence(self) -> str:
        """Get the protein sequence."""
        return self.sequence
