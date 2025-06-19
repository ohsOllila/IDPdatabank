from typing import Dict, Any, Optional
from pathlib import Path
from buffer_manager import BufferManager
from file_handler import FileHandler


class Experiment:
    """
    Handles experiment data processing using composition for buffer and file operations.
    """

    def __init__(self, experiment_type: str, path: str) -> None:
        """
        Initialize the Experiment with type and path.

        Args:
            experiment_type: Type of experiment (e.g., 'spin_relaxation')
            path: Path to the experiment directory
        """
        self.experiment_type = experiment_type
        self.path = Path(path)
        self.file_handler = FileHandler()
        self.buffer_manager: Optional[BufferManager] = None
        self.sequence: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}
        self.ionic_strength: Optional[float] = None
        self.temperature: Optional[float] = None

        self._initialize()

    def _initialize(self) -> None:
        """Initialize experiment data and buffer."""
        if self.experiment_type == "spin_relaxation":
            self._load_spin_relaxation_data()

        if self.metadata:
            self._initialize_buffer()

            simulation_ions = self.buffer_manager.get_all_components()
            simulation_ionic_strength = 0
            for key in simulation_ions.keys():
                simulation_ionic_strength += (
                    self.buffer_manager.calculate_ionic_strength(key)
                )
            self.ionic_strength = simulation_ionic_strength
    def _load_spin_relaxation_data(self) -> None:
        """Load data specific to spin relaxation experiments."""
        t1_metadata_file = self.path / "T1_metadata.yaml"
        sequence_file = self.path / "fasta.yaml"

        # Load sequence data
        if self.file_handler.file_exists(sequence_file):
            self.sequence = self.file_handler.read_yaml(sequence_file)["sequence"][0]

        # Load T1 metadata
        if self.file_handler.file_exists(t1_metadata_file):
            self.metadata = self.file_handler.read_yaml(t1_metadata_file)

    def _initialize_buffer(self) -> None:
        """Initialize buffer manager with experiment data."""
        if not self.metadata:
            return

        try:
            ph = float(self.metadata["Sample_condition_variable"]["ph"])
            self.buffer_manager = BufferManager(ph=ph)

            # Add buffer components
            for component in self.metadata.get("Sample_component", []):
                mol_name = component["Mol_common_name"]
                conc_val = float(component["Concentration_val"])
                conc_unit = component["Concentration_val_units"]

                if self.buffer_manager:
                    self.buffer_manager.add_component(mol_name, conc_val, conc_unit)

        except (KeyError, ValueError) as e:
            raise ValueError(f"Error initializing buffer: {str(e)}")

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
