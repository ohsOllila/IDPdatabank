from typing import Dict, Any, Optional
from pathlib import Path
from .buffer_manager import BufferManager
from .file_handler import FileHandler


class Simulation:
    """
    Handles simulation data processing using composition for buffer and file operations.
    """
    
    def __init__(self, info_path: str) -> None:
        """
        Initialize the Simulation with path to info file.
        
        Args:
            info_path: Path to the simulation info YAML file
        """
        self.info_path = Path(info_path)
        self.file_handler = FileHandler()
        self.buffer_manager: Optional[BufferManager] = None
        self.info: Dict[str, Any] = {}
        self.composition: Dict[str, Any] = {}
        self.sequence: str = ""
        
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize simulation data and buffer."""
        try:
            # Load simulation info
            self.info = self.file_handler.read_yaml(self.info_path)
            self.composition = self.info.get("COMPOSITION", {})
            
            # Extract sequence
            protein_data = self.composition.get("PROTEIN", {})
            self.sequence = protein_data.get("SEQUENCE", "")
            
            # Initialize buffer if buffer data is available
            self._initialize_buffer()
            
        except Exception as e:
            raise RuntimeError(f"Error initializing simulation: {str(e)}")
    
    def _initialize_buffer(self) -> None:
        """Initialize buffer manager with simulation data if available."""
        # Check if buffer data exists in the simulation info
        buffer_data = self.info.get("BUFFER", {})
        ph = buffer_data.get("ph", 7.0)  # Default to neutral pH if not specified
        
        self.buffer_manager = BufferManager(ph=ph)
        
        # Add buffer components if they exist
        components = buffer_data.get("components", [])
        for comp in components:
            name = comp.get("name")
            concentration = comp.get("concentration")
            unit = comp.get("unit", "M")  # Default to Molar if not specified
            
            if name is not None and concentration is not None:
                self.buffer_manager.add_component(name, float(concentration), unit)
    
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
