# BuildDatabank Module

## Overview

The BuildDatabank module is a Python package designed to manage, process, and analyze experimental and simulation data for Intrinsically Disordered Proteins (IDPs). It provides tools for handling buffer compositions, comparing experimental and simulation conditions, and matching protein sequences. The module (tries to)adheres to the "Composition over Inheritance" principle, favoring flexible component composition over deep inheritance hierarchies.

## Key Features

- **Buffer Management**: Handle complex buffer compositions with pH-dependent properties
- **Sequence Analysis**: Align and compare protein sequences with configurable scoring
- **Parameter Matching**: Flexible comparison of experimental and simulation parameters
- **Extensible Design**: Modular architecture for easy extension
- **Type Hints**: Full Python type hints for better IDE support and code clarity

## Features

### Core Components

1. **Buffer Management**
   - `BufferManager`: Handles buffer composition and calculations
   - `SimulationBufferManager`: Specialized buffer manager for simulation data
   - `buffer_molecule_data.py`: Centralized storage for buffer component properties

2. **Data Handling**
   - `FileHandler`: Manages file I/O operations, particularly for YAML files
   - `Experiment`: Represents experimental data and metadata
   - `Simulation`: Represents simulation data and metadata

3. **Analysis Tools**
   - `ParameterComparator`: Compares experimental and simulation parameters
   - `build_databank_utils.py`: Utility functions for sequence alignment and analysis

4. **Search Functionality**
   - `searchDatabank.py`: Main script for searching and matching experimental data with simulations
   - `searchDatabank_we.py`: Alternative implementation with different matching criteria

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd idpdatabank_bkav
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Workflow

1. **Initialize Experiment and Simulation Objects**
   ```python
   from experiment import Experiment
   from simulation import Simulation
   
   # Create experiment instance
   exp = Experiment(experiment_type="spin_relaxation", path="/path/to/experiment")
   
   # Create simulation instance
   sim = Simulation(path="/path/to/simulation")
   ```

2. **Working with Buffer Data**
   ```python
   # Get buffer component data
   buffer_data = exp.buffer_manager.get_all_components()
   
   # Calculate ionic strength
   ionic_strength = exp.buffer_manager.calculate_ionic_strength("sodium_chloride")
   ```

3. **Search and Match**
   The main search functionality is available through `searchDatabank.py`:
   ```bash
   python searchDatabank.py
   ```

## Project Structure

```
Scripts/
├── databank_general_utils.py  # Shared utilities (logging, file I/O, etc.)
└── BuildDatabank/
    ├── __init__.py           # Package initialization
    ├── buffer_manager.py      # Buffer management and calculations
    ├── buffer_molecule_data.py # Buffer component definitions
    ├── build_databank_utils.py # Build-specific utility functions
    ├── experiment.py          # Experiment class implementation
    ├── models.py              # Data models and Pydantic schemas
    ├── parameter_comparator.py # Parameter comparison logic
    ├── searchDatabank.py      # Main search functionality
    ├── searchDatabank_we.py   # Alternative search implementation
    └── simulation.py          # Simulation class implementation
```

### databank_general_utils.py

This module contains shared utilities used across the project:

- **Logging Configuration**:
  - `setup_colored_logging()`: Configures colored console output
  - `setup_colored_warnings()`: Sets up colored warning messages

- **File Operations**:
  - `FileHandler` class: Handles YAML file I/O and path operations

Example usage:
```python
from databank_general_utils import setup_colored_logging, FileHandler

# Set up logging
logger = setup_colored_logging(__name__)
logger.info("This is an info message")

# Use file handler
file_handler = FileHandler()
data = file_handler.read_yaml("config.yaml")
```

## Key Features in Detail

### Buffer Management

The `BufferManager` class provides comprehensive handling of buffer compositions, including:
- pH-dependent calculations
- Ionic strength calculations
- Unit conversions
- Component management

### Sequence Alignment

The module includes functionality for aligning and comparing protein sequences using BioPython's alignment tools, with support for custom scoring matrices.

### Parameter Comparison

The `ParameterComparator` class allows for flexible comparison of experimental and simulation parameters with configurable thresholds and matching criteria.

## Shared Utilities

The `databank_general_utils.py` module provides common functionality used throughout the codebase:

### Logging

- Use `setup_colored_logging(__name__)` to get a configured logger
- Logging levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Warnings are automatically captured and colored

### File Operations

The `FileHandler` class provides:
- Safe YAML file reading/writing
- Cross-platform path handling
- Consistent error handling

## Configuration

### Buffer Components

Buffer components are defined in `buffer_molecule_data.py`. Each component can have:

- `ph_dependent`: Boolean indicating if properties vary with pH
- `charges`: List of charges for each species
- `stoichiometry`: Stoichiometric coefficients
- `ph_ranges`: For pH-dependent components, defines properties at different pH ranges

Example component definition:
```python
"sodium_phosphate": {
    "ph_dependent": True,
    "ph_ranges": [
        {"min_ph": 0, "max_ph": 2, "charges": [0], "stoichiometry": [1]},
        # Additional pH ranges...
    ]
}
```

### Parameter Comparison

Thresholds for parameter matching can be configured in `parameter_comparator.py`:

- Absolute vs. relative comparison thresholds
- Custom comparison functions
- Weighted scoring for different parameters

## API Reference

### Core Classes

#### BufferManager

```python
class BufferManager:
    def __init__(self, ph: float = 0.0):
        """Initialize with pH value."""
        
    def add_component(self, name: str, concentration: float, unit: str) -> None:
        """Add a buffer component."""
        
    def calculate_ionic_strength(self, component: str) -> float:
        """Calculate ionic strength for a component."""
```

#### Experiment

```python
class Experiment:
    def __init__(self, experiment_type: str, path: str):
        """Initialize with experiment type and path."""
        
    def get_sequence(self) -> str:
        """Get protein sequence."""
        
    def get_buffer_components(self) -> Dict[str, Dict[str, float]]:
        """Get all buffer components."""
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   - Ensure all dependencies are installed from `requirements.txt`
   - Check for version conflicts

2. **Buffer Component Not Found**
   - Verify the component is defined in `buffer_molecule_data.py`
   - Check for typos in component names

3. **Sequence Alignment Failures**
   - Ensure sequences are in single-letter amino acid code
   - Check for invalid characters in sequences

### Debugging

Enable debug logging for more detailed output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Write docstrings for all public methods and classes
- Include type hints for better code clarity
- Add tests for new features
- Update documentation when making API changes

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or support, please contact:
- Email: [your-email@example.com](mailto:your-email@example.com)
- Issue Tracker: [GitHub Issues](https://github.com/yourusername/idpdatabank_bkav/issues)

## Acknowledgments

- Built with ❤️ by [Your Name/Team]
- Thanks to all contributors who have helped improve this project
- Inspired by [relevant projects or papers]
