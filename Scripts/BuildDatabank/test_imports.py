#!/usr/bin/env python3
"""Test script to verify imports and basic functionality."""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    # Try relative imports first
    try:
        from Scripts.BuildDatabank.experiment import Experiment
        from Scripts.BuildDatabank.simulation import Simulation
        from Scripts.BuildDatabank.buffer_manager import BufferManager
        from Scripts.BuildDatabank.file_handler import FileHandler
    except ImportError:
        # Fall back to direct imports if running from the BuildDatabank directory
        from experiment import Experiment
        from simulation import Simulation
        from buffer_manager import BufferManager
        from file_handler import FileHandler
    
    print("✅ All imports successful!")
    
    # Test basic functionality
    try:
        # This is just a basic test - you might need to adjust paths
        exp = Experiment("spin_relaxation", "Data/Experiments/spin_relaxation/BMRBid19993")
        print(f"✅ Experiment created successfully")
        
        # Try to calculate ionic strength if the component exists
        try:
            ionic_strength = exp.calculate_ionic_strength("sodium chloride")
            print(f"✅ Ionic strength calculation successful: {ionic_strength}M")
        except Exception as e:
            print(f"⚠️  Ionic strength calculation failed (this might be expected): {e}")
            print("This might be due to missing data files or incorrect paths.")
            
    except Exception as e:
        print(f"❌ Error creating experiment: {e}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you're running this from the correct directory and all files are in place.")
    print("The directory structure should be:")
    print("Scripts/")
    print("└── BuildDatabank/")
    print("    ├── __init__.py")
    print("    ├── buffer_manager.py")
    print("    ├── experiment.py")
    print("    ├── file_handler.py")
    print("    └── simulation.py")

if __name__ == "__main__":
    pass
