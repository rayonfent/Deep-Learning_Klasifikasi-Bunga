"""
Root level app.py to support Streamlit deployment.
This file serves as the entry point for Streamlit Cloud deployment.
It imports and runs the main application from the flower_app module.
"""

import sys
import os
from pathlib import Path

# Get the directory containing this file
current_dir = Path(__file__).parent
flower_app_dir = current_dir / "flower_app"

# Add flower_app to the path so imports work correctly
sys.path.insert(0, str(flower_app_dir))

# Set environment variable to point to model files
os.environ['APP_DIR'] = str(flower_app_dir)

# Import everything from the main app
if __name__ != "__main__":
    from flower_app.app import *  # noqa: F401, F403