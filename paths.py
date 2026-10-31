"""Where the app keeps things. Code and models sit in this folder; user data (recorded voices,
the library, presets, settings, the API key, the pronunciation dictionary) goes to DATA_DIR,
which defaults to this folder too. The Docker image sets DATA_DIR=/data, a volume.

Only the standard library here: the GPU workers import it from their own envs.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_DIR = Path(os.environ.get("DATA_DIR") or ROOT)
