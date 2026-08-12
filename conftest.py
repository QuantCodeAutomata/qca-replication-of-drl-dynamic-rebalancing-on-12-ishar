"""Project-level pytest configuration."""

import sys
import os

# Ensure the project root is on the path so `src.*` imports resolve in tests.
sys.path.insert(0, os.path.dirname(__file__))
