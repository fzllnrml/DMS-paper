"""Independent continuation of DMS E0–E9. Importing this package never runs jobs."""

import sys

# Read-only imports must not create __pycache__ in the frozen first version/src.
sys.dont_write_bytecode = True

__version__ = "2.0.0"
