# conftest.py
import os

# these will be set before any tests or fixtures run
os.environ.setdefault("FEELPP_NUM_PROCS", "1")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MPLBACKEND", "Agg")