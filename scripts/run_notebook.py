"""Execute the Ki modelling notebook from the repository root."""

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "Ki_modelling.ipynb"

if not NOTEBOOK.exists():
    raise FileNotFoundError(
        "notebooks/Ki_modelling.ipynb is not present. "
        "Add the exam notebook before running this command."
    )

cmd = [
    sys.executable,
    "-m",
    "jupyter",
    "nbconvert",
    "--to",
    "notebook",
    "--execute",
    "--inplace",
    str(NOTEBOOK),
]

subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
