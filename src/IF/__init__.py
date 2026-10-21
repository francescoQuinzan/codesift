from pathlib import Path

# Repository root, three levels up from src/IF/__init__.py.
# Valid when the package is installed in editable mode (pip install -e .).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SEMGREP_INSTRUCTIONS_PATH = PROJECT_ROOT / "semgrep_instructions"
RESULTS_PATH = PROJECT_ROOT / "results"

# Kept for code that still uses the old private name.
_PROJECT_ROOT = PROJECT_ROOT
