"""Repository-relative defaults; importing this module never creates directories."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def _root(variable, fallback):
    return Path(os.environ.get(variable, str(fallback))).expanduser().resolve()

DATA_ROOT = _root('POI_DATA_ROOT', REPO_ROOT / 'data')
RUNS_ROOT = _root('POI_RUNS_ROOT', REPO_ROOT / 'runs')
RESULTS_ROOT = _root('POI_RESULTS_ROOT', REPO_ROOT / 'results')

def run_path(relative):
    return RUNS_ROOT / relative
