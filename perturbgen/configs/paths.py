from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
# This assumes the structure is like:
# T_perturb/
# ├── data/
# ├── perturbgen/
# │   ├── configs/
# │   ├── res/
# │   └── tokenized_data/
# Define the project name based on the root directory
PROJECT_DIR = ROOT / "T_perturb"

# Paths for data
DATA_DIR = ROOT / "data"
RESULTS_DIR = PROJECT_DIR / "res"
TOKENIZED_DIR = PROJECT_DIR / "tokenized_data"

# make directories if they do not exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)