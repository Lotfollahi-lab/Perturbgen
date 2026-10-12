from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

# The Perturbgen repository itself (ROOT is the directory that contains it).
REPO_DIR = Path(__file__).resolve().parents[2]
# Files shipped with the repository, used as defaults so no private paths are needed.
PP_DIR = REPO_DIR / "perturbgen" / "pp"
GENE_MEDIAN_PATH = PP_DIR / "gene_median_dict_gftokens_gc95M.pkl"
TOKEN_DICT_PATH = PP_DIR / "token_dict_gftokens_gc95M.pkl"
GENE_MAPPING_PATH = PP_DIR / "ensembl_mapping_dict_gc95M.pkl"
ENCODER_CKPT_PATH = (
    REPO_DIR / "pretraining_cohort"
    / "20250709_1223_cellgen_train_masking_lr_5e-05_wd_1e-06_batch_64"
      "_ptime_pos_sin_m_pow_tp_1-2-3_s_42-epoch=00.ckpt"
)
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