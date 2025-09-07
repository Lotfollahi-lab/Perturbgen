import os
BASE_DIR = os.path.dirname(__file__)

# Defaults: your custom copies living inside pp/
GENE_MEDIAN_FILE = os.path.join(BASE_DIR, "gene_median_dict_gftokens_gc95M.pkl")
TOKEN_DICTIONARY_FILE = os.path.join(BASE_DIR, "token_dict_gftokens_gc95M.pkl")
ENSEMBL_MAPPING_FILE  = os.path.join(BASE_DIR, "ensembl_mapping_dict_gc95M.pkl")