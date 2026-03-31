import argparse
import os
import pickle
from typing import Dict
import sys
import numpy as np
import scanpy as sc
import tqdm
import pickle
from pathlib import Path
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from scipy.sparse import csr_matrix, issparse
from scmaskgit.src.utils import (  # tokenid_mapping,;
    map_ensembl_to_genename,
    map_input_ids_to_row_id,
    pairing_src_to_tgt_cells,
    str2bool,
    subset_adata,
    tokenid_mapping,
)

seed_no = 42
np.random.seed(seed_no)

#     # set working directory to root of repository


def get_args():


    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--h5ad_path',
        type=str,
        default='./data/20241026_HSPC/cd34.h5ad',
        help='Path to h5ad file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='foundation_pretrained',
        choices=['cytoimmgen', 'eb', 'hspc', 'mnc'],
    )
    parser.add_argument(
        '--gene_filtering_mode',
        type=str,
        default='hvg',
        choices=['hvg', 'degs', 'all'],
        help='Gene filtering mode: hvg or degs',
    )
    parser.add_argument(
        '--var_list',
        type=str,
        nargs='+',
        default=[
            'assignment_id',
            'sex',
            'tissue',
            'phase',
            'celltype_v2',
            'donor_tissue',
            'diff_state',
            'dataset',
        ],
        help='List of variables to keep in the dataset',
    )
    parser.add_argument(
        '--pairing_mode',
        type=str,
        default='mapping',
        choices=['stratified', 'random', 'mapping'],
        help='Cell pairing mode',
    )
    parser.add_argument(
        '--pairing_obs',
        type=str,
        default='diff_state',
        help='Observation to use for cell pairing'
        'and encoding the different states (e.g. time, hierarchy).',
    )
    parser.add_argument(
        '--nproc',
        type=int,
        default=8,
        help='Number of processes to use for tokenisation',
    )
    parser.add_argument(
        '--reference_time',
        type=str,
        default='stem',
        help='Control time point for cell pairing' 'which is feed into Geneformer',
    )
    parser.add_argument(
        '--time_point_order',
        type=str,
        nargs='+',
        default=[
            'stem',
            'intermediate',
            'terminal',
        ],
        help='Order of time points in the dataset',
    )
    parser.add_argument(
        '--exclude_non_GF_genes',
        type=str2bool,
        default=True,
        help='Exclude genes in anndata that are not in Geneformer dictionary',
    )
    parser.add_argument(
        '--src_mode',
        type=str,
        default='Geneformer',
        choices=['Geneformer', 'Transformer'],
        help='Mode for tokenisation',
    )
    parser.add_argument(
        '--n_hvg',
        type=int,
        default=5000,
        help='Number of highly variable genes to keep',
    )

    args = parser.parse_args()
    return args

data_directory=Path("/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/cohort/Processed/TRACE_Approach/")
ensemble_mapping= "/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_filtered_tokenid.pkl"
with open(ensemble_mapping, "rb") as f:
    predefined_dict = pickle.load(f)
save_path = "/lustre/scratch126/cellgen/team361/av13/scmaskgit/data/"
for file_path in data_directory.glob(f"*.h5ad"):
    adata = sc.read_h5ad(file_path)
    ensembl_ids = adata.var["ensembl_id"].tolist()
    missing_ids = [ensembl_id for ensembl_id in ensembl_ids if ensembl_id not in predefined_dict]
    if missing_ids:
        print(f"The following Ensembl IDs are missing in the dictionary: {missing_ids}")
    else:
        print("All Ensembl IDs exist in the dictionary!")
    adata = adata[:, ~adata.var.index.isin(missing_ids)]
    if 'gene_id' in adata.var.columns:
        adata.var.drop(columns=['gene_id'], inplace=True)
        print("Column 'gene_id' has been removed from adata.var.")
    else:
        print("Column 'gene_id' does not exist in adata.var.")
    file_name = Path(file_path).name
    adata.write(f"/lustre/scratch126/cellgen/team361/av13/data/pretrained_anndata2/{file_name}")


# Preprocess adata
# ----------------
args = get_args()
print('Start preprocessing adata...')
gene_median_file = "/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/filtered_trace_median.pkl"
trace_ensemble_mapping= "/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_filtered_tokenid.pkl"
trace_gene_token = "/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/Aggregated_Medians/trace_gene_mapping.pkl"
# raise
tk = TranscriptomeTokenizer(
    custom_attr_name_dict={"cell_type": "cell_type"},
    nproc=1,
    model_input_size=4096,
    collapse_gene_ids=True,
    special_token=True,
    gene_median_file=gene_median_file,
    token_dictionary_file=trace_ensemble_mapping,
)
custom_attr_name_dict={"cell_type": "cell_type"}
output_base_path = "/lustre/scratch126/cellgen/team361/av13/data/tokenized_data/trace2/"
if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

data_directory = "/lustre/scratch126/cellgen/team361/av13/data/pretrained_anndata"

for file_path in Path(data_directory).glob("*.h5ad"):
    print(f"Tokenizing {file_path}")
    file_tokenized_cells, file_cell_metadata = tk.tokenize_anndata(file_path)

    # Assume 'create_dataset' correctly formats the dataset
    tokenized_dataset = tk.create_dataset(file_tokenized_cells, file_cell_metadata)

    # Construct the output file path
    output_file_path = Path(output_base_path) / (file_path.stem + ".dataset")
    tokenized_dataset.save_to_disk(str(output_file_path))

tk.tokenize_data(
    data_directory="/lustre/scratch126/cellgen/team361/av13/data/pretrained_anndata/test",
    output_directory=output_file_path,
    output_prefix=file_name,
    file_format="h5ad",  # format [loom, h5ad]
    use_generator=True,
)

