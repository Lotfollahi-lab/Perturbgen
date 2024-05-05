import argparse
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
from scipy.sparse import csr_matrix

from T_perturb.src.utils import (
    map_ensembl_to_genename,
    map_input_ids_to_row_id,
    pairing_resting_to_activated_cells,
    subset_adata,
    tokenid_mapping,
)

seed_no = 42
np.random.seed(seed_no)

if os.getcwd().split('/')[-1] != 'healthy_imm_expr':
    # set working directory to root of repository
    os.chdir('/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/')
    print('Changed working directory to root of repository')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--h5ad_path',
        type=str,
        # default='./data/h5d_files/cytoimmgen.h5ad',
        default='./data/20240423_eb/EB.h5ad',
        help='Path to h5ad file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        # default='cytoimmgen',
        default='eb',
        choices=['cytoimmgen', 'eb'],
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
        type=list,
        # default=[
        #     'Cell_population',
        #     'Cell_type',
        #     'Time_point',
        #     'Age',
        #     'Sex',
        #     'batch',
        #     'Cell_culture_batch',
        #     'Phase',
        #     'Donor',
        #     'cell_pairing_index',
        # ],
        default=[
            'Time_point',
        ],
        help='List of variables to keep in the dataset',
    )
    parser.add_argument(
        '--pairing_mode',
        type=str,
        # default='stratified',
        default='random',
        choices=['stratified', 'random'],
        help='Cell pairing mode',
    )
    parser.add_argument(
        '--nproc',
        type=int,
        default=64,
        help='Number of processes to use for tokenisation',
    )
    parser.add_argument(
        '--reference_time',
        type=str,
        # default='0h',
        default='Day 00-03',
        help='Control time point for cell pairing' 'which is feed into Geneformer',
    )
    parser.add_argument(
        '--time_point_order',
        type=list,
        # default=[
        #     '0h',
        #     '16h',
        #     '40h',
        #     '5d',
        # ],
        default=[
            'Day 00-03',
            'Day 06-09',
            'Day 12-15',
            'Day 18-21',
            'Day 24-27',
        ],
        help='Order of time points in the dataset',
    )
    args = parser.parse_args()
    return args


# Preprocess adata
# ----------------
args = get_args()
print('Start preprocessing adata...')
adata = sc.read_h5ad(args.h5ad_path)
# required columns for tokenisation:
# adata.var['ensembl_id'] and adata.obs['n_counts']
if 'ensembl_id' not in adata.var.columns:
    adata = map_ensembl_to_genename(
        adata,
        './data/h5d_files/phase2_data_qced_cells_cellCycleScored_geneMetadata.csv.gz',
    )
    adata.var['ensembl_id'] = adata.var_names
else:
    adata.var_names = adata.var['ensembl_id']

# gene_filtering_mode = 'degs'
if args.gene_filtering_mode == 'hvg':
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata.X = adata.layers['counts']  # need raw counts
    adata = adata[:, adata.var['highly_variable']]
    del adata.layers['counts']
elif args.gene_filtering_mode == 'degs':
    # Filter adata for only DEGs
    degs = pd.read_csv(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'generative_modelling_omic_notebooks/'
        'pp/res/deg/significant_deg_1.5logfc_0.05padj_hvg_5k.csv'
    )
    unique_degs = degs['names'].unique()
    adata = adata[:, adata.var['gene_name'].isin(unique_degs)]
elif args.gene_filtering_mode == 'all':
    pass
else:
    raise ValueError('Invalid gene filtering mode')

# filter adata for only genes occuring in the token dictionary
(adata_subset, token_id_to_row_id_dict, row_id_to_gene_name) = tokenid_mapping(
    adata,
    './generative_modelling_omic/Geneformer/geneformer/token_dictionary.pkl',
)
# save mapping dictionnary
with open(
    f'./T_perturb/T_perturb/pp/res/{args.dataset}/'
    f'tokenid_to_rowid_{args.gene_filtering_mode}.pkl',
    'wb',
) as f:
    pickle.dump(token_id_to_row_id_dict, f)

with open(
    f'./T_perturb/T_perturb/pp/res/{args.dataset}'
    f'/token_id_to_genename_{args.gene_filtering_mode}.pkl',
    'wb',
) as f:
    pickle.dump(row_id_to_gene_name, f)
adata_subset.layers['counts'] = adata_subset.X.copy()
# make new directory to store h5ad files
paired_h5ad_dir = (
    f'./T_perturb/T_perturb/pp/res/{args.dataset}'
    f'/h5ad_pairing_{args.gene_filtering_mode}'
)
if not os.path.exists(paired_h5ad_dir):
    os.makedirs(paired_h5ad_dir)
# add unique index to adata obs for cell pairing
idx_column = 'cell_pairing_index'
adata_subset.obs[idx_column] = range(adata_subset.shape[0])
args.var_list = args.var_list + [idx_column]
# save filtered adata with DEGs
adata_subset.obs['n_counts'] = adata_subset.X.sum(axis=1)
# create sparse matrix if not already
if not isinstance(adata_subset.X, csr_matrix):
    adata_subset.X = csr_matrix(adata_subset.X)

adata_subset.write_h5ad(
    f'{paired_h5ad_dir}/{args.dataset}_{args.gene_filtering_mode}.h5ad'
)

print('Finished preprocessing adata.')
print('Start tokenisation of adata...')
input_dir = paired_h5ad_dir

output_dir = (
    f'./T_perturb/T_perturb/pp/res/{args.dataset}'
    f'/dataset_{args.gene_filtering_mode}'
)
var_to_keep: Dict[str, str] = {v: v for v in args.var_list}.copy()
tk = TranscriptomeTokenizer(var_to_keep, nproc=args.nproc)
tk.tokenize_data(
    input_dir,  # input directory - all h5ad files in this directory will be tokenised
    output_dir,  # output directory - tokenised h5ad files will be saved here
    f'{args.dataset}_{args.gene_filtering_mode}',  # name of output file
    file_format='h5ad',  # format [loom, h5ad]
)
print('Finished tokenisation.')
# ---------------- Cell pairing and save adata/dataset by time point ----------------
# filter and save dataset by time point
dataset = load_from_disk(
    f'{output_dir}/{args.dataset}_{args.gene_filtering_mode}.dataset'
)
adata_subset = sc.read_h5ad(
    f'{paired_h5ad_dir}/{args.dataset}_{args.gene_filtering_mode}.h5ad'
)

# Pairing resting to activated cells and tokenise individual datasets
cell_pairings = pairing_resting_to_activated_cells(
    adata_subset=adata_subset,
    pairing_mode=args.pairing_mode,
    seed_no=seed_no,
    reference_time=args.reference_time,
)
paired_dataset_dir = f'./res/dataset_{args.gene_filtering_mode}'
if not os.path.exists(paired_dataset_dir):
    os.makedirs(paired_dataset_dir)
dataset_mapped = dataset.map(
    lambda example: map_input_ids_to_row_id(example, token_id_to_row_id_dict)
)
n_tgt_iter = 1  # for enumerating the timepoints
for time_point in tqdm.tqdm(args.time_point_order):
    # subset adata by cell pairings
    adata_tmp = subset_adata(adata_subset, cell_pairings[time_point])
    # separate src and target directory
    if not os.path.exists(f'{paired_h5ad_dir}_src'):
        os.makedirs(f'{paired_h5ad_dir}_src')
    if not os.path.exists(f'{paired_h5ad_dir}_tgt'):
        os.makedirs(f'{paired_h5ad_dir}_tgt')
    if not os.path.exists(f'{output_dir}_src'):
        os.makedirs(f'{output_dir}_src')
    if not os.path.exists(f'{output_dir}_tgt'):
        os.makedirs(f'{output_dir}_tgt')

    # subset dataset by cell pairings
    if time_point == args.reference_time:
        adata_tmp.write_h5ad(f'./{paired_h5ad_dir}_src/{time_point}.h5ad')
        # do not map input ids to row ids for Geneformer input
        # Geneformer needs initial token ids to extract correct embeddings
        dataset_tmp = dataset.select(cell_pairings[time_point])
        dataset_tmp.save_to_disk(f'{output_dir}_src/{time_point}.dataset')
    else:
        adata_tmp.write_h5ad(
            f'{paired_h5ad_dir}_tgt/' f'{n_tgt_iter}_{time_point}.h5ad'
        )
        dataset_tmp = dataset_mapped.select(cell_pairings[time_point])
        dataset_tmp.save_to_disk(f'{output_dir}_tgt/{n_tgt_iter}_{time_point}.dataset')
        n_tgt_iter += 1
