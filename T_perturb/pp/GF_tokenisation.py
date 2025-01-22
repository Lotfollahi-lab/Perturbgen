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
from scipy.sparse import csr_matrix, issparse

from T_perturb.src.utils import (  # tokenid_mapping,;
    map_ensembl_to_genename,
    map_input_ids_to_row_id,
    pairing_src_to_tgt_cells,
    str2bool,
    subset_adata,
    tokenid_mapping,
)

seed_no = 42
np.random.seed(seed_no)

if os.getcwd().split('/')[-1] != 't_generative':
    # set working directory to root of repository
    os.chdir('/lustre/scratch126/cellgen/team361/kl11/t_generative/')
    print('Changed working directory to root of repository')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--h5ad_path',
        type=str,
        # default='./data/h5d_files/cytoimmgen.h5ad',
        default='./data/20241026_HSPC/cd34.h5ad',
        help='Path to h5ad file',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        # default='cytoimmgen',
        default='hspc_pbmc_median',
        choices=[
            'cytoimmgen',
            'cytoimmgen_pbmc_median',
            'eb',
            'eb_pbmc_median',
            'eb_GF_26k_median',
            'mnc',
            'hspc',
            'hspc_pbmc_median',
            'hspc_GF_26k_median',
        ],
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
            'assignment_id',
            # 'age',
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
        # default='stratified',
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
        # default='0h',
        default='stem',
        help='Control time point for cell pairing' 'which is feed into Geneformer',
    )
    parser.add_argument(
        '--time_point_order',
        type=str,
        nargs='+',
        # default=[
        #     '0h',
        #     '16h',
        #     '40h',
        #     '5d',
        # ],
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
        default=10000,
        help='Number of highly variable genes to keep',
    )
    parser.add_argument(
        '--gene_median_path',
        type=str,
        default='/lustre/scratch126/cellgen/team361/am74/'
        'Adib/TRACE/Loom_cohort/tdigest/2nd_run/'
        'Dictionaries/filtered_trace_median.pkl',
        # default='T_perturb/Geneformer/geneformer/gene_median_dictionary_gc95M.pkl',
        help='Path to gene median file',
    )
    parser.add_argument(
        '--token_dict_path',
        type=str,
        default='/lustre/scratch126/cellgen/team361/am74/'
        'Adib/TRACE/Loom_cohort/tdigest/2nd_run/'
        'Dictionaries/trace_filtered_tokenid.pkl',
        # default='T_perturb/Geneformer/geneformer/token_dictionary_gc95M.pkl',
        help='Path to token dictionary file',
    )
    parser.add_argument(
        '--gene_mapping_path',
        type=str,
        # default='/lustre/scratch126/cellgen/team361/am74/'
        # 'Adib/TRACE/Loom_cohort/tdigest/2nd_run/'
        # 'Dictionaries/filtered_trace_median.pkl',
        default='T_perturb/Geneformer/geneformer/gene_name_id_dict_gc95M.pkl',
        help='Path to gene mapping file',
    )
    args = parser.parse_args()
    return args


# Preprocess adata
# default='0h',
# ----------------
args = get_args()
print('Start preprocessing adata...')
adata = sc.read_h5ad(args.h5ad_path)

gene_filter_mode_suffix = (
    f'{args.gene_filtering_mode}'
    if args.gene_filtering_mode == 'all'
    else f'{args.n_hvg}_{args.gene_filtering_mode}'
)

if args.dataset.startswith('cytoimmgen'):
    adata = map_ensembl_to_genename(
        adata,
        './data/h5d_files/phase2_data_qced_cells_cellCycleScored_geneMetadata.csv.gz',
    )
    adata.var['ensembl_id'] = adata.var_names
else:
    adata.var['gene_name'] = adata.var_names
    adata.var_names = adata.var['ensembl_id']

# gene_filtering_mode = 'degs'
if args.gene_filtering_mode == 'hvg':
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    else:
        adata.X = adata.layers['counts'].copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(
        adata, n_top_genes=args.n_hvg, batch_key=args.pairing_obs
    )
    adata = adata[:, adata.var['highly_variable']].copy()
    adata.X = adata.layers['counts']  # need raw counts
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
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    else:
        adata.X = adata.layers['counts'].copy()
else:
    raise ValueError('Invalid gene filtering mode')

# create gene mapping file
with open(args.token_dict_path, 'rb') as f:
    token_dict = pickle.load(f)
with open(args.gene_mapping_path, 'rb') as f:
    genename_dict = pickle.load(f)


# exclude special tokens
special_tokens = [0, 1, 2, 3]
token_dict = {k: v for k, v in token_dict.items() if v not in special_tokens}
# change keys and values of genename_dict
genename_dict = {v: k for k, v in genename_dict.items()}
# map token_dict to gene names
token_to_genename = {
    v: genename_dict[k] for k, v in token_dict.items() if k in genename_dict
}

os.makedirs(f'./T_perturb/T_perturb/pp/res/{args.dataset}', exist_ok=True)

# save token_dict
with open(
    f'./T_perturb/T_perturb/pp/res/{args.dataset}/'
    f'token_id_to_genename_{gene_filter_mode_suffix}.pkl',
    'wb',
) as file:
    pickle.dump(token_to_genename, file)


# # ignore CLS token because of redudancy with time task token
# if pd.NA in token_id_to_row_id_dict:
#     del token_id_to_row_id_dict[pd.NA]

# # save mapping dictionnary
# with open(
#     f'./T_perturb/T_perturb/pp/res/{args.dataset}/'
#     f'tokenid_to_rowid_{args.gene_filtering_mode}.pkl',
#     'wb',
# ) as f:
#     pickle.dump(token_id_to_row_id_dict, f)


# make new directory to store h5ad files
paired_h5ad_dir = (
    f'./T_perturb/T_perturb/pp/res/{args.dataset}'
    f'/h5ad_pairing_{gene_filter_mode_suffix}'
)

if not os.path.exists(paired_h5ad_dir):
    os.makedirs(paired_h5ad_dir)
# add unique index to adata obs for cell pairing
idx_column = 'cell_pairing_index'
adata.obs[idx_column] = range(adata.shape[0])
adata.obs.index = adata.obs[idx_column].astype(str)
adata.obs.index.name = None

# create sparse matrix if not already
if not (issparse(adata.X)):
    adata.X = csr_matrix(adata.X)
# adata.obs = adata.obs[args.var_list]
adata.var = adata.var[['gene_name', 'ensembl_id']]
adata.obs['n_counts'] = adata.X.sum(axis=1)
# save adata
adata.write_h5ad(f'{paired_h5ad_dir}/{args.dataset}.h5ad')

# # load adata
# adata = sc.read_h5ad(
#     f'{paired_h5ad_dir}/{args.dataset}.h5ad'
# )

# subset adata to only genes in the token dictionary
# filter adata for only genes occuring in the token dictionary
if args.gene_filtering_mode == 'all':
    adata_subset = adata
else:
    (adata_subset, token_id_to_row_id_dict, row_id_to_gene_name) = tokenid_mapping(
        adata,
        # './T_perturb/Geneformer/geneformer/token_dictionary_gc95M.pkl',
        args.token_dict_path,
        exclude_non_GF_genes=True,
    )
    # save row id to gene name mapping
    with open(
        f'./T_perturb/T_perturb/pp/res/{args.dataset}'
        f'/token_id_to_genename_{gene_filter_mode_suffix}.pkl',
        'wb',
    ) as file:
        pickle.dump(row_id_to_gene_name, file)
    # save token id to row id mapping
    with open(
        f'./T_perturb/T_perturb/pp/res/{args.dataset}/'
        f'tokenid_to_rowid_{gene_filter_mode_suffix}.pkl',
        'wb',
    ) as file:
        pickle.dump(token_id_to_row_id_dict, file)
    if args.exclude_non_GF_genes is True:
        adata_subset.write_h5ad(f'{paired_h5ad_dir}/{args.dataset}.h5ad')
print('Finished preprocessing adata.')
print('Start tokenisation of adata...')
output_dir = (
    f'./T_perturb/T_perturb/pp/res/{args.dataset}/' f'dataset_{gene_filter_mode_suffix}'
)
var_to_keep: Dict[str, str] = {v: v for v in args.var_list}.copy()
# add cell_pairing_index to var_to_keep
var_to_keep['cell_pairing_index'] = 'cell_pairing_index'

tk = TranscriptomeTokenizer(
    custom_attr_name_dict=var_to_keep,
    nproc=4,
    model_input_size=4096,
    collapse_gene_ids=True,
    special_token=True,
    gene_median_file=args.gene_median_path,
    token_dictionary_file=args.token_dict_path,
    # gene_mapping_file=args.gene_mapping_path,
)
# time it
# Proceed with your main logic
file_name = args.dataset

tk.tokenize_data(
    data_directory=paired_h5ad_dir,
    output_directory=output_dir,
    output_prefix=file_name,
    file_format='h5ad',  # format [loom, h5ad]
    use_generator=False,
)

print('Finished tokenisation.')
# ---------------- Cell pairing and save adata/dataset by time point ----------------
# filter and save dataset by time point
dataset = load_from_disk(f'{output_dir}/{file_name}.dataset')
# load csv
if args.pairing_mode == 'mapping':
    mapping_df = pd.read_csv('T_perturb/T_perturb/pp/res/hspc/cd34_pos_mapping.csv')
else:
    mapping_df = None
adata_subset = sc.read_h5ad(f'{paired_h5ad_dir}/{args.dataset}.h5ad')

# # Pairing resting to activated cells and tokenise individual datasets
cell_pairings = pairing_src_to_tgt_cells(
    adata_subset=adata_subset,
    pairing_mode=args.pairing_mode,
    pairing_obs=args.pairing_obs,
    seed_no=seed_no,
    mapping_df=mapping_df,
)

paired_dataset_dir = (
    f'./T_perturb/T_perturb/res/{args.dataset}/' f'dataset_{gene_filter_mode_suffix}'
)
# token_id_to_row_id_dict = pickle.load(
#     open(
#         f'./T_perturb/T_perturb/pp/res/{args.dataset}/'
#         f'tokenid_to_rowid_{args.gene_filtering_mode}.pkl',
#         'rb',
#     )
# )
if args.gene_filtering_mode == 'all':
    dataset_mapped = dataset
else:
    dataset_mapped = dataset.map(
        lambda example: map_input_ids_to_row_id(
            example, token_id_to_row_id_dict, ignore_tokens=[2]
        ),
        num_proc=4,
    )
n_tgt_iter = 1  # for enumerating the timepoints
for time_point in tqdm.tqdm(args.time_point_order):
    # subset adata by cell pairings
    adata_tmp = subset_adata(adata_subset, cell_pairings[time_point])
    # separate src and target directory
    src_adata_dir = f'{paired_h5ad_dir}_src'
    tgt_adata_dir = f'{paired_h5ad_dir}_tgt'
    src_dataset_dir = f'{output_dir}_src'
    tgt_dataset_dir = f'{output_dir}_tgt'
    if not os.path.exists(src_adata_dir):
        os.makedirs(src_adata_dir)
    if not os.path.exists(tgt_adata_dir):
        os.makedirs(tgt_adata_dir)
    if not os.path.exists(src_dataset_dir):
        os.makedirs(src_dataset_dir)
    if not os.path.exists(tgt_dataset_dir):
        os.makedirs(tgt_dataset_dir)
    # subset dataset by cell pairings
    if time_point == args.reference_time:
        adata_tmp.write_h5ad(f'{src_adata_dir}/{time_point}.h5ad')
        # do not map input ids to row ids for Geneformer input
        # Geneformer needs initial token ids to extract correct embeddings
        if args.src_mode == 'Geneformer':
            dataset_tmp = dataset.select(cell_pairings[time_point])
            dataset_tmp.save_to_disk(f'{src_dataset_dir}/{time_point}.dataset')
        else:
            src_adata_transf_dir = f'{paired_h5ad_dir}_src_transformer'
            if not os.path.exists(src_adata_transf_dir):
                os.makedirs(src_adata_transf_dir)
            dataset_tmp = dataset_mapped.select(cell_pairings[time_point])
            dataset_tmp.save_to_disk(f'{src_adata_transf_dir}/{time_point}.dataset')
    else:
        adata_tmp.write_h5ad(f'{tgt_adata_dir}/' f'{n_tgt_iter}_{time_point}.h5ad')
        dataset_tmp = dataset_mapped.select(cell_pairings[time_point])
        dataset_tmp.save_to_disk(f'{tgt_dataset_dir}/{n_tgt_iter}_{time_point}.dataset')
        n_tgt_iter += 1
