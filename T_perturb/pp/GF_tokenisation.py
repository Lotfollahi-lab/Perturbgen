import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import scanpy as sc
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer

from T_perturb.src.utils import (
    map_deg_to_tokenid,
    map_ensembl_to_genename,
    map_token_id_to_genename,
    pairing_resting_to_activated_cells,
    subset_adata,
)

seed_no = 42
np.random.seed(seed_no)

if os.getcwd().split('/')[-3] != 'T_perturb':
    # set working directory to root of repository
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/pp'
    )
    print('Changed working directory to root of repository')

# Preprocess adata
# ----------------
print('Start preprocessing adata...')
adata = sc.read_h5ad(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
    't_generative/data/h5d_files/cytoimmgen.h5ad'
)
adata = map_ensembl_to_genename(
    adata,
    Path(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'data/h5d_files/phase2_data_qced_cells_cellCycleScored_geneMetadata.csv.gz'
    ),
)

adata.var['ensembl_id'] = adata.var.index
# Filter adata for only DEGs
degs = pd.read_csv(
    '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
    'generative_modelling_omic_notebooks/'
    'pp/res/deg/significant_deg_1.5logfc_0.05padj_hvg_5k.csv'
)
unique_degs = degs['names'].unique()

adata = adata[:, adata.var['gene_name'].isin(unique_degs)]
# filter adata for only genes occuring in the token dictionary
deg_to_tokenid_dict, adata_subset = map_deg_to_tokenid(
    adata,
    Path(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'generative_modelling_omic/Geneformer/geneformer/token_dictionary.pkl'
    ),
)

adata_subset = map_token_id_to_genename(adata_subset)
adata_subset.layers['counts'] = adata_subset.X.copy()
# make new directory to store h5ad files
paired_h5ad_dir = './res/h5ad_pairing'
if not os.path.exists(paired_h5ad_dir):
    os.makedirs(paired_h5ad_dir)
# add unique index to adata obs for cell pairing
adata_subset.obs['cell_pairing_index'] = range(adata_subset.shape[0])
# save filtered adata with DEGs
adata_subset.write_h5ad(f'{paired_h5ad_dir}/cytoimmgen_tokenised_degs.h5ad')
var_list = [
    'Cell_population',
    'Cell_type',
    'Time_point',
    'Age',
    'Sex',
    'batch',
    'Cell_culture_batch',
    'Phase',
    'Donor',
    'cell_pairing_index',
]
var_to_keep: Dict[str, str] = {v: v for v in var_list}.copy()
print('Finished preprocessing adata.')
print('Start tokenisation of adata...')
input_dir = paired_h5ad_dir
output_dir = './res/dataset'
tk = TranscriptomeTokenizer(var_to_keep, nproc=16)
tk.tokenize_data(
    input_dir,  # input directory - all h5ad files in this directory will be tokenised
    output_dir,  # output directory - tokenised h5ad files will be saved here
    'cytoimmgen_tokenised_degs_paired',  # name of output file
    file_format='h5ad',  # format [loom, h5ad]
)
print('Finished tokenisation.')
# filter and save dataset by time point
dataset = load_from_disk('./res/dataset/cytoimmgen_tokenised_degs_paired.dataset')

pairing_mode = 'stratified'
# Pairing resting to activated cells and tokenise individual datasets
cell_pairings = pairing_resting_to_activated_cells(
    adata_subset=adata_subset, pairing_mode=pairing_mode, seed=seed_no
)
adata_0h = subset_adata(adata_subset, cell_pairings['0h'])
adata_16h = subset_adata(adata_subset, cell_pairings['16h'])
adata_40h = subset_adata(adata_subset, cell_pairings['40h'])
adata_5d = subset_adata(adata_subset, cell_pairings['5d'])
obs_cols = [
    'cell_pairing_index',
    'Cell_type',
    'Donor',
    'Cell_culture_batch',
    'Cell_population',
    'Phase',
    'Age',
    'Sex',
]


def prune_adata(adata, obs_cols):
    adata.obs = adata.obs[obs_cols]
    adata.var = adata.var[['gene_name', 'ensembl_id']]
    return adata


adata_0h.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_degs_{pairing_mode}_pairing_0h.h5ad'
)
adata_16h.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_degs_{pairing_mode}_pairing_16h.h5ad'
)
adata_40h.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_degs_{pairing_mode}_pairing_40h.h5ad'
)
adata_5d.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_degs_{pairing_mode}_pairing_5d.h5ad'
)


# use dictionary to map token_id to input_ids
def map_input_ids(dataset):
    dataset['input_ids'] = [
        deg_to_tokenid_dict.get(item, item) for item in dataset['input_ids']
    ]
    return dataset


dataset = dataset.map(map_input_ids)
dataset_0h = dataset.select(cell_pairings['0h'])
dataset_16h = dataset.select(cell_pairings['16h'])
dataset_40h = dataset.select(cell_pairings['40h'])
dataset_5d = dataset.select(cell_pairings['5d'])
dataset_0h.save_to_disk(
    f'./res/dataset/cytoimmgen_tokenised_degs_{pairing_mode}_pairing_0h.dataset'
)
dataset_16h.save_to_disk(
    f'./res/dataset/cytoimmgen_tokenised_degs_{pairing_mode}_pairing_16h.dataset'
)
dataset_40h.save_to_disk(
    f'./res/dataset/cytoimmgen_tokenised_degs_{pairing_mode}_pairing_40h.dataset'
)
dataset_5d.save_to_disk(
    f'./res/dataset/cytoimmgen_tokenised_degs_{pairing_mode}_pairing_5d.dataset'
)
