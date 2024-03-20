import os
import pickle
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
gene_filtering_mode = 'hvg'
# gene_filtering_mode = 'degs'
if gene_filtering_mode == 'hvg':
    adata.layers['counts'] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata.X = adata.layers['counts']  # need raw counts
    adata = adata[:, adata.var['highly_variable']]
    del adata.layers['counts']
else:
    # Filter adata for only DEGs
    degs = pd.read_csv(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'generative_modelling_omic_notebooks/'
        'pp/res/deg/significant_deg_1.5logfc_0.05padj_hvg_5k.csv'
    )
    unique_degs = degs['names'].unique()
    adata = adata[:, adata.var['gene_name'].isin(unique_degs)]
# filter adata for only genes occuring in the token dictionary
tokenid_to_subsetid, adata_subset = map_deg_to_tokenid(
    adata,
    Path(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'generative_modelling_omic/Geneformer/geneformer/token_dictionary.pkl'
    ),
)
with open(
    f'./res/tokenid_to_subsetid_{gene_filtering_mode}.pkl',
    'wb',
) as f:
    pickle.dump(tokenid_to_subsetid, f)

adata_subset, token_id_to_genename = map_token_id_to_genename(adata_subset)
with open(
    f'./res/token_id_to_genename_{gene_filtering_mode}.pkl',
    'wb',
) as f:
    pickle.dump(token_id_to_genename, f)
adata_subset.layers['counts'] = adata_subset.X.copy()
# make new directory to store h5ad files
paired_h5ad_dir = f'./res/h5ad_pairing_{gene_filtering_mode}'
if not os.path.exists(paired_h5ad_dir):
    os.makedirs(paired_h5ad_dir)
# add unique index to adata obs for cell pairing
adata_subset.obs['cell_pairing_index'] = range(adata_subset.shape[0])
# save filtered adata with DEGs
adata_subset.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenised_{gene_filtering_mode}.h5ad'
)
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
    f'cytoimmgen_tokenised_{gene_filtering_mode}_paired',  # name of output file
    file_format='h5ad',  # format [loom, h5ad]
)
print('Finished tokenisation.')
# ---------------- Cell pairing and save adata/dataset by time point ----------------
# filter and save dataset by time point
dataset = load_from_disk(
    f'{output_dir}/' f'cytoimmgen_tokenised_{gene_filtering_mode}_paired.dataset'
)
adata_subset = sc.read_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenised_{gene_filtering_mode}.h5ad'
)
pairing_mode = 'stratified'
# Pairing resting to activated cells and tokenise individual datasets
cell_pairings = pairing_resting_to_activated_cells(
    adata_subset=adata_subset, pairing_mode=pairing_mode, seed=seed_no
)

adata_0h = subset_adata(adata_subset, cell_pairings['0h'])
adata_16h = subset_adata(adata_subset, cell_pairings['16h'])
adata_40h = subset_adata(adata_subset, cell_pairings['40h'])
adata_5d = subset_adata(adata_subset, cell_pairings['5d'])


adata_0h.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_{pairing_mode}_pairing_0h.h5ad'
)
adata_16h.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_{pairing_mode}_pairing_16h.h5ad'
)
adata_40h.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_{pairing_mode}_pairing_40h.h5ad'
)
adata_5d.write_h5ad(
    f'{paired_h5ad_dir}/cytoimmgen_tokenisation_{pairing_mode}_pairing_5d.h5ad'
)


# use dictionary to map token_id to input_ids
def map_input_ids(dataset):
    dataset['input_ids'] = [
        tokenid_to_subsetid.get(item, item) for item in dataset['input_ids']
    ]
    return dataset


dataset = dataset.map(map_input_ids)
paired_dataset_dir = f'./res/dataset_{gene_filtering_mode}'
if not os.path.exists(paired_dataset_dir):
    os.makedirs(paired_dataset_dir)
dataset_0h = dataset.select(cell_pairings['0h'])
dataset_16h = dataset.select(cell_pairings['16h'])
dataset_40h = dataset.select(cell_pairings['40h'])
dataset_5d = dataset.select(cell_pairings['5d'])
dataset_0h.save_to_disk(
    f'{paired_dataset_dir}/' f'cytoimmgen_tokenised_{pairing_mode}_pairing_0h.dataset'
)
dataset_16h.save_to_disk(
    f'{paired_dataset_dir}/' f'cytoimmgen_tokenised_{pairing_mode}_pairing_16h.dataset'
)
dataset_40h.save_to_disk(
    f'{paired_dataset_dir}/' f'cytoimmgen_tokenised_{pairing_mode}_pairing_40h.dataset'
)
dataset_5d.save_to_disk(
    f'{paired_dataset_dir}/' f'cytoimmgen_tokenised_{pairing_mode}_pairing_5d.dataset'
)
