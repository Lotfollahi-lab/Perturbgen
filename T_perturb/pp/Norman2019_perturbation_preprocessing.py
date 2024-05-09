print('imports')
import os
import pickle
from typing import Dict
import numpy as np
import pandas as pd
import scanpy as sc
from datasets import load_from_disk
from geneformer import TranscriptomeTokenizer
import matplotlib.pyplot as plt
from typing import Dict, List
import tqdm
import gc
from datasets import load_from_disk
from pertpy import data
import itertools
from gears.data_utils import get_dropout_non_zero_genes, get_DE_genes
from transformers import BertForMaskedLM
from torch import tensor
from T_perturb.src.utils import (
    subset_adata,
)

# Set up -------------------
# paths
# base_path = '/lustre/groups/imm01/workspace/irene.bonafonte/Projects/2024Mar_Tperturb'
base_path = '/lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb'
data_path = 'datasets'
dataset_name = 'Norman2019'
pp_path = 'T_perturb/T_perturb/pp/res'
geneformer_path = f'{base_path}/../../Software/Geneformer'


# arguments
gene_filtering_mode = 'hvg'

print('download')
# download ----------------
adata = data.norman_2019()

# pre-process ----------------
# add some annotations to the perturbations
print('preprocess')
ann_f = f'{base_path}/{data_path}/{dataset_name}/metadata/norman_annotation.csv'
if os.path.exists(ann_f):
    annotation = pd.read_csv(ann_f, sep=';')
    annotation = {p: a for p, a in zip(annotation.Perturbation.values, annotation.Annotation.values)}
    adata.obs['perturbation_annotation'] = adata.obs['perturbation_name'].map(annotation)
    adata.obs.loc[adata.obs.perturbation_annotation.isna(),'perturbation_annotation'] = ''
    adata.obs['perturbation_annotation'] = adata.obs['perturbation_annotation'].astype(str)

# reformat perturbation name
adata.obs['n_perturbations'] = 1
adata.obs.loc[adata.obs.perturbation_name=='control', 'n_perturbations'] = 0
adata.obs.loc[adata.obs.perturbation_name.str.contains('+', regex=False),'n_perturbations'] = 2
adata.obs['perturbation_name_l'] = adata.obs['perturbation_name'].str.split('+').values
adata.obs['perturbation1_name'] = adata.obs['perturbation_name_l'].apply(lambda x: x[0])
adata.obs['perturbation2_name'] = adata.obs['perturbation_name_l'].apply(lambda x: x[1] if len(x) > 1 else 'control')

# re-naming for geneformer
adata.var['gene_name'] = adata.var_names
adata.var = adata.var.rename(columns={'index': 'ensembl_id'})
adata.var_names = adata.var['ensembl_id']
adata.obs = adata.obs.rename(columns={'total_counts': 'n_counts'})

# add token ids
tpath = f'{geneformer_path}/geneformer/token_dictionary.pkl'
with open(tpath, 'rb') as f:
    token_id_dict = pickle.load(f)
    
adata.var['token_id'] = adata.var_names.map(token_id_dict)
adata = adata[:, adata.var['token_id'].notna()]
adata.var['token_id'] = adata.var['token_id'].astype('Int64')

# list of perturbed genes
import itertools
perturbations = [p for p in set(itertools.chain.from_iterable(adata.obs.perturbation_name_l.to_list())) if p != 'control']

# exclude cells with perturbations on genes without a token id
missing_perturbations = set([p for p in perturbations if not p in adata.var.gene_name.values])
print('The following perturbed genes cannot be tokenized with GF: ', missing_perturbations)
missing_mask = adata.obs.perturbation_name_l.apply(lambda x: len(missing_perturbations.intersection(x)) > 0)
adata = adata[~missing_mask]

# after cell filtering, add unique index to adata obs for cell pairing
adata.obs['cell_pairing_index'] = range(adata.shape[0])

# highly variable gene filter + perturbed genes
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
adata.var.loc[adata.var.gene_name.isin(perturbations),'highly_variable'] = True
adata.X = adata.layers['counts'] # we need raw counts
adata = adata[:, adata.var['highly_variable']] # filter

# update token_id_dict with index instead of ensembl IDs
token_id_df = adata.var.copy()
adata.var['rowidx'] = np.arange(0, len(token_id_df)) + 1
token_id_df.index = np.arange(0, len(token_id_df)) + 1
token_id_dict = dict(zip(token_id_df['token_id'], token_id_df.index))
token_id_dict[0] = 0

# save row idx to GF token id dictionary
with open(f'{base_path}/{pp_path}/{dataset_name}_token_dictionary_{gene_filtering_mode}.pkl', 'wb') as f:
    pickle.dump(token_id_dict, f)

# perturbed genes info
adata.uns['perturbation_id'] = token_id_df.loc[token_id_df.gene_name.isin(perturbations), ['gene_name','ensembl_id','token_id']].copy()
adata.uns['perturbation_id']['rowidx'] = adata.uns['perturbation_id'].index
adata.obs.drop(columns='perturbation_name_l', inplace=True) # can't be written with adata

# Differentially expressed genes per perturbation (for metric) ----------------
print('deg gears')
# reformat for GEARS
adata.obs['condition'] = adata.obs.perturbation_name.astype(str).copy()
adata.obs.loc[adata.obs['condition']=='control','condition'] = 'ctrl'
adata.obs.loc[adata.obs['n_perturbations']==1,'condition'] = adata.obs.loc[adata.obs['n_perturbations']==1,'condition'] + '+ctrl'
adata.obs['cell_type'] = 'A549'
adata_gears = adata.copy()

# compute DEG per perturbation
adata_gears = get_DE_genes(adata_gears, False)
adata_gears = get_dropout_non_zero_genes(adata_gears)

# change to our IDs and add to adata
gears2gene = adata_gears.obs[['condition_name','perturbation_name']].drop_duplicates().set_index('condition_name', inplace=False, drop=True).to_dict()['perturbation_name']
ens2idx = adata_gears.var['rowidx'].to_dict()

# add to adata
adata.uns['top_non_dropout_de_20'] = {gears2gene[p]: np.vectorize(ens2idx.__getitem__)(g) for p, g in adata_gears.uns['top_non_dropout_de_20'].items()}
del adata_gears
gc.collect()

# Save pre-processed object
adata.write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_hvg.h5ad')

# Tokenize ---------
print('tokenize')
var_list = [var for var in ['guide_identity', 'perturbation_name', 'perturbation_annotation', 'n_perturbations','perturbation1_name', 'perturbation2_name', 'leiden', 'pct_counts_mt', 'n_genes_by_counts', 'n_counts', 'cell_pairing_index'] if var in adata.obs.columns]
var_to_keep: Dict[str, str] = {v: v for v in var_list}.copy()

tk = TranscriptomeTokenizer(var_to_keep, nproc=15) #, model_input_size=5000)
tk.tokenize_data(
    f'{base_path}/{data_path}/{dataset_name}/adata/',  # input directory - all h5ad files in this directory will be tokenised
    f'{base_path}/{data_path}/{dataset_name}/dataset',  # output directory - tokenised h5ad files will be saved here
    f'filtered_tokenised_hvg',  # name of output file
    file_format='h5ad',  # format [loom, h5ad]
)

# Pairing --------
print('pairing')
# load
dataset = load_from_disk(f'{base_path}/{data_path}/{dataset_name}/dataset/filtered_tokenised_hvg.dataset')
adata = sc.read_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_hvg.h5ad')
adata.obs = adata.obs.reset_index()

# obs df for ctrl and perturbed
adata_ctrl = adata.obs.loc[adata.obs['perturbation_name'] == 'control', :]
adata_perturbed = adata.obs.loc[adata.obs['perturbation_name'] != 'control', :]

# ctrl dataset to check if perturbed genes are in input_ids
dataset_ctrl = dataset.select(adata_ctrl.index.values)

# replace adata_ctrl index by row number
adata_ctrl = adata_ctrl.reset_index(drop=True)

# new df to store info about perturbation occurrance
multiple_perturb = adata.obs[adata.obs.n_perturbations>1].perturbation_name.unique()
multiple_perturb = pd.DataFrame({
    'gene_name': multiple_perturb,
    'token_id': ['+'.join(adata.uns['perturbation_id'].loc[adata.uns['perturbation_id'].gene_name.isin(mp.split('+')),'token_id'].astype(str).tolist()) for mp in multiple_perturb],
    'rowidx': ['+'.join(adata.uns['perturbation_id'].loc[adata.uns['perturbation_id'].gene_name.isin(mp.split('+')),'rowidx'].astype(str).tolist()) for mp in multiple_perturb],
    'n_perturbations': [2 for p in multiple_perturb]
}, index=multiple_perturb)

adata.uns['perturbations'] = adata.uns['perturbation_id'].copy().set_index('gene_name', drop=False).drop(columns='ensembl_id')
adata.uns['perturbations']['n_perturbations'] = 1
adata.uns['perturbations'] = pd.concat([adata.uns['perturbations'],multiple_perturb])
adata.uns['perturbations']['n_cells'] = adata.obs['perturbation_name'].value_counts()[adata.uns['perturbations'].gene_name] # number of cells with that perturbation

# cells where single perturbed genes are present
ctrl_cells_idx = {p: [] for p in adata.uns['perturbation_id'].token_id.values}
for cell in tqdm.tqdm(dataset_ctrl):
    for gene in cell['input_ids']:
        if gene in ctrl_cells_idx.keys():
            ctrl_cells_idx[gene].append(cell['cell_pairing_index'])

# cells where multiple perturbed genes are present (get it using intersection from prev)
for mp in adata.uns['perturbations'].loc[adata.uns['perturbations'].n_perturbations == 2, 'token_id'].str.split('+'):
    ctrl_cells_idx[f'{mp[0]}+{mp[1]}'] = list(set(ctrl_cells_idx[int(mp[0])]).intersection(ctrl_cells_idx[int(mp[1])]))

# add in uns how many control cells have the perturbed genes within input list
adata.uns['perturbations']['in_n_ctrl_cells'] = [len(ctrl_cells_idx[p]) for p in adata.uns['perturbations']['token_id']]

# Exclude perturbations of genes not sufficiently expressed
filters = {'in_n_ctrl_cells': 50, 'n_cells': 50}
include = adata.uns['perturbations'].gene_name.tolist()
print(f'Initial number of perturbations: {len(include)}')
for f, v in filters.items():
    to_del = adata.uns['perturbations']['gene_name'].values[adata.uns['perturbations'][f] < v].tolist()
    for el in to_del:
        include.remove(el)
print(f'Number of perturbations after filtering: {len(include)}')

# initiate dictionary to store cell pairings
cell_pairings: Dict[str, List[int]] = {'control': [], 'perturbed': [], 'perturbed_gene': []}

# randomly sample from each time point (two samples: ctrl or perturbed -including all perturbations-, ctrl cells are repeated)
# while removing from the unperturbed, the perturbed gene
for idx, row in tqdm.tqdm(adata_perturbed.iterrows(), total=adata_perturbed.shape[0]):
    perturbed_genes = adata_perturbed.loc[idx, 'perturbation_name']
    if perturbed_genes in include:
        # perturbation name
        perturbed_token = adata.uns['perturbations'].loc[perturbed_genes, 'token_id']
        perturbed_idx = adata.uns['perturbation_id'].loc[adata.uns['perturbation_id'].gene_name.isin(perturbed_genes.split('+')),'rowidx'].tolist()
        cell_pairings['perturbed_gene'].append(perturbed_idx) 
        # perturbed cell idx
        cell_pairings['perturbed'].append(idx)
        # choose resting within cells expressing the perturbed gene
        cell_pairings['control'].append(np.random.choice(ctrl_cells_idx[perturbed_token])) 

# Encode perturbation
with open(f'{base_path}/{pp_path}/{dataset_name}_token_dictionary_{gene_filtering_mode}.pkl', 'rb') as f:
    token_id_dict = pickle.load(f)

# use dictionary to map token_id to input_ids
def map_input_ids(dataset_):
    dataset_['input_ids'] = [
        token_id_dict.get(item, item) for item in dataset_['input_ids']
    ]
    return dataset_

# divide into ctrl and perturbed
dataset_ctrl = dataset.select(cell_pairings['control'])
dataset_perturbed = dataset.select(cell_pairings['perturbed'])

# map token ids to our 5k dimension for the perturbed dataset
dataset_perturbed = dataset_perturbed.map(map_input_ids)

# add column with perturbed gene
dataset_ctrl = dataset_ctrl.add_column("perturbation_id", cell_pairings['perturbed_gene'])
dataset_perturbed = dataset_perturbed.add_column("perturbation_id", cell_pairings['perturbed_gene'])

# add list of genes to use for testing in the perturbed dataset
dataset_perturbed = dataset_perturbed.add_column("testing_genes_subset", [adata.uns['top_non_dropout_de_20'][p] for p in dataset_perturbed['perturbation_name']])
dataset_perturbed.save_to_disk(f'{base_path}/{data_path}/dataset/filtered_tokenised_hvg_pairing_perturbed.dataset')

# add perturbed gene embedding to the control dataset - Geneformer -----------------
gf = BertForMaskedLM.from_pretrained(
            f'{geneformer_path}/geneformer-12L-30M',
            output_attentions=False,
            output_hidden_states=True
)
print(gf._modules['bert'].embeddings.word_embeddings)

# get gene embeddings for perturbed genes from GF (including padding value)
gene_embeddings = pd.DataFrame(
    np.squeeze(gf._modules['bert'].embeddings.word_embeddings(tensor(np.insert(adata.uns['perturbation_id'].token_id.values, 0, 0).astype(int))).detach().numpy()), 
    index=np.insert(adata.uns['perturbation_id'].rowidx.values, 0, 0)
)

# store gene embedding of each perturbed gene (list of embeddings)
# if only one perturbation, add a padded value in 2nd position
gene_embeddings_list = []
for pg in tqdm.tqdm(cell_pairings['perturbed_gene']):
    if len(pg) == 1:
        pg = pg + [0]
    gene_embeddings_list.append([embed.values for row, embed in gene_embeddings.loc[pg].iterrows()])

dataset_ctrl_wembed = dataset_ctrl.add_column("perturbation_embedding", gene_embeddings_list)

# save
dataset_ctrl_wembed.save_to_disk(f'{base_path}/{data_path}/dataset/filtered_tokenised_hvg_pairing_GFpert_control.dataset')

# subset adata
adata_ctrl = subset_adata(adata, cell_pairings['control'])
adata_perturbed = subset_adata(adata, cell_pairings['perturbed'])

# save
adata_ctrl.write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_hvg_pairing_control.h5ad')
adata_perturbed.write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_hvg_pairing_perturbed.h5ad')

