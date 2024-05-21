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
from gears import PertData, GEARS
import itertools

# Set up -------------------
# paths
# base_path = '/lustre/groups/imm01/workspace/irene.bonafonte/Projects/2024Mar_Tperturb'
base_path = '/lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb'
data_path = 'datasets'
dataset_name = 'Norman2019'
pp_path = 'T_perturb/T_perturb/pp/res'
geneformer_path = f'{base_path}/../../Software/Geneformer'
# seed
seed_no = 2
np.random.seed(seed_no)
# arguments
gene_filtering_mode = 'gears'
subset_dataset=False

# download ----------------
adata = PertData('./data')
adata.load(data_name = 'norman')
adata = adata.adata

# pre-process ----------------
print('preprocess')
adata.X = adata.layers['counts']
adata.obs['perturbation_name'] = adata.obs.condition.str.replace('+ctrl','').str.replace('ctrl+','').str.replace('ctrl','control')
adata.obs['n_counts'] = adata.X.sum(axis=1)
adata.var['ensembl_id'] = adata.var_names

# add some annotations to the perturbations
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

# add token ids
tpath = f'{geneformer_path}/geneformer/token_dictionary.pkl'
with open(tpath, 'rb') as f:
    token_id_dict = pickle.load(f)
adata.var['token_id'] = adata.var_names.map(token_id_dict)
adata.var['token_id'] = adata.var['token_id'].astype('Int64')

# list of perturbed genes
perturbations = [p for p in set(itertools.chain.from_iterable(adata.obs.perturbation_name_l.to_list())) if p != 'control']
missing_perturbations = set([p for p in perturbations if not p in adata.var.gene_name[~adata.var.token_id.isna()].values])
print('The following perturbed genes cannot be tokenized with GF: ', missing_perturbations)
missing_mask = adata.obs.perturbation_name_l.apply(lambda x: len(missing_perturbations.intersection(x)) > 0)
adata = adata[~missing_mask]

# after cell filtering, add unique index to adata obs for cell pairing
adata.obs['cell_pairing_index'] = range(adata.shape[0])

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
adata_gears = adata.copy()

# compute DEG per perturbation
adata_gears = get_DE_genes(adata_gears, False)
adata_gears = get_dropout_non_zero_genes(adata_gears)

# change to our IDs and add to adata
gears2gene = adata_gears.obs[['condition_name','perturbation_name']].drop_duplicates().set_index('condition_name', inplace=False, drop=True).to_dict()['perturbation_name']
ens2idx = adata_gears.var['rowidx'].to_dict()

# add to adata
adata.uns['top_non_dropout_de_20'] = {gears2gene[p]: np.vectorize(ens2idx.__getitem__)(g) for p, g in adata_gears.uns['top_non_dropout_de_20'].items()}
adata.uns['top_non_zero_de_20'] = {gears2gene[p]: np.vectorize(ens2idx.__getitem__)(g) for p, g in adata_gears.uns['top_non_zero_de_20'].items()}
adata.uns['non_zeros_gene_idx'] = adata_gears.uns['non_zeros_gene_idx']
adata.uns['non_dropout_gene_idx'] = adata_gears.uns['non_dropout_gene_idx']
adata.uns['rank_genes_groups_cov_all'] = {gears2gene[p]: np.vectorize(ens2idx.__getitem__)(g) for p, g in adata_gears.uns['rank_genes_groups_cov_all'].items()}
del adata_gears
gc.collect()

print('saving adata')
adata.write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_{gene_filtering_mode}.h5ad')
# only tokenizable genes for GF (as src_cell)
adata[:,~adata.var.token_id.isna()].write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/gf_filtered_tokenised_{gene_filtering_mode}.h5ad')

# Tokenize ---------
print('tokenize')
var_list = [var for var in ['perturbation_name', 'n_perturbations','perturbation1_name', 'perturbation2_name', 'n_counts', 'cell_pairing_index'] if var in adata.obs.columns]
var_to_keep: Dict[str, str] = {v: v for v in var_list}.copy()
tk = TranscriptomeTokenizer(var_to_keep, nproc=8) #, model_input_size=5000)
tk.tokenize_data(
    f'{base_path}/{data_path}/adata',  # input directory - all h5ad files in this directory will be tokenised
    f'{base_path}/{data_path}/{dataset_name}/dataset',  # output directory - tokenised h5ad files will be saved here
    f'filtered_tokenised_{gene_filtering_mode}',  # name of output file
    file_format='h5ad',  # format [loom, h5ad]
)

# Pairing --------
print('pairing')
# load
dataset = load_from_disk(f'{base_path}/{data_path}/{dataset_name}/dataset/filtered_tokenised_{gene_filtering_mode}.dataset')
adata = sc.read_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_{gene_filtering_mode}.h5ad')
adata.obs = adata.obs.reset_index()

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

cell_pairings: Dict[str, List[int]] = {'control': [], 'perturbed': [], 'perturbed_gene': []}

# obs df for ctrl and perturbed
adata_ctrl = adata.obs.loc[adata.obs['perturbation_name'] == 'control', :]
adata_ctrl = adata_ctrl.reset_index(drop=True)
adata_perturbed = adata.obs.loc[adata.obs['perturbation_name'] != 'control', :]
ctrl_cells_idx = adata_ctrl.cell_pairing_index.values

# randomly sample from each time point (two samples: ctrl or perturbed -including all perturbations-, ctrl cells are repeated)
for idx, row in tqdm.tqdm(adata_perturbed.iterrows(), total=adata_perturbed.shape[0]):
    perturbed_genes = adata_perturbed.loc[idx, 'perturbation_name']
    # perturbation name
    perturbed_token = adata.uns['perturbations'].loc[perturbed_genes, 'token_id']
    perturbed_idx = adata.uns['perturbation_id'].loc[adata.uns['perturbation_id'].gene_name.isin(perturbed_genes.split('+')),'rowidx'].tolist()
    cell_pairings['perturbed_gene'].append(perturbed_idx) 
    # perturbed cell idx
    cell_pairings['perturbed'].append(idx)
    # choose resting within cells expressing the perturbed gene
    cell_pairings['control'].append(np.random.choice(ctrl_cells_idx)) 

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
dataset_perturbed.save_to_disk(f'{base_path}/{data_path}/{dataset_name}/dataset/filtered_tokenised_{gene_filtering_mode}_pairing_perturbed.dataset')

# add perturbed gene embedding to the control dataset - Binary -----------------
gene_embeddings_list = []
n_perturbations_list = []
for pg in tqdm.tqdm(cell_pairings['perturbed_gene']):
    binary_embed_one, binary_embed_two = np.zeros(adata.shape[1]), np.zeros(adata.shape[1])
    binary_embed_one[pg[0]] = 1
    if len(pg) == 1:
        pg = pg + [0]
        n_perturbations_list.append(np.array([False, True]))
    else:
        n_perturbations_list.append(np.array([False, False]))
        binary_embed_two[pg[1]] = 1
    gene_embeddings_list.append([binary_embed_one,binary_embed_two])

dataset_ctrl_wembed = dataset_ctrl.add_column("perturbation_embedding", gene_embeddings_list)
dataset_ctrl_wembed = dataset_ctrl_wembed.add_column("n_perturbations_bool", n_perturbations_list)
dataset_ctrl_wembed.save_to_disk(f'{base_path}/{data_path}/{dataset_name}/dataset/filtered_tokenised_{gene_filtering_mode}_pairing_binarypert_control.dataset')

# add perturbed gene embedding to the control dataset - Geneformer -----------------
gf = BertForMaskedLM.from_pretrained(
            f'{geneformer_path}',
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
n_perturbations_list = []
for pg in tqdm.tqdm(cell_pairings['perturbed_gene']):
    if len(pg) == 1:
        pg = pg + [0]
        n_perturbations_list.append(np.array([False, True]))
    else:
        n_perturbations_list.append(np.array([False, False]))
    gene_embeddings_list.append([embed.values for row, embed in gene_embeddings.loc[pg].iterrows()])

dataset_ctrl_wembed = dataset_ctrl.add_column("perturbation_embedding", gene_embeddings_list)
dataset_ctrl_wembed = dataset_ctrl_wembed.add_column("n_perturbations_bool", n_perturbations_list)

# save
dataset_ctrl_wembed.save_to_disk(f'{base_path}/{data_path}/{dataset_name}/dataset/filtered_tokenised_{gene_filtering_mode}_pairing_GFpert_control.dataset')

# add perturbed gene embedding to the control dataset - gene2vec -----------------
gene_embeddings = pd.read_csv('https://github.com/jingcheng-du/Gene2vec/raw/master/pre_trained_emb/gene2vec_dim_200_iter_9.txt', sep='\t', header=None, index_col=0)
gene_embeddings = pd.DataFrame({g: [float(dim) for dim in embeds[:-1]] for g, embeds in zip(gene_embeddings.index, gene_embeddings[1].str.split(' '))}).T
pad_embed = gene_embeddings.mean(axis=0) # mean embedding for padding (when it's only one perturbation)
gene_embeddings = gene_embeddings[gene_embeddings.index.isin(adata.uns['perturbation_id'].gene_name)]

# change gene name by our idxs
adata.uns['perturbation_id'].set_index('gene_name', inplace=True, drop=False)
gene_embeddings.index = adata.uns['perturbation_id'].loc[gene_embeddings.index,'rowidx'].values
gene_embeddings.loc[0,:] = pad_embed.values
gene_embeddings.head(3)

# store gene embedding of each perturbed gene 
gene_embeddings_list = []
n_perturbations_list = []
for pg in tqdm.tqdm(cell_pairings['perturbed_gene']):
    if len(pg) == 1:
        pg = pg + [0]
        n_perturbations_list.append(np.array([False, True]))
    else:
        n_perturbations_list.append(np.array([False, False]))
    gene_embeddings_list.append([embed.values for row, embed in gene_embeddings.loc[pg].iterrows()])

dataset_ctrl_wembed = dataset_ctrl.add_column("perturbation_embedding", gene_embeddings_list)
dataset_ctrl_wembed = dataset_ctrl_wembed.add_column("n_perturbations_bool", n_perturbations_list)
dataset_ctrl_wembed.save_to_disk(f'{base_path}/{data_path}/{dataset_name}/dataset/filtered_tokenised_{gene_filtering_mode}_pairing_gene2vecpert_control.dataset')

# save adata --------------
# subset adata
adata_ctrl = subset_adata(adata, cell_pairings['control'])
adata_perturbed = subset_adata(adata, cell_pairings['perturbed'])

print(adata_ctrl.shape, adata_perturbed.shape)
# save
adata_ctrl.write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_{gene_filtering_mode}_pairing_control.h5ad')
adata_perturbed.write_h5ad(f'{base_path}/{data_path}/{dataset_name}/adata/filtered_tokenised_{gene_filtering_mode}_pairing_perturbed.h5ad')




