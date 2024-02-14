import os
from pathlib import Path
from typing import Dict, List

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import tqdm
from datasets import load_from_disk

from T_perturb.src.utils import map_deg_to_tokenid, subset_adata

seed_no = 42
np.random.seed(seed_no)
if os.getcwd().split('/')[-3] != 'T_perturb':
    os.chdir(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/'
        't_generative/T_perturb/T_perturb/pp'
    )
    print('Changed working directory to root of repository')
dataset = load_from_disk('./res/dataset/cytoimmgen_tokenised_degs.dataset')
adata = sc.read_h5ad('./res/h5ad_data/cytoimmgen_tokenisation_degs.h5ad')
deg_to_tokenid_dict, adata_subset = map_deg_to_tokenid(
    adata,
    Path(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'generative_modelling_omic/Geneformer/geneformer/token_dictionary.pkl'
    ),
)


# use dictionary to map token_id to input_ids
def map_input_ids(dataset):
    dataset['input_ids'] = [
        deg_to_tokenid_dict.get(item, item) for item in dataset['input_ids']
    ]
    return dataset


dataset = dataset.map(map_input_ids)

# replace index by row number
adata_subset_ = adata_subset.copy()
adata_subset_.obs = adata_subset_.obs.reset_index()
metadata_df = pd.DataFrame(
    {
        'Donor': dataset['Donor'],
        'Cell_type': dataset['Cell_type'],
        'Time_point': dataset['Time_point'],
    }
)
pairing_mode = 'stratified'  # choose between 'random' and 'stratified'
# find index for each time point
adata_0h_ = adata_subset_.obs.loc[adata_subset_.obs['Time_point'] == '0h', :]
adata_16h_ = adata_subset_.obs.loc[adata_subset_.obs['Time_point'] == '16h', :]
adata_40h_ = adata_subset_.obs.loc[adata_subset_.obs['Time_point'] == '40h', :]
adata_5d_ = adata_subset_.obs.loc[adata_subset_.obs['Time_point'] == '5d', :]
# initiate dictionary to store cell pairings
cell_pairings: Dict[str, List[int]] = {'0h': [], '16h': [], '40h': [], '5d': []}
if pairing_mode == 'stratified':
    # drop Donor if they do not have Cell_type, Donor in all the Time_points
    adata_grouped = adata_subset_.obs[
        adata_subset_.obs.groupby(['Donor', 'Cell_type'])['Time_point'].transform(
            'nunique'
        )
        == 4
    ]
    dropped_donors = adata.obs['Donor'].nunique() - adata_grouped['Donor'].nunique()
    print(f'dropped {dropped_donors} donors')
    resting_cells = adata_grouped.loc[adata_grouped['Time_point'] == '0h', :]
    grouped = adata_grouped.groupby(['Donor', 'Cell_type'])
    for idx, resting in tqdm.tqdm(
        resting_cells.iterrows(), total=resting_cells.shape[0]
    ):
        # get the indices of the other time points for the same cell type and donor
        group = grouped.get_group((resting['Donor'], resting['Cell_type']))
        indices_16h = group[group['Time_point'] == '16h'].index
        indices_40h = group[group['Time_point'] == '40h'].index
        indices_5d = group[group['Time_point'] == '5d'].index
        cell_pairings['0h'].append(idx)
        cell_pairings['16h'].append(np.random.choice(indices_16h))
        cell_pairings['40h'].append(np.random.choice(indices_40h))
        cell_pairings['5d'].append(np.random.choice(indices_5d))

elif pairing_mode == 'random':
    # randomly sample from each time point
    for idx, row in tqdm.tqdm(adata_0h_.iterrows(), total=adata_0h_.shape[0]):
        cell_pairings['0h'].append(idx)
        cell_pairings['16h'].append(np.random.choice(adata_16h_.index))
        cell_pairings['40h'].append(np.random.choice(adata_40h_.index))
        cell_pairings['5d'].append(np.random.choice(adata_5d_.index))
else:
    raise ValueError('pairing_mode must be either random or stratified')

# subset dataset to store separate dataset for each time point
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
# subset adata to store separate adata for each time point
# cell_pairings = {k: list(map(str, v)) for k, v in cell_pairings.items()}
# use index to subset adata

adata_0h = subset_adata(adata_subset_, cell_pairings['0h'])
adata_16h = subset_adata(adata_subset_, cell_pairings['16h'])
adata_40h = subset_adata(adata_subset_, cell_pairings['40h'])
adata_5d = subset_adata(adata_subset_, cell_pairings['5d'])
adata_0h.write_h5ad(
    f'./res/h5ad_data/cytoimmgen_tokenisation_degs_{pairing_mode}_0h.h5ad'
)
adata_16h.write_h5ad(
    f'./res/h5ad_data/cytoimmgen_tokenisation_degs_{pairing_mode}_16h.h5ad'
)
adata_40h.write_h5ad(
    f'./res/h5ad_data/cytoimmgen_tokenisation_degs_{pairing_mode}_40h.h5ad'
)
adata_5d.write_h5ad(
    f'./res/h5ad_data/cytoimmgen_tokenisation_degs_{pairing_mode}_5d.h5ad'
)

df = pd.DataFrame(
    adata_subset_.X.A, index=adata_subset_.obs.index, columns=adata_subset_.var.index
)
# use row index instead of index
df.reset_index(drop=True, inplace=True)
subset_df = df.loc[cell_pairings['16h']]
obs = adata_subset_.obs.loc[cell_pairings['16h']]
obs.index = obs['level_0']
var = adata.var.loc[df.columns]
subset_adata = anndata.AnnData(X=subset_df.values, obs=obs, var=var)
