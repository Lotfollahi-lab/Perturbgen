import pickle
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import tqdm
from scipy.sparse import csr_matrix


def map_ensembl_to_genename(
    adata: ad.AnnData,
    mapping_path: Path,
) -> ad.AnnData:
    """
    Description:
    ------------
    This function maps ensembl ids to gene names.
    """
    mapping_path = Path(mapping_path)
    assert mapping_path.exists(), '.csv mapping file does not exist'
    # read in .csv file to map ensembl ids to gene names
    mapping_df = pd.read_csv(mapping_path)
    # rename column gene_ids to ensembl_id
    mapping_df = mapping_df.rename(columns={'gene_ids': 'ensembl_id'})
    # left join adata.var with mapping_df to add ensembl ids to adata.var
    adata.var['gene_name'] = adata.var_names
    adata.var = adata.var.merge(
        mapping_df[['index', 'ensembl_id']],
        left_on='gene_name',
        right_on='index',
        how='left',
    )
    # create ensembl_id column and drop index and ensembl_id columns
    adata.var_names = adata.var['ensembl_id']
    adata.var = adata.var.drop(columns=['index', 'ensembl_id'])

    return adata


def map_deg_to_tokenid(adata_deg: ad.AnnData, token_id_path: Path):
    with open(token_id_path, 'rb') as f:
        token_id_dict = pickle.load(f)
    adata_deg.var['token_id'] = adata_deg.var_names.map(token_id_dict)
    adata_deg.var['token_id'] = adata_deg.var['token_id'].astype('Int64')
    adata_deg_df = adata_deg[:, adata_deg.var['token_id'].notna()].var
    adata_deg_subset = adata_deg[:, adata_deg.var['token_id'].notna()].copy()
    # enumerate token_id based on row index
    adata_deg_df.index = np.arange(0, len(adata_deg_df)) + 1
    token_id_dict = dict(zip(adata_deg_df['token_id'], adata_deg_df.index))
    token_id_dict[0] = 0
    return token_id_dict, adata_deg_subset


def subset_adata(adata, cell_pairings):
    adata_ = adata.copy()
    # check if obs index is not range index
    if adata_.obs.index.dtype != 'int64':
        adata_.obs = adata_.obs.reset_index()
    df = pd.DataFrame(adata_.X.A, index=adata_.obs.index, columns=adata_.var.index)
    # use row index instead of index
    df.reset_index(drop=True, inplace=True)
    subset_df = df.loc[cell_pairings]
    obs = adata_.obs.loc[cell_pairings]
    obs.index = obs['level_0']
    var = adata_.var.loc[df.columns]
    adata_subsetted = ad.AnnData(X=subset_df.values, obs=obs, var=var)
    adata_subsetted.obs_names.name = None
    adata_subsetted.X = csr_matrix(adata_subsetted.X)
    return adata_subsetted


def map_token_id_to_genename(adata_subset):
    """
    Description:
    ------------
    This function maps subset_token_id to gene_name and saves the dictionary as pickle.

    Parameters:
    -----------
    adata_subset: `~anndata.AnnData`
        Annotated data matrix subsetted to include only DEGs.
    Returns:
    --------
    adata_subset: `~anndata.AnnData`
        Annotated data matrix with subset_token_id and gene_name.
    """
    # create dictionary to map subset_token_id to gene_name
    adata_subset.var['subset_token_id'] = np.arange(adata_subset.n_vars) + 1
    # save dictionary as pickle
    subset_tokenid_to_deg = dict(
        zip(adata_subset.var['subset_token_id'], adata_subset.var['gene_name'])
    )
    with open(
        '/lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/'
        'T_perturb/T_perturb/pp/res/dataset/token_dictionary_for_subset_token_id.pkl',
        'wb',
    ) as f:
        pickle.dump(subset_tokenid_to_deg, f)
    return adata_subset


def pairing_resting_to_activated_cells(
    adata_subset: sc.AnnData, pairing_mode: str, seed: int = 42
):
    """
    Description:
    ------------
    This function pairs resting cells to activated cells based on time point.

    Parameters:
    -----------
    adata_subset: `~anndata.AnnData`
        Annotated data matrix subsetted to include only DEGs.
    pairing_mode: `str`
        Mode to pair cells. Choose between 'random' and 'stratified'.
    seed: `int`
        Seed for random number generator.

    Returns:
    --------
    cell_pairings: `dict`
        Dictionary containing pairing indices of resting and activated cells.
    """
    np.random.seed(seed)
    # replace index by row number
    adata_subset_ = adata_subset.copy()
    adata_subset_.obs = adata_subset_.obs.reset_index()

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
        dropped_donors = (
            adata_subset.obs['Donor'].nunique() - adata_grouped['Donor'].nunique()
        )
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
    return cell_pairings


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def label_encoder(adata, encoder, condition_key=None):
    """
    Description:
    ------------
    Encode labels of Annotated `adata` matrix.

    Parameters:
    ----------
    adata: : `~anndata.AnnData`
         Annotated data matrix.
    encoder: Dict
         dictionary of encoded labels.
    condition_key: String
         column name of conditions in `adata.obs` data frame.

    Returns:
    -------
    labels: `~numpy.ndarray`
         Array of encoded labels
    label_encoder: Dict
         dictionary with labels and encoded labels as key, value pairs.
    """
    unique_conditions = list(np.unique(adata.obs[condition_key]))
    labels = np.zeros(adata.shape[0])

    if not set(unique_conditions).issubset(set(encoder.keys())):
        missing_labels = set(unique_conditions).difference(set(encoder.keys()))
        print(
            f'Warning: Labels in adata.obs[{condition_key}]'
            'is not a subset of label-encoder!'
        )
        print(f'The missing labels are: {missing_labels}')
        print('Therefore integer value of those labels is set to -1')
        for data_cond in unique_conditions:
            if data_cond not in encoder.keys():
                labels[adata.obs[condition_key] == data_cond] = -1

    for condition, label in encoder.items():
        labels[adata.obs[condition_key] == condition] = label
    labels = [int(x) for x in labels]
    return labels
