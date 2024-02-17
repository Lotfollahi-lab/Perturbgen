import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch


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
    df = pd.DataFrame(adata.X.A, index=adata.obs.index, columns=adata.var.index)
    # use row index instead of index
    df.reset_index(drop=True, inplace=True)
    subset_df = df.loc[cell_pairings]
    obs = adata.obs.loc[cell_pairings]
    obs.index = obs['level_0']
    var = adata.var.loc[df.columns]
    adata_subsetted = ad.AnnData(X=subset_df.values, obs=obs, var=var)
    return adata_subsetted


def one_hot_encoder(idx, n_cls):
    assert torch.max(idx).item() < n_cls
    if idx.dim() == 1:
        idx = idx.unsqueeze(1)
    onehot = torch.zeros(idx.size(0), n_cls)
    onehot = onehot.to(idx.device)
    onehot.scatter_(1, idx.long(), 1)
    return onehot


def label_encoder(adata, encoder, condition_key=None):
    """Encode labels of Annotated `adata` matrix.
    Parameters
    ----------
    adata: : `~anndata.AnnData`
         Annotated data matrix.
    encoder: Dict
         dictionary of encoded labels.
    condition_key: String
         column name of conditions in `adata.obs` data frame.

    Returns
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
