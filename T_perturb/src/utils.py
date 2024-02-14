import pickle
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


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
