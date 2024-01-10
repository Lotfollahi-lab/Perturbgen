import pandas as pd
import scanpy as sc
from pathlib import Path
import anndata as ad

def map_ensembl_to_genename(
    adata: ad.AnnData,
    mapping_path: str,
    ) -> ad.AnnData:
    """
    Description:
    ------------
    This function maps ensembl ids to gene names.
    """
    mapping_path=Path(mapping_path)
    assert mapping_path.exists(), ".csv mapping file does not exist"
    #read in .csv file to map ensembl ids to gene names
    mapping_df=pd.read_csv(mapping_path)
    #rename column gene_ids to ensembl_id
    mapping_df=mapping_df.rename(columns={"gene_ids":"ensembl_id"})
    #left join adata.var with mapping_df to add ensembl ids to adata.var
    adata.var["gene_name"]=adata.var_names
    adata.var=adata.var.merge(mapping_df[["index","ensembl_id"]],left_on="gene_name",right_on="index",how="left")
    #create ensembl_id column and drop index and ensembl_id columns
    adata.var_names=adata.var["ensembl_id"]
    adata.var=adata.var.drop(columns=["index","ensembl_id"])
    
    return adata