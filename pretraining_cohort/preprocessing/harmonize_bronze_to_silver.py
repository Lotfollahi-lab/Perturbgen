import os
import sys
import gc
import scanpy as sc
import numpy as np
import pandas as pd
import pickle
import argparse
import yaml
from scipy.stats import median_abs_deviation


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_pickle(file_path, name):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded {name} from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)


def perform_qc(adata, output_dir, filename, config):
    # ------------------- Raw counts check -------------------
    if adata.X.max() < 50:
        try:
            adata.X = adata.raw.X
            print(f"{filename}: Set to raw.X")
        except AttributeError:
            try:
                adata.X = adata.layers["counts"]
                print(f"{filename}: Set to layers['counts']")
            except KeyError:
                print(f"{filename}: No raw or counts layer found")
    else:
        print(f"{filename}: Already raw")

    # Add ensembl_id column if not present
    if 'ensembl_id' not in adata.var.columns:
        adata.var['ensembl_id'] = adata.var['feature_id']  # var_names are Ensembl IDs
    
    # ------------------- Gene Filtering -------------------
    if "gene_filtering" in config:
        # Load non-coding genes to remove (Ensembl IDs)
        remove_ensembl_txt = config["gene_filtering"]["remove_ensembl_txt"]
        remove_df = pd.read_csv(remove_ensembl_txt)
        remove_ensembl_set = set(remove_df["Gene stable ID"].dropna().unique())
        
        # Load gene dictionary (gene names → Ensembl IDs)
        pickle_file = config["gene_filtering"]["ensembl_mapping_dict"]
        with open(pickle_file, "rb") as f:
            gene_dict = pickle.load(f)
        
        # Filter gene dictionary (remove LINC and MIR genes)
        filtered_gene_dict = {
            k: v for k, v in gene_dict.items() 
            if not (k.startswith("LINC") or k.startswith("MIR"))
        }
        # Get unique Ensembl IDs from filtered genes
        filtered_gene_names = set(filtered_gene_dict.keys())
        
        # Remove unwanted Ensembl IDs (e.g., non-coding, low-quality genes)
        mask_remove = ~adata.var['ensembl_id'].isin(remove_ensembl_set)
        adata = adata[:, mask_remove].copy()
        print(f"{filename}: Genes after removing unwanted Ensembl IDs: {adata.n_vars}")

        # Keep only genes in filtered_ensembl_ids (protein-coding, non-LINC, non-MIR)
        mask_keep = adata.var['feature_name'].isin(filtered_gene_names)
        adata = adata[:, mask_keep].copy()
        print(f"{filename}: Genes after keeping mapped protein-coding genes: {adata.n_vars}")

    # ------------------- Quality control -------------------
    # Use feature_name for QC metrics
    if 'feature_name' in adata.var.columns:
        # Mitochondrial genes
        adata.var["mt"] = adata.var['feature_name'].str.startswith("MT-", na=False)
        
        # Ribosomal genes
        adata.var["ribo"] = adata.var['feature_name'].str.startswith(("RPS", "RPL"), na=False)
        
        # Hemoglobin genes
        adata.var["hb"] = adata.var['feature_name'].str.contains("^HB[^(P)]", regex=True, na=False)
        
        print(f"{filename}: Using feature_name for QC metrics")
    else:
        print(f"{filename}: Warning - feature_name not found in var, skipping mt/ribo/hb detection")
        adata.var["mt"] = False
        adata.var["ribo"] = False
        adata.var["hb"] = False

    # Compute QC metrics
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], inplace=True, percent_top=[20], log1p=True
    )

    # Outlier detection function
    def is_outlier(adata, metric: str, nmads: int):
        M = adata.obs[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
            M > np.median(M) + nmads * median_abs_deviation(M)
        )
        return outlier

    # Flag general outliers
    adata.obs["outlier"] = (
        is_outlier(adata, "log1p_total_counts", 5)
        | is_outlier(adata, "log1p_n_genes_by_counts", 5)
        | is_outlier(adata, "pct_counts_in_top_20_genes", 5)
    )

    # Flag mitochondrial outliers
    adata.obs["mt_outlier"] = is_outlier(adata, "pct_counts_mt", 4) | (
        adata.obs["pct_counts_mt"] > 20
    )

    # Show counts
    print(adata.obs["outlier"].value_counts(dropna=False))
    print(adata.obs["mt_outlier"].value_counts(dropna=False))
    print(f"Total number of cells before filtering: {adata.n_obs}")

    # Filter low-quality cells
    adata = adata[(~adata.obs.outlier) & (~adata.obs.mt_outlier)].copy()
    print(f"Number of cells after filtering low-quality cells: {adata.n_obs}")

    sc.pp.filter_genes(adata, min_cells=5)
    print(f'{filename}: Number of genes after filtering low-expressed: {adata.n_vars}')

    # Add n_counts to obs
    adata.obs['n_counts'] = (
        adata.X.sum(axis=1).A1 if hasattr(adata.X, "A1") else adata.X.sum(axis=1)
    )
    
    output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_qc.h5ad")
    adata.write_h5ad(output_filepath)
    print(f"{filename}: QC completed. Saved to {output_filepath}.")


def harmonize_dataset(config):
    # Load all mapping dictionaries
    dev_stage_map = load_pickle(config["dev_stage_map"], "dev_stage_map")
    tissue_map = load_pickle(config["tissue_map"], "tissue_map")
    disease_map = load_pickle(config["disease_map"], "disease_map")

    input_dir = config["input_dir"]
    output_base_dir = config["output_base_dir"]
    excluded_file = config.get("excluded_file", None)
    do_exclude = config.get("do_exclude_file", False)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".h5ad"):
            continue

        # Skip the excluded file completely if do_exclude is True
        if do_exclude and filename == excluded_file:
            print(f"Skipping excluded file: {filename}")
            continue

        filepath = os.path.join(input_dir, filename)
        print(f"Processing: {filename}")
        adata = sc.read_h5ad(filepath)

        # Harmonize metadata
        # Development stage
        dev_stage_series = adata.obs["development_stage"].astype(str)
        adata.obs["trace_dev_stage"] = dev_stage_series.map(dev_stage_map).fillna(dev_stage_series)
        
        # Tissue
        tissue_series = adata.obs["tissue"].astype(str)
        adata.obs["tissue"] = tissue_series.map(tissue_map).fillna(tissue_series)
        
        disease_col = None
        for col in ["disease"]:
            if col in adata.obs.columns:
                disease_col = col
                break
                
        if disease_col:
            disease_series = adata.obs[disease_col].astype(str)
            adata.obs["trace_disease"] = disease_series.map(disease_map).fillna(disease_series)
            print(f"{filename}: Mapped disease terms using {disease_col}")
        else:
            print(f"{filename}: Warning - no disease column found")
            adata.obs["trace_disease"] = "unknown"

        # Add study identifier
        adata.obs["study"] = os.path.splitext(filename)[0]

        output_dir = os.path.join(output_base_dir, os.path.splitext(filename)[0])
        os.makedirs(output_dir, exist_ok=True)

        perform_qc(adata, output_dir, filename, config)

        del adata
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Harmonize bronze cellxgene files to silver tier.")
    parser.add_argument('--config_file', type=str, required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config_file)
    harmonize_dataset(config)