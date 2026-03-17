import scanpy as sc
import os
import gc
import anndata as ad
import crick
import pickle
import numpy as np
from tqdm import tqdm
from scipy import sparse

# Directories
input_dir = "/nfs/team361/am74/Cytomeister/pretrain_cohort_version_2/arc_institute_processed_harmonized_"
output_dir = "/nfs/team361/am74/Cytomeister/outputs/median_arc"
os.makedirs(output_dir, exist_ok=True)

for root, dirs, files in os.walk(input_dir):
    for input_file in files:
        if input_file.endswith(".h5ad"):
            print(f"Processing file: {input_file}")
            file_path = os.path.join(root, input_file)

            try:
                adata = sc.read_h5ad(file_path)
                var_names = adata.var['ensembl_id']
                n_genes = len(var_names)

                # Extract count matrix
                X = adata.X
                if sparse.issparse(X):
                    X = X.toarray()

                # Normalize
                cell_counts = adata.obs["n_counts"].values
                norm_X = (X.T / cell_counts).T * 10_000  

                # Prepare tdigests
                median_digest_dict = {}
                progress = tqdm(range(n_genes), desc=f"Processing {input_file}")
                
                for idx in progress:
                    gene_data = norm_X[:, idx].astype(np.float32)
                    nonzero = gene_data[gene_data > 0]
                    digest = crick.tdigest.TDigest()
                    digest.update(nonzero)
                    median_digest_dict[var_names[idx]] = digest

                # Save output
                output_file = input_file.replace(".h5ad", ".gene_median_digest_dict.pickle")
                output_path = os.path.join(output_dir, output_file)
                with open(output_path, "wb") as fp:
                    pickle.dump(median_digest_dict, fp)

                print(f"Saved to: {output_path}")

            except Exception as e:
                print(f"Error processing {input_file}: {e}")

            finally:
                del adata, X, norm_X
                gc.collect()

print("Processing complete.")
