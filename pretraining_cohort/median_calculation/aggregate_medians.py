import os
import pickle
import math
import crick.tdigest

def merge_digest(dict_key_ensembl_id, dict_value_tdigest, new_tdigest_dict):
    """Merge new tdigest into the existing one."""
    new_gene_tdigest = new_tdigest_dict.get(dict_key_ensembl_id)
    if new_gene_tdigest is not None:
        dict_value_tdigest.merge(new_gene_tdigest)
    return dict_value_tdigest

def initialize_tdigests(all_genes):
    """Initialize a dictionary of TDigest objects for a gene list."""
    return {gene: crick.tdigest.TDigest() for gene in all_genes}

def merge_all_tdigests(rootdir):
    """Merge all TDigest objects from pickled files in the specified directory."""
    total_tdigest_dict = {}
    
    for subdir, _, files in os.walk(rootdir):
        for file in files:
            if file.endswith(".gene_median_digest_dict.pickle"):
                file_path = os.path.join(subdir, file)
                with open(file_path, "rb") as fp:
                    tdigest_dict = pickle.load(fp)
                
                for k in tdigest_dict.keys():
                    if k not in total_tdigest_dict:
                        total_tdigest_dict[k] = crick.tdigest.TDigest()

                # Merge tdigests
                total_tdigest_dict = {
                    k: merge_digest(k, v, tdigest_dict)
                    for k, v in total_tdigest_dict.items()
                }
    return total_tdigest_dict

def save_dict_as_pickle(data_dict, output_path):
    """Save a dictionary to a pickle file."""
    with open(output_path, "wb") as fp:
        pickle.dump(data_dict, fp)

def main(rootdir, output_dir):
    print(f"Processing .pkl files in: {rootdir}")
    total_tdigest_dict = merge_all_tdigests(rootdir)

    total_tdigest_path = os.path.join(output_dir, "total_gene_tdigest_dict.pickle")
    save_dict_as_pickle(total_tdigest_dict, total_tdigest_path)

    total_median_dict = {k: v.quantile(0.5) for k, v in total_tdigest_dict.items()}
    total_median_path = os.path.join(output_dir, "total_gene_median_dict.pickle")
    save_dict_as_pickle(total_median_dict, total_median_path)

    detected_median_dict = {k: v for k, v in total_median_dict.items() if not math.isnan(v)}
    detected_median_path = os.path.join(output_dir, "detected_gene_median_dict.pickle")
    save_dict_as_pickle(detected_median_dict, detected_median_path)

    print("Processing completed. Results saved in:", output_dir)

rootdir = "/nfs/team361/am74/Cytomeister/outputs/median_100m"  #
output_dir = "/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/" 

if __name__ == "__main__":
    main(rootdir, output_dir)
