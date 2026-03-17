import os
from geneformer import TranscriptomeTokenizer
from datasets import disable_caching

# Paths to required files
median_file = "/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/median_trace_subsetgeneformertokenid.pkl"
token_dict_file = "/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/tokenid_trace_subsetfeneformer.pkl"
gene_map_file = "/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/ensembl_mapping_dict_gc95M.pkl"

# Initialize tokenizer
tokenizer = TranscriptomeTokenizer(
    {"tissue": "tissue", "disease": "disease"},
    model_input_size=4096,
    gene_median_file=median_file,
    token_dictionary_file=token_dict_file,
    gene_mapping_file=gene_map_file,
    nproc=32
)
disable_caching()

base_input_dir = '/nfs/team361/am74/Cytomeister/pretrain_cohort_version_2/processed_harmonized_/'
output_dir = '/nfs/team361/am74/Cytomeister/outputs/tokenization_100m/'

# Traverse all subdirectories
for root, dirs, files in os.walk(base_input_dir):
    for file in files:
        if file.endswith(".h5ad"):
            h5ad_path = os.path.join(root, file)
            file_prefix = os.path.splitext(file)[0] 
            print(f"Tokenizing: {h5ad_path}")
            try:
                tokenizer.tokenize_data(
                    data_directory=root,
                    use_generator = True,
                    output_directory=output_dir,
                    output_prefix=file_prefix,
                    file_format='h5ad'
                )
            except Exception as e:
                print(f"Failed to tokenize {h5ad_path}: {e}")
            os.system("rm -rf ~/.cache/huggingface/datasets")
