#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=shared:num=1" # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb # working directory
#BSUB -o T_perturb/log/stratified_pairing_GF_tokenisation_LPS_100m%J.out # output file
#BSUB -e T_perturb/log/stratified_pairing_GF_tokenisation_LPS_100m%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
cwd=$(pwd)

echo '--- Start tokenisation'

python3 /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/T_perturb/pp/GF_tokenisation.py \
--h5ad_path '/nfs/team361/am74/Cytomeister/Evaluation_datasets/LPS/full_lps_new.h5ad' \
--dataset OURmed100m_LPS_all_tps_2k_hvg_noNormal \
--gene_filtering_mode hvg \
--var_list cell_pairing_index time_after_LPS cell_type_harmonized \
--gene_median_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/median_trace_subsetgeneformertokenid.pkl' \
--token_dict_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/tokenid_trace_subsetfeneformer.pkl' \
--gene_mapping_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/ensembl_mapping_dict_gc95M.pkl' \
--main_pairing_obs 'cell_type_harmonized' \
--pairing_mode stratified \
--time_obs time_after_LPS \
--reference_time '90m_LPS' \
--time_point_order '90m_LPS' '6h_LPS' '10h_LPS' \
--nproc 8 \
--n_hvg 2000

echo '--- Finished tokenisation'


# --pairing_mode mapping \
# --pairing_file '/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/skin_organoid_cell_pairings_clean.csv' \
