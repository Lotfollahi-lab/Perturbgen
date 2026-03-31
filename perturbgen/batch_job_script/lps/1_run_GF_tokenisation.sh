#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen # working directory
#BSUB -o logs/GF_tokenisation_lps_%J.out # output file
#BSUB -e logs/GF_tokenisation_lps_%J.err # error file
#BSUB -M 40000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40000] rusage[mem=40000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation_lps # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

echo '--- Start tokenisation'

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path '/nfs/team361/cytomeister/data/lps/full_lps.h5ad' \
--dataset lps_90min_perturb \
--gene_filtering_mode hvg \
--cell_gene_filter False \
--remove_mito_ribo_genes False \
--exclude_non_GF_genes True \
--hvg_mode 'before_tokenisation' \
--var_list cell_pairing_index time_after_LPS cell_type_harmonized \
--pairing_mode stratified \
--main_pairing_obs 'cell_type_harmonized' \
--time_obs time_after_LPS \
--reference_time '90m_LPS' \
--time_point_order '90m_LPS' '6h_LPS' '10h_LPS' \
--nproc 4 \
--n_hvg 2000 \
--gene_median_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/median_trace_scenario3.pkl' \
--token_dict_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/tokenid_trace_scenario3.pkl' \
--gene_mapping_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/ensembl_mapping_dict_gc95M.pkl'

echo '--- Finished tokenisation'
