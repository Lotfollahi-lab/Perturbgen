#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 4 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen # working directory
#BSUB -o logs/GF_tokenisation_hspc_%J.out # output file
#BSUB -e logs/GF_tokenisation_hspc_%J.err # error file
#BSUB -M 40000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40000] rusage[mem=40000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation_hspc # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path '/lustre/scratch126/cellgen/lotfollahi/kl11/data/hspc/TV6_2020_regressed_final.h5ad' \
--finetune_adata_path 'T_perturb/tokenized_data/hspc_pbmc_median_inter_tissue_all_tf_100M/h5ad_pairing_5000_hvg_src/intermediate.h5ad' \
--dataset hspc_pbmc_median_etv6  \
--var_list celltypes ETV6status orig.indent \
--pairing_mode mapping \
--time_obs 'celltypes' \
--gene_filtering_mode 'all' \
--cell_gene_filter True \
--remove_mito_ribo_genes True \
--gene_filtering_mode 'all' \
--nproc 4 \
--reference_time MEP \
--time_point_order MEP MkP_Mk \
--gene_median_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/median_trace_scenario3.pkl' \
--token_dict_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/tokenid_trace_scenario3.pkl' \
--gene_mapping_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/ensembl_mapping_dict_gc95M.pkl'
echo "--- Finished tokenisation"

# hspc_GF_26k_median
# --gene_median_path '/lustre/scratch126/cellgen/lotfollahi/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_median.pkl' \
# --token_dict_path '/lustre/scratch126/cellgen/lotfollahi/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_token.pkl' \
# --gene_mapping_path '/lustre/scratch126/cellgen/lotfollahi/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_gene_mapping.pkl'
