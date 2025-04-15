#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/GF_tokenisation_hspc_%J.out # output file
#BSUB -e logs/GF_tokenisation_hspc_%J.err # error file
#BSUB -M 40000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40000] rusage[mem=40000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation_hspc # job name

# activate python environment
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path './data/20241026_HSPC/cd34.h5ad' \
--dataset hspc_pbmc_median_inter_tissue_seuratv3_b \
--var_list assignment_id sex tissue phase\
 celltype_v2 donor_tissue diff_state dataset\
 cell_pairing_index \
--pairing_mode mapping \
--time_obs 'diff_state' \
--opt_pairing_obs 'tissue' \
--gene_filtering_mode 'hvg' \
--n_hvg 10000 \
--nproc 4 \
--reference_time intermediate \
--time_point_order intermediate terminal \
--gene_median_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/filtered_trace_median.pkl' \
--token_dict_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_filtered_tokenid.pkl' \
# --gene_mapping_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/filtered_trace_geneid.pkl'
echo "--- Finished tokenisation"

# hspc_GF_26k_median
# --gene_median_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_median.pkl' \
# --token_dict_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_token.pkl' \
# --gene_mapping_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_gene_mapping.pkl'
