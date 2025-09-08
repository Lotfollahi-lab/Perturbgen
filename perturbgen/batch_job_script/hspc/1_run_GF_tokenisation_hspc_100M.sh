#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 4 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
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
--h5ad_path '/lustre/scratch126/cellgen/lotfollahi/kl11/data/hspc/cd34.h5ad' \
--dataset hspc_pbmc_median_inter_tissue_all_tf_100M \
--var_list assignment_id sex tissue phase\
 celltype_v2 donor_tissue diff_state dataset\
 cell_pairing_index \
--pairing_mode mapping \
--time_obs 'diff_state' \
--pairing_file 'T_perturb/cytomeister/pp/hspc/cd34_pos_mapping.csv' \
--main_pairing_obs 'celltype_v2' \
--opt_pairing_obs 'tissue' \
--gene_filtering_mode 'hvg' \
--cell_gene_filter True \
--remove_mito_ribo_genes True \
--hvg_mode 'after_tokenisation' \
--n_hvg 5000 \
--nproc 4 \
--genes_to_include_path 'T_perturb/cytomeister/pp/hspc/1639_Human_TF.csv' \
--reference_time intermediate \
--time_point_order intermediate terminal \
--gene_median_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/median_trace_subsetgeneformertokenid.pkl' \
--token_dict_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/tokenid_trace_subsetfeneformer.pkl' \
--gene_mapping_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/ensembl_mapping_dict_gc95M.pkl'
echo "--- Finished tokenisation"

# hspc_GF_26k_median
# --gene_median_path '/lustre/scratch126/cellgen/lotfollahi/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_median.pkl' \
# --token_dict_path '/lustre/scratch126/cellgen/lotfollahi/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_token.pkl' \
# --gene_mapping_path '/lustre/scratch126/cellgen/lotfollahi/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_gene_mapping.pkl'
