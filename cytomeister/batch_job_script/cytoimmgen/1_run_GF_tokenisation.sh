#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/GF_tokenisation_%J.out # output file
#BSUB -e logs/GF_tokenisation_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path 'data/cytoimmgen/cytoimmgen_qc.h5ad' \
--dataset 'cytoimmgen_100M_cellpopulation' \
--gene_filtering_mode 'hvg' \
--time_obs 'Time_point' \
--var_list Cell_population Cell_type Time_point Age\
 Sex batch Cell_culture_batch Phase\
 Donor cell_pairing_index \
--pairing_mode stratified \
--main_pairing_obs 'Cell_type' \
--remove_mito_ribo_genes False \
--cell_gene_filter False \
--nproc 4 \
--reference_time '0h' \
--time_point_order '0h' '16h' '40h' '5d' \
--n_hvg 2000 \
--hvg_mode 'before_tokenisation' \
--exclude_non_GF_genes False \
--gene_median_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/median_trace_scenario3.pkl' \
--token_dict_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/tokenid_trace_scenario3.pkl' \
--gene_mapping_path '/nfs/team361/am74/Cytomeister/outputs/median/aggregate/scenario_3/ensembl_mapping_dict_gc95M.pkl'

echo '--- Finished tokenisation'
