#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=shared:num=1" # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb # working directory
#BSUB -o cytomeister/log/stratified_pairing_GF_tokenisation_SKO_100m%J.out # output file
#BSUB -e cytomeister/log/stratified_pairing_GF_tokenisation_SKO_100m%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
cwd=$(pwd)

echo '--- Start tokenisation'

python3 /lustre/scratch126/cellgen/lotfollahi/dv8/trace/Perturbgen/perturbgen/pp/GF_tokenisation.py \
--h5ad_path '/nfs/team361/am74/Cytomeister/Evaluation_datasets/skin_organoid_muzzdevelopmentatlas/organoid_pp_cellmeister_addednewtmpsday6_UPDATEDENSEMBL.h5ad' \
--dataset OURmed100m_sko_all_tps_5k_genesIncluded_hvg_100Med_new_1 \
--gene_filtering_mode hvg \
--var_list day org_annot0 strain cell_pairing_index \
--gene_median_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/median_trace_subsetgeneformertokenid.pkl' \
--token_dict_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/tokenid_trace_subsetfeneformer.pkl' \
--gene_mapping_path '/nfs/team361/am74/Cytomeister/outputs/median_100m/aggregated_median/total1000_subsetgeneformertokenID_TRACE/ensembl_mapping_dict_gc95M.pkl' \
--genes_to_include_path '/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/genes_cytomeister_dv_ensembl.csv' \
--main_pairing_obs 'org_annot0' \
--pairing_mode mapping \
--pairing_file '/lustre/scratch126/cellgen/lotfollahi/dv8/trace/T_perturb/skin_organoid_cell_pairings_v2_cleaned.csv' \
--time_obs day \
--reference_time 'day-6' \
--time_point_order 'day-6' 'day-29' 'day-48' 'day-85' 'day-133' \
--nproc 8 \
--n_hvg 5000

echo '--- Finished tokenisation'
