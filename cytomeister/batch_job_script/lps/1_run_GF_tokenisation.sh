#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=shared:num=1" # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/log/random_pairing_GF_tokenisation_%J.out # output file
#BSUB -e T_perturb/log/random_pairing_GF_tokenisation_%J.err # error file
#BSUB -M 40000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>40000] rusage[mem=40000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
cwd=$(pwd)

echo '--- Start tokenisation'

python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/GF_tokenisation.py \
--h5ad_path '/lustre/scratch126/cellgen/team298/dv8/trace_paper/lps_data/concatenated_lps_data_dv.h5ad' \
--dataset 2k_hvg_ourMED_all_tps_butNormal \
--gene_filtering_mode hvg \
--var_list cell_type_cellgen_harm donor_cellgen_harm time_after_LPS cell_pairing_index  \
--pairing_mode stratified \
--pairing_obs time_after_LPS \
--nproc 8 \
--n_hvg 2000 \
--gene_median_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/filtered_trace_median.pkl' \
--token_dict_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_filtered_tokenid.pkl' \

echo '--- Finished tokenisation'
