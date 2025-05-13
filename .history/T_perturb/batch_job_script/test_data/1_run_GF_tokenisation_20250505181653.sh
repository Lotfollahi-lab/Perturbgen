#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=shared:num=1" # request for exclusive access to gpu
#BSUB -n 8 # number of cores
#BSUB -G team298 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/logs/valid_pairing_GF_tokenisation_%J.out # output file
#BSUB -e T_perturb/logs/valid_pairing_GF_tokenisation_%J.err # error file
#BSUB -M 80000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>80000] rusage[mem=80000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
cwd=$(pwd)

echo '--- Start tokenisation'

python3 /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb/T_perturb/pp/GF_tokenisation.py \
--h5ad_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/HSPC_eval/control_data_pair_obs_added.h5ad' \
--dataset validation_data_2kHVG_ourMED \
--gene_filtering_mode hvg \
--var_list annotation_simplified replicate cell_pairing_index pairing  \
--pairing_mode stratified \
--pairing_obs pairing \
--time_obs pairing \
--opt_pairing_obs None \
--gene_filtering_mode 'hvg' \
--remove_mito_ribo_genes False \
--reference_time "early" \
--time_point_order "early" "late" \
--nproc 8 \
--n_hvg 2000 \
--gene_median_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/filtered_trace_median.pkl' \
--token_dict_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_filtered_tokenid.pkl' \
> T_perturb/logs/valid_token_$(date +%Y%m%d_%H%M%S).out \
2> T_perturb/logs/valid_token_$(date +%Y%m%d_%H%M%S).err

echo '--- Finished tokenisation'

#source /lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/T_perturb/.venv/bin/activate