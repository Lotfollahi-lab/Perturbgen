#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 8 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister # working directory
#BSUB -o logs/tokenisation_%J.out # output file
#BSUB -e logs/tokenisation_%J.err # error file
#BSUB -M 15000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>15000] rusage[mem=15000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation_eb # job name

# activate python environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

echo '--- Start tokenisation'

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path './data/20240423_eb/EB.h5ad' \
--dataset eb_pbmc_median \
--gene_filtering_mode hvg \
--var_list Time_point \
--pairing_mode random \
--pairing_obs Time_point \
--nproc 8 \
--reference_time 'Day 00-03' \
--time_point_order 'Day 00-03' 'Day 06-09' 'Day 12-15' 'Day 18-21' 'Day 24-27' \
--n_hvg 2000 \
--gene_median_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/filtered_trace_median.pkl' \
--token_dict_path '/lustre/scratch126/cellgen/team361/am74/Adib/TRACE/Loom_cohort/tdigest/2nd_run/Dictionaries/trace_filtered_tokenid.pkl' \

echo '--- Finished tokenisation'
