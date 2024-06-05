#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 32 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/tokenisation_%J.out # output file
#BSUB -e logs/tokenisation_%J.err # error file
#BSUB -M 100000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>100000] rusage[mem=100000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation # job name

# activate python environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

echo '--- Start tokenisation'

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path './data/20240423_eb/EB.h5ad' \
--dataset eb \
--gene_filtering_mode hvg \
--var_list Time_point \
--pairing_mode random \
--nproc 32 \
--reference_time 'Day 00-03' \
--time_point_order 'Day 00-03' 'Day 06-09' 'Day 12-15' 'Day 18-21' 'Day 24-27' \
--exclude_non_GF_genes False

echo '--- Finished tokenisation'
