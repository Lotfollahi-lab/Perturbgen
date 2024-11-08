#make a date directory if it does not exist
#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=shared:num=1" # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G team298 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/ 
#BSUB -o logs/random_pairing_GF_tokenisation_%J.out # output file
#BSUB -e logs/random_pairing_GF_tokenisation_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--h5ad_path '/lustre/scratch126/cellgen/team298/dv8/trace_paper/concatenated_lps_data.h5ad' \
--dataset 'lps' \
--gene_filtering_mode 'hvg' \
--var_list cell_type_cellgen_harm time_after_LPS donor_cellgen_harm\
Sex batch \
--pairing_mode stratified \
--nproc 32 \
