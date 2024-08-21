#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 32 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team205/bair/T_perturb/T_perturb # working directory
#BSUB -o logs/masking_%J.out # output file
#BSUB -e logs/masking_%J.err # error file
#BSUB -M 1500000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>1500000] rusage[mem=1500000]' # RAM memory part 1. Default: 100MB
#BSUB -J GF_tokenisation # job name

# activate python environment
# source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd="/lustre/scratch126/cellgen/team205/bair/T_perturb/T_perturb"

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--dataset 'cytoimmgen' \
--gene_filtering_mode 'hvg' \
--var_list Cell_population Cell_type Time_point Age\
Sex batch Cell_culture_batch Phase\
Donor cell_pairing_index \
--pairing_mode stratified \
--nproc 32 \
