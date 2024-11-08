#make a date directory if it does not exist
#!/bin/bash
#BSUB -q normal # run CPU job
#BSUB -n 32 # number of cores
#BSUB -G team298 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/T_perturb/ 
#BSUB -o T_perturb/logs_n/random_pairing_GF_tokenisation_%J.out # output file
#BSUB -e T_perturb/logs_n/random_pairing_GF_tokenisation_%J.err # error file
#BSUB -M 150000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>150000] rusage[mem=150000]' # RAM memory part 1. Default: 100MB
#BSUB -J random_pairing_GF_tokenisation # job name

# activate python environment
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

echo "--- Start tokenisation"

python3 $cwd/pp/GF_tokenisation.py \
--dataset 'cytoimmgen' \
--gene_filtering_mode 'hvg' \
--var_list Cell_population Cell_type Time_point Age\
Sex batch Cell_culture_batch Phase\
Donor cell_pairing_index \
--pairing_mode random \
--nproc 32 \
