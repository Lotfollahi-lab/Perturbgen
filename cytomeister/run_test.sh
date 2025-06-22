#!/bin/bash
#BSUB -q normal # CPU job
#BSUB -n 1 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/test_transformer_%J.out # output file
#BSUB -e logs/test_transformer_%J.err # error file
#BSUB -M 1000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>1000] rusage[mem=1000]" # RAM memory part 1. Default: 100MB
#BSUB -J test_transformer # job name

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start testing CellGen training"
python -m unittest discover /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/tests/
echo "Testing CellGen training finished ---"
