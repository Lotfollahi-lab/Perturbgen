#!/bin/bash
#BSUB -q normal # CPU job
#BSUB -n 1 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/perturbgen # working directory
#BSUB -o logs/test_transformer_%J.out # output file
#BSUB -e logs/test_transformer_%J.err # error file
#BSUB -M 1000  # RAM memory part 2. Default: 100MB
#BSUB -R "select[mem>1000] rusage[mem=1000]" # RAM memory part 1. Default: 100MB
#BSUB -J test_transformer # job name

# load cuda
module load cuda-12.1.1

# activate environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start testing CellGen training"
python -m unittest discover /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen/tests/
# python -m unittest perturbgen.tests.test_cellgen_training
echo "Testing CellGen training finished ---"
