#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-cellgeni-a100)
#BSUB -gpu 'mode=exclusive_process:num=4' # request for exclusive access to gpu
#BSUB -n 8 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team298/dv8/trace_paper/trace_final/T_perturb # working directory
#BSUB -o T_perturb/log/sweep/%J_lps_sweep.out # output file
#BSUB -e T_perturb/log/sweep/%J_lps_sweep.err # error file
#BSUB -M 80000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>80000] rusage[mem=80000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_sweep # job name

# load cuda
module load cuda-12.1.1

# activate environment
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run sweep
echo "--- Start sweep"
#paste wandb with sweep id
wandb agent lotfollahi/ttransformer_sweep/etkxxptb
echo "--- Finished sweep"
