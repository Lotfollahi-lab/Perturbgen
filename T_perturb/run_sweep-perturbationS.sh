#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=4' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb/ # working directory
#BSUB -o ../../logs/sweep_%J.out # output file
#BSUB -e ../../logs/sweep_%J.err # error file
#BSUB -M 70GB  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>70GB] rusage[mem=70GB]' # RAM memory part 1. Default: 100MB
#BSUB -J countSweep # job name

source ~/.bashrc
conda activate Tperturb
cwd=$(pwd)
module load cuda-12.1.1
export WANDB_DIR=$cwd/wandb
export WANDB_ARTIFACT_LOCATION=$cwd/wandb/artifacts
export WANDB_ARTIFACT_DIR=$cwd/wandb/artifacts
export WANDB_CACHE_DIR=$cwd/wandb/artifacts
export WANDB_CONFIG_DIR=$cwd/wandb
export WANDB_DATA_DIR=$cwd/wandb
wandb agent irene-bp/ttransformer/rc3ci4e6
