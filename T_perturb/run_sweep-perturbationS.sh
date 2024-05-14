#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=3' # request for exclusive access to gpu
#BSUB -n 32 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/ip14/Projects/2024Mar_Tperturb/T_perturb/T_perturb/ # working directory
#BSUB -o ../../logs/sweep_%J.out # output file
#BSUB -e ../../logs/sweep_%J.err # error file
#BSUB -M 64GB  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>64GB] rusage[mem=64GB]' # RAM memory part 1. Default: 100MB
#BSUB -J 4.1_train_mask_sweep_Norman2019 # job name

source ~/.bashrc
conda activate Tperturb
cwd=$(pwd)
export WANDB_DIR=$cwd/wandb

wandb agent irene-bp/ttransformer/vcml7k3y
