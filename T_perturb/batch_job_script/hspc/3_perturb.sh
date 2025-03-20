#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu :gmodel=NVIDIAA100_SXM4_80GB :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 8 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/ # working directory
#BSUB -o T_perturb/logs/perturb_cluster75_%J.out # output file
#BSUB -e T_perturb/logs/perturb_cluster75_%J.err # error file
#BSUB -M 80000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>80000] rusage[mem=80000]' # RAM memory part 1. Default: 100MB
#BSUB -J hspc_perturb_cluster75 # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/Perturb/val.py \
--config /lustre/scratch126/cellgen/team361/kl11/t_generative/T_perturb/T_perturb/configs/eval/HSPC/mask_src_inference_perturbation.yaml
echo "--- Completed perturbation"
