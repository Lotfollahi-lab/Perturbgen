#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1:gmodel=NVIDIAA100_SXM4_80GB" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -n 8 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/ # working directory
#BSUB -o T_perturb/logs/perturb_cluster_prkar2b_%J.out # output file
#BSUB -e T_perturb/logs/perturb_cluster_prkar2b_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J hspc_perturb_cluster_prkar2b # job name

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/Perturb/val.py \
--config T_perturb/T_perturb/configs/eval/HSPC/mask_src_inference_perturbation_prkar2b.yaml
echo "--- Completed perturbation"
