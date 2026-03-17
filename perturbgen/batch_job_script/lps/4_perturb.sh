#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB :gmodel=NVIDIAA100_SXM4_80GB
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11 # working directory
#BSUB -o T_perturb/perturbgen/logs/perturb_IL1B_src_%J.out # output file
#BSUB -e T_perturb/perturbgen/logs/perturb_IL1B_src_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J lps_perturb_IL1B_src # job name

# load cuda
module load cuda-12.1.1
# activate conda environment
source /nfs/team361/cytomeister/.cytomeister/bin/activate

# activate pyenv
# source /lustre/scratch126/cellgen/team361/kl11/t_generative/.cellgen_4096/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen/Perturb/val.py \
--config /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen/configs/eval/LPS/mask_src_inference_perturbation_IL1B_90min.yaml
echo "--- Completed perturbation"