#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -J hspc_perturb_cluster_[1-12]%4 # job array with 35 jobs, max 4 running at the same time
#BSUB -n 4 # number of cores
#BSUB -G team361 # groupname for billing team361
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/ # working directory
#BSUB -o T_perturb/perturbgen/logs/mask_TF_%J_%I.out # output file
#BSUB -e T_perturb/perturbgen/logs/mask_TF_%J_%I.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB

# load cuda
module load cuda-12.1.1

# activate pyenv
source /nfs/team361/cytomeister/.cytomeister/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

PERTURBED_GENE=$(sed -n "${LSB_JOBINDEX}p" $cwd/T_perturb/perturbgen/configs/eval/HSPC/perturb_TF.txt)

echo "[$(date)] Starting job index $LSB_JOBINDEX"
echo "Current gene to be perturbed: $PERTURBED_GENE"

CONFIG_PATH="T_perturb/perturbgen/configs/eval/HSPC/mask_src_inference_perturbation_jobarray.yaml"
TMP_CONFIG_PATH="T_perturb/perturbgen/configs/eval/HSPC/mask_src_inference_perturbation_${PERTURBED_GENE}_${LSB_JOBID}.yaml"

sed "s/{{PERTURBED_GENE}}/${PERTURBED_GENE}/g" $CONFIG_PATH > $TMP_CONFIG_PATH

if [[ -z "$PERTURBED_GENE" ]]; then
  echo "Error: LSB_JOBINDEX ($LSB_JOBINDEX) does not correspond to a valid gene in genes_to_perturb.txt" >&2
  exit 1
fi

# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/perturbgen/Perturb/val.py \
--config $TMP_CONFIG_PATH 

# Clean up temporary config file
trap "rm -f $TMP_CONFIG_PATH" EXIT
