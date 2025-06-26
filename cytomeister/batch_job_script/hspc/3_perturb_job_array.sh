#!/bin/bash
#BSUB -q gpu-huge # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1:gmodel=NVIDIAA100_SXM4_80GB" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -J hspc_perturb_cluster_[1-35]%4 # job array with 35 jobs, max 4 running at the same time
#BSUB -n 4 # number of cores
#BSUB -G cellulargenetics-priority # groupname for billing team361
#BSUB -cwd /lustre/scratch126/cellgen/lotfollahi/kl11/ # working directory
#BSUB -o T_perturb/cytomeister/logs/perturb_cluster_%J_%I.out # output file
#BSUB -e T_perturb/cytomeister/logs/perturb_cluster_%J_%I.err # error file
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

GENES=(
  TGFBR2 SLC2A10 INHBA ZNF423 TGFB1I1 CDKN1C BMP10 SMAD3 CITED2 TGFB2 RBPJ
  BMP2 TSC22D1 SOX11 FOXC1 TGFB3 NOTCH1 DAB2 SMAD4 GDF7 BMPER HIPK2 TGFB1
  GDF11 HES5 ZEB2 CREBBP ENG BMPR2 THBS1 ACVR2A BMP4 INMG3 BMP6 CRB2
)
INDEX=$((LSB_JOBINDEX - 1))
PERTURBED_GENE=${GENES[$INDEX]}

echo "[$(date)] Starting job index $LSB_JOBINDEX"
echo "Current gene to be perturbed: $PERTURBED_GENE"

CONFIG_PATH="T_perturb/cytomeister/configs/eval/HSPC/mask_src_inference_perturbation_jobarray.yaml"
TMP_CONFIG_PATH="T_perturb/cytomeister/configs/eval/HSPC/mask_src_generate_perturbation_${PERTURBED_GENE}_${LSB_JOBID}.yaml"

sed "s/{{PERTURBED_GENE}}/${PERTURBED_GENE}/g" $CONFIG_PATH > $TMP_CONFIG_PATH


# python3 $cwd/train.py \
python3 /lustre/scratch126/cellgen/lotfollahi/kl11/T_perturb/cytomeister/Perturb/val.py \
--config $TMP_CONFIG_PATH || {
  echo "[$(date)] Job index $LSB_JOBINDEX failed for gene $PERTURBED_GENE"
  exit 1
}

# Clean up temporary config file
rm -f $TMP_CONFIG_PATH
