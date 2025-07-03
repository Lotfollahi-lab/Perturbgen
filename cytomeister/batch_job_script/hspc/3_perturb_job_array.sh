#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu "mode=exclusive_process:num=1" # request for exclusive access to gpu  :gmodel=NVIDIA_H100_HBM3_80GB
#BSUB -J hspc_perturb_cluster_[1-48]%4 # job array with 35 jobs, max 4 running at the same time
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
HK2 ADORA2B GFPT1 FKBP4 GOT2 CYB5A VEGFA FUT8 P4HA1 GALE SLC25A13 HS2ST1 FAM162A ME2 PAXIP1 AURKA
DEPDC1 TPI1 CDK1 COPB2 PGM2 IDH1 PKM POLR3K PPP2CB PGAM1 PSMC4 GOT1 B4GALT4 HMMR RPE KIF2A AK4
LDHA SLC16A3 NASP BTK IGHM IGLL1 CD79A CD79B BLNK PIK3R1 PIK3CD RBM8A FLI1 ARPC1B MPIG6B
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
