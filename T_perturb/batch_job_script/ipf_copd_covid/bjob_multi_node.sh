#!/bin/bash
#BSUB -q gpu-lotfollahi                             # name of the queue
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request gpus per host
#BSUB -G teamtrynka                                    # groupname for billing
#BSUB -o logs/ipf_copd_masking_%J.out # output file
#BSUB -e logs/ipf_copd_masking_%J.err # error file
#BSUB -M 10G                                         # RAM memory per host
#BSUB -R "select[mem>10G] rusage[mem=10G]"            # same as above
#BSUB -n 6                                          # number of cores in total
#BSUB -R "span[ptile=3]"                            # split X cores per host

## This example requests 6 cores in total, split 3 cores per host over 2 hosts. (2x2=4 GPU cores)

set -eo pipefail

# initialize the module system
. /usr/share/modules/init/bash
module load cuda-12.1.1
module load ISG/openmpi

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # disable infiniband to prevent annoying errors
export UCX_IB_MLX5_DEVX=n

# activate environment with the right dependencies
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.cellgen_torch25/bin/activate

# Get the number of hosts and GPUs from LSF
NUM_HOSTS=$(sed 's/ /\n/g' <<< $LSB_HOSTS  | sort | uniq | wc -l)
NUM_GPUS=$(bjobs -noheader -o 'gpu_num' "$LSB_JOBID")
GPU_PER_HOST=$((NUM_GPUS / NUM_HOSTS))

# Run the script with mpirun, requesting one process per GPU
mpirun \
    -n ${NUM_GPUS} \
    --map-by ppr:${GPU_PER_HOST}:node \
    --display-allocation \
    python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/train.py \
    --train_mode masking \
    --split True \
    --splitting_mode random \
    --output_dir "./T_perturb/T_perturb/plt/res/ipf_copd" \
    --src_dataset "./CellGen-reproducibility/covid_ipf_copd/processed_data/dataset_hvg_src/control.dataset" \
    --tgt_dataset "./CellGen-reproducibility/covid_ipf_copd/processed_data/dataset_hvg_tgt/disease.dataset" \
    --pairing_metadata "./CellGen-reproducibility/covid_ipf_copd/processed_data/metadata.pkl" \
    --batch_size 64 \
    --max_len 2048 \
    --epochs 20 \
    --tgt_vocab_size 25426 \
    --cellgen_lr 0.0001 \
    --cellgen_wd 0.0001 \
    --mlm_prob 0.15 \
    --n_workers 16 \
    --d_ff 128 \
    --num_layers 1 \
    --n_task_conditions 3 \
    --var_list cell_type disease \
    --encoder_type GF_frozen \
    --moe_type moe_ffn \
    --alpha 0.5 \
    --seed 100

# Also run a simple GPU check on all hosts
mpirun \
    -n ${NUM_GPUS} \
    --map-by ppr:${GPU_PER_HOST}:node \
    bash -c "nvidia-smi --query-gpu=name --format=csv,noheader | wc -l && nvidia-smi"

# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/.petra_cuda12/bin/activate
cwd=$(pwd)

export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# Interpolation
# python3 $cwd/train.py \

echo "--- Finished computing model"
echo "Done 🧙🧙"
