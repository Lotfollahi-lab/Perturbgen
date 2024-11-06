#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=2:block=yes' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/ipf_copd_masking_%J.out # output file
#BSUB -e logs/ipf_copd_masking_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'span[ptile=16]'  # Allocate 4 CPU cores per node
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J ipf_copd_masking # job name

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
echo "--- Finished computing model"
