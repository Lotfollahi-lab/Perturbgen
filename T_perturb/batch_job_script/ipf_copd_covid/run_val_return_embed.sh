#!/bin/bash
#BSUB -q gpu-lotfollahi # name of the partition to run job on (options: gpu-normal, gpu-huge, gpu-lotfollahi)
#BSUB -gpu 'mode=exclusive_process:num=1:block=yes' # request for exclusive access to gpu
#BSUB -n 16 # number of cores
#BSUB -G teamtrynka # groupname for billing
#BSUB -cwd /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb # working directory
#BSUB -o logs/ipf_copd_embed_%J.out # output file
#BSUB -e logs/ipf_copd_embed_%J.err # error file
#BSUB -M 50000  # RAM memory part 2. Default: 100MB
#BSUB -R 'span[ptile=16]'  # Allocate 4 CPU cores per node
#BSUB -R 'select[mem>50000] rusage[mem=50000]' # RAM memory part 1. Default: 100MB
#BSUB -J ipf_copd_embed # job name


# load cuda
module load cuda-12.1.1

# activate pyenv
source /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/.torch25/bin/activate
cwd=$(pwd)

# export WANDB_DIR=$cwd/wandb
# run script
echo "--- Start computing model"

# # Run python script for rna
# python3 $cwd/val.py \
python3 /lustre/scratch123/hgi/projects/healthy_imm_expr/t_generative/T_perturb/T_perturb/val.py \
--test_mode masking \
--split True \
--splitting_mode stratified \
--return_embed True \
--generate False \
--ckpt_masking_path "./T_perturb/T_perturb/Model/checkpoints/20240910_2111_cellgen_train_masking_lr_0.0001_wd_0.0001_batch_64_mlmp_0.15_ntask_3_s_100-epoch=19.ckpt" \
--output_dir "./CellGen-reproducibility/covid_ipf_copd/res" \
--src_dataset "./CellGen-reproducibility/covid_ipf_copd/processed_data/dataset_hvg_src/control.dataset" \
--tgt_dataset "./CellGen-reproducibility/covid_ipf_copd/processed_data/dataset_hvg_tgt/disease.dataset" \
--pairing_metadata "./CellGen-reproducibility/covid_ipf_copd/processed_data/metadata.pkl" \
--tokenid_to_genename_dict  "./CellGen-reproducibility/ipf_copd/tokenid_to_genename.pkl" \
--batch_size 48 \
--max_len 2048 \
--tgt_vocab_size 25426 \
--cellgen_lr 0.0001 \
--cellgen_wd 0.0001 \
--n_workers 16 \
--d_ff 128 \
--num_layers 1 \
--n_task_conditions 3 \
--var_list cell_type disease \
--encoder_type GF_frozen \
--moe_type moe_ffn \
--seed 100
echo "--- Finished computing model"
